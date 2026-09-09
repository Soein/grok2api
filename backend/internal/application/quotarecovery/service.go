package quotarecovery

import (
	"context"
	"errors"
	"log/slog"
	"sync"
	"sync/atomic"
	"time"

	accountdomain "github.com/chenyme/grok2api/backend/internal/domain/account"
	"github.com/chenyme/grok2api/backend/internal/pkg/batch"
	"github.com/chenyme/grok2api/backend/internal/repository"
)

const (
	defaultRecoveryWorkers = 25
	recoveryClaimLease     = 2 * time.Minute
	recoveryProbeTimeout   = 30 * time.Second
	recoveryReconcileEvery = time.Minute
	recoveryReconcileLimit = 1000
	consoleProbeInterval   = 24 * time.Hour
)

type quotaSynchronizer interface {
	ProbeQuotaMode(ctx context.Context, accountID uint64, mode string) (accountdomain.QuotaWindow, error)
	ListDueQuotaWindows(ctx context.Context, now time.Time, limit int) ([]accountdomain.QuotaWindow, error)
}

type recoveryPolicySynchronizer interface {
	ProbeQuotaModeForRecovery(context.Context, uint64, string, time.Time, []accountdomain.Provider, bool) (accountdomain.QuotaWindow, error)
}

type recoveryPolicyLister interface {
	ListDueQuotaWindowsForRecovery(context.Context, time.Time, int, []accountdomain.Provider, bool) ([]accountdomain.QuotaWindow, error)
}

type recoveryPolicyCursorLister interface {
	ListDueQuotaWindowsForRecoveryAfter(context.Context, time.Time, int, *accountdomain.QuotaWindow, []accountdomain.Provider, bool) ([]accountdomain.QuotaWindow, error)
}

// Stats counts distinct reconciled/claimed candidates and claim outcomes.
// Scanned may exceed the execution limit; Claimed never does.
type Stats struct {
	Scanned, Claimed, Recovered, Failed, Skipped int
}

type Service struct {
	logger          *slog.Logger
	queue           repository.QuotaRecoveryQueue
	syncer          quotaSynchronizer
	mu              sync.RWMutex
	base            time.Duration
	max             time.Duration
	bulkPool        *batch.Pool
	running         atomic.Bool
	reconcileCursor *accountdomain.QuotaWindow
}

func NewService(logger *slog.Logger, queue repository.QuotaRecoveryQueue, syncer quotaSynchronizer, base, max time.Duration) *Service {
	return &Service{logger: logger, queue: queue, syncer: syncer, base: base, max: max, bulkPool: batch.NewPool(defaultRecoveryWorkers)}
}

func (s *Service) SetBulkPool(pool *batch.Pool) {
	if pool != nil {
		s.bulkPool = pool
	}
}

func (s *Service) UpdateConfig(base, max time.Duration) {
	s.mu.Lock()
	s.base, s.max = base, max
	s.mu.Unlock()
}

// RunBatch restores due queue entries and consumes at most limit claims across
// the allowed providers. Cancellation leaves unacknowledged claims to expire.
func (s *Service) RunBatch(ctx context.Context, now time.Time, limit, workers int, allowed []accountdomain.Provider, includeDisabled bool) (Stats, error) {
	var stats Stats
	if err := ctx.Err(); err != nil {
		return stats, err
	}
	if limit <= 0 || len(allowed) == 0 || !s.running.CompareAndSwap(false, true) {
		return stats, nil
	}
	defer s.running.Store(false)
	policy, ok := s.syncer.(recoveryPolicySynchronizer)
	if !ok {
		return stats, errors.New("quota recovery policy synchronizer is required")
	}
	limit = min(limit, 100)
	workers = max(1, min(workers, limit, 10))
	var windows []accountdomain.QuotaWindow
	var err error
	if lister, ok := s.syncer.(recoveryPolicyCursorLister); ok {
		windows, err = lister.ListDueQuotaWindowsForRecoveryAfter(ctx, now, recoveryReconcileLimit, s.reconcileCursor, allowed, includeDisabled)
		if err == nil && len(windows) == 0 && s.reconcileCursor != nil {
			s.reconcileCursor = nil
			windows, err = lister.ListDueQuotaWindowsForRecoveryAfter(ctx, now, recoveryReconcileLimit, nil, allowed, includeDisabled)
		}
	} else if lister, ok := s.syncer.(recoveryPolicyLister); ok {
		windows, err = lister.ListDueQuotaWindowsForRecovery(ctx, now, recoveryReconcileLimit, allowed, includeDisabled)
	} else {
		windows, err = s.syncer.ListDueQuotaWindows(ctx, now, recoveryReconcileLimit)
	}
	if err != nil {
		return stats, err
	}
	for _, window := range windows {
		if err := s.queue.EnsureQuotaRecovery(ctx, accountdomain.QuotaRecoveryEvent{AccountID: window.AccountID, Mode: window.Mode, DueAt: now}); err != nil {
			return stats, err
		}
	}
	if len(windows) == recoveryReconcileLimit {
		last := windows[len(windows)-1]
		s.reconcileCursor = &last
	} else {
		s.reconcileCursor = nil
	}
	values, err := s.queue.ClaimDueQuotaRecoveries(ctx, now, limit, recoveryClaimLease)
	if err != nil {
		return stats, err
	}
	stats.Scanned, stats.Claimed = len(windows), len(values)
	type candidateKey struct {
		id   uint64
		mode string
	}
	scanned := make(map[candidateKey]struct{}, len(windows))
	for _, window := range windows {
		scanned[candidateKey{window.AccountID, window.Mode}] = struct{}{}
	}
	for _, value := range values {
		if _, exists := scanned[candidateKey{value.AccountID, value.Mode}]; !exists {
			stats.Scanned++
		}
	}
	outcomes := make([]string, len(values))
	indices := make([]int, len(values))
	for index := range indices {
		indices[index] = index
	}
	results, _, runErr := batch.Run(ctx, indices, batch.Options{Workers: workers}, func(workCtx context.Context, index int) error {
		value := values[index]
		outcome, err := s.processOne(workCtx, now, value, func(probeCtx context.Context) (accountdomain.QuotaWindow, error) {
			return policy.ProbeQuotaModeForRecovery(probeCtx, value.AccountID, value.Mode, now, allowed, includeDisabled)
		})
		outcomes[index] = outcome
		return err
	})
	for index, result := range results {
		if result.Err != nil && !errors.Is(result.Err, context.Canceled) && !errors.Is(result.Err, context.DeadlineExceeded) {
			runErr = errors.Join(runErr, result.Err)
		}
		switch outcomes[index] {
		case "recovered":
			stats.Recovered++
		case "failed":
			stats.Failed++
		default:
			stats.Skipped++
		}
	}
	if ctx.Err() != nil {
		runErr = errors.Join(runErr, ctx.Err())
	}
	s.logger.Info("quota_recovery_batch", "scanned", stats.Scanned, "claimed", stats.Claimed, "recovered", stats.Recovered, "failed", stats.Failed, "skipped", stats.Skipped)
	return stats, runErr
}

func (s *Service) Run(ctx context.Context) {
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	reconcileTicker := time.NewTicker(recoveryReconcileEvery)
	defer reconcileTicker.Stop()
	s.reconcileDue(ctx, time.Now().UTC())
	for {
		select {
		case <-ctx.Done():
			return
		case now := <-ticker.C:
			s.runDue(ctx, now.UTC())
		case now := <-reconcileTicker.C:
			s.reconcileDue(ctx, now.UTC())
		}
	}
}

func (s *Service) runDue(ctx context.Context, now time.Time) {
	workers := s.bulkPool.Limit()
	values, err := s.queue.ClaimDueQuotaRecoveries(ctx, now, workers, recoveryClaimLease)
	if err != nil {
		s.logger.Warn("quota_recovery_claim_failed", "reason", "queue_unavailable")
		return
	}
	results, summary, runErr := batch.Run(ctx, values, batch.Options{Workers: workers, Pool: s.bulkPool}, func(workCtx context.Context, value accountdomain.QuotaRecoveryEvent) error {
		s.runOne(workCtx, now, value)
		return nil
	})
	for index, result := range results {
		if _, ok := result.Err.(*batch.PanicError); ok {
			s.logger.Error("quota_recovery_panicked", "account_id", values[index].AccountID, "mode", values[index].Mode, "reason", "worker_panic")
		}
	}
	if runErr != nil {
		s.logger.Warn("quota_recovery_batch_canceled", "submitted", summary.Submitted, "completed", summary.Completed)
	}
}

func (s *Service) runOne(ctx context.Context, now time.Time, value accountdomain.QuotaRecoveryEvent) {
	_, err := s.processOne(ctx, now, value, func(probeCtx context.Context) (accountdomain.QuotaWindow, error) {
		return s.syncer.ProbeQuotaMode(probeCtx, value.AccountID, value.Mode)
	})
	if err != nil && ctx.Err() == nil {
		s.logger.Warn("quota_recovery_queue_update_failed", "account_id", value.AccountID, "mode", value.Mode)
	}
}

func (s *Service) processOne(ctx context.Context, now time.Time, value accountdomain.QuotaRecoveryEvent, probe func(context.Context) (accountdomain.QuotaWindow, error)) (string, error) {
	probeCtx, cancel := context.WithTimeout(ctx, recoveryProbeTimeout)
	window, probeErr := probe(probeCtx)
	cancel()
	if ctx.Err() != nil {
		return "skipped", ctx.Err()
	}
	var skip interface{ QuotaRecoverySkip() (bool, time.Time) }
	if errors.As(probeErr, &skip) {
		permanent, retryAt := skip.QuotaRecoverySkip()
		if permanent {
			return "skipped", s.queue.AckQuotaRecovery(ctx, value)
		}
		if !retryAt.After(now) {
			retryAt = now.Add(s.backoff(1))
		}
		value.DueAt = retryAt
		return "skipped", s.queue.RescheduleQuotaRecovery(ctx, value)
	}
	if probeErr == nil && window.Remaining > 0 {
		if err := s.queue.AckQuotaRecovery(ctx, value); err != nil {
			return "failed", err
		}
		return "recovered", nil
	}
	value.Attempts++
	if probeErr == nil && window.ResetAt != nil && window.ResetAt.After(now) {
		value.DueAt = *window.ResetAt
	} else if probeErr == nil && (value.Mode == "console" || value.Mode == "console_image" || value.Mode == "console_video") {
		// Console usage currently exposes no reset timestamp. A healthy zero
		// result is therefore rechecked after the fixed 24-hour prediction
		// window; transport failures still use bounded exponential backoff.
		value.DueAt = now.Add(consoleProbeInterval)
	} else if probeErr == nil && window.WindowSeconds > 0 {
		// Preserve Provider-specific upstream window semantics. In particular,
		// Grok Web can report a duration without an absolute reset timestamp.
		value.DueAt = now.Add(time.Duration(window.WindowSeconds) * time.Second)
	} else {
		value.DueAt = now.Add(s.backoff(value.Attempts))
	}
	if err := s.queue.RescheduleQuotaRecovery(ctx, value); err != nil {
		return "failed", err
	}
	return "failed", nil
}

func (s *Service) reconcileDue(ctx context.Context, now time.Time) {
	windows, err := s.syncer.ListDueQuotaWindows(ctx, now, recoveryReconcileLimit)
	if err != nil {
		s.logger.Warn("quota_recovery_reconcile_failed", "reason", "storage_unavailable")
		return
	}
	for _, window := range windows {
		if err := s.queue.EnsureQuotaRecovery(ctx, accountdomain.QuotaRecoveryEvent{AccountID: window.AccountID, Mode: window.Mode, DueAt: now}); err != nil {
			s.logger.Warn("quota_recovery_reconcile_schedule_failed", "account_id", window.AccountID, "mode", window.Mode, "reason", "queue_unavailable")
		}
	}
}

func (s *Service) backoff(attempt int) time.Duration {
	s.mu.RLock()
	base, maximum := s.base, s.max
	s.mu.RUnlock()
	if base <= 0 {
		base = 30 * time.Second
	}
	if maximum < base {
		maximum = 30 * time.Minute
	}
	value := base
	for index := 1; index < attempt && value < maximum; index++ {
		value *= 2
	}
	if value > maximum {
		return maximum
	}
	return value
}
