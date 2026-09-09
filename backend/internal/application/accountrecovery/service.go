// Package accountrecovery schedules bounded quota and credential recovery using
// the existing provider state machines and request-side atomic claims.
package accountrecovery

import (
	"context"
	"errors"
	"log/slog"
	"sync/atomic"
	"time"

	"github.com/chenyme/grok2api/backend/internal/application/quotarecovery"
	"github.com/chenyme/grok2api/backend/internal/domain/account"
	"github.com/chenyme/grok2api/backend/internal/infra/config"
	"github.com/chenyme/grok2api/backend/internal/pkg/batch"
	"github.com/chenyme/grok2api/backend/internal/repository"
)

var ErrBusy = errors.New("account recovery patrol already running")

type quotaStore interface {
	ListDueBuildQuotaRecoveryIDs(context.Context, time.Time, bool, int) ([]uint64, error)
}

type authenticationRecoverer interface {
	ListBuildReauthCandidates(context.Context, time.Time, bool, int) ([]uint64, error)
	RecoverBuildAuthentication(context.Context, uint64, bool, time.Duration, time.Duration) (account.RecoveryResult, error)
}

type buildProber interface {
	ProbeBuildQuotaRecovery(context.Context, uint64, []string, bool) (account.RecoveryResult, error)
}

type windowRecoverer interface {
	RunBatch(context.Context, time.Time, int, int, []account.Provider, bool) (quotarecovery.Stats, error)
}

type activationStore interface {
	ListPendingRecoveryActivationIDs(context.Context, int) ([]uint64, error)
	PrepareRecoveredActivationQuota(context.Context, uint64, time.Time) (bool, error)
}

// Stats distinguishes recovered credentials from restored quota; neither count
// implies that an explicitly disabled account was enabled.
type Stats struct {
	Scanned, Claimed, Recovered, Failed, Skipped int
	QuotaRecovered, Reauthenticated              int
}

type Service struct {
	logger  *slog.Logger
	cfg     config.AccountRecoveryConfig
	store   quotaStore
	auth    authenticationRecoverer
	build   buildProber
	windows windowRecoverer
	lock    repository.DistributedLock
	running atomic.Bool
	phase   uint64
}

func NewService(logger *slog.Logger, cfg config.AccountRecoveryConfig, store quotaStore, auth authenticationRecoverer, build buildProber, windows windowRecoverer, lock repository.DistributedLock) *Service {
	return &Service{logger: logger, cfg: cfg, store: store, auth: auth, build: build, windows: windows, lock: lock}
}

// Run waits for the current round before starting another. The first round is
// immediate; subsequent rounds wait the configured interval after completion.
func (s *Service) Run(ctx context.Context) {
	if !s.cfg.Enabled {
		<-ctx.Done()
		return
	}
	timer := time.NewTimer(0)
	defer timer.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-timer.C:
		}
		if _, err := s.RunOnce(ctx); err != nil && ctx.Err() == nil && !errors.Is(err, ErrBusy) {
			s.logger.Warn("account_recovery_patrol_failed", "reason", "round_failed")
		}
		if ctx.Err() != nil {
			return
		}
		timer.Reset(s.cfg.Interval.Value())
	}
}

// RunOnce applies one shared account budget across provider windows, Build
// quota probes and SSO reauthorization. A distributed round lease complements
// each account's own request-side lease.
func (s *Service) RunOnce(ctx context.Context) (stats Stats, err error) {
	if !s.cfg.Enabled {
		return stats, nil
	}
	if err := ctx.Err(); err != nil {
		return stats, err
	}
	if !s.running.CompareAndSwap(false, true) {
		return stats, ErrBusy
	}
	defer s.running.Store(false)
	budget := max(s.cfg.ProbeTimeout.Value(), min(s.cfg.Interval.Value(), 5*time.Minute))
	runCtx, cancel := context.WithTimeout(ctx, budget)
	defer cancel()
	if s.lock == nil {
		return stats, errors.New("account recovery lock is unavailable")
	}
	release, acquired, err := s.lock.Acquire(runCtx, "account-recovery-patrol", budget+time.Minute)
	if err != nil {
		return stats, err
	}
	if !acquired {
		return stats, ErrBusy
	}
	defer release()
	defer func() {
		s.logger.Info("account_recovery_patrol", "scanned", stats.Scanned, "claimed", stats.Claimed, "recovered", stats.Recovered, "failed", stats.Failed, "skipped", stats.Skipped, "quota_recovered", stats.QuotaRecovered, "credentials_recovered", stats.Reauthenticated, "interrupted", runCtx.Err() != nil)
	}()
	if s.cfg.Build {
		if store, ok := s.store.(activationStore); ok {
			ids, loadErr := store.ListPendingRecoveryActivationIDs(runCtx, s.cfg.BatchSize)
			if loadErr != nil {
				return stats, loadErr
			}
			for _, id := range ids {
				if _, prepareErr := store.PrepareRecoveredActivationQuota(runCtx, id, time.Now().UTC()); prepareErr != nil {
					return stats, prepareErr
				}
			}
		}
	}
	remaining := s.cfg.BatchSize
	start := int(s.phase % 3)
	s.phase++
	for offset := 0; offset < 3 && remaining > 0; offset++ {
		if err := runCtx.Err(); err != nil {
			return stats, err
		}
		now := time.Now().UTC()
		switch (start + offset) % 3 {
		case 0:
			if !s.cfg.Build {
				continue
			}
			ids, loadErr := s.store.ListDueBuildQuotaRecoveryIDs(runCtx, now, s.cfg.IncludeDisabled, remaining)
			if loadErr != nil {
				return stats, loadErr
			}
			part, batchErr := s.runAccounts(runCtx, ids, "quota", func(workCtx context.Context, id uint64) (account.RecoveryResult, error) {
				result, probeErr := s.build.ProbeBuildQuotaRecovery(workCtx, id, s.cfg.BuildModels, s.cfg.IncludeDisabled)
				if result.Skipped && !result.Claimed && workCtx.Err() == nil {
					if deferred, ok := s.store.(interface {
						DeferUnclaimedQuotaRecovery(context.Context, uint64, time.Time, time.Time) error
					}); ok {
						probeErr = errors.Join(probeErr, deferred.DeferUnclaimedQuotaRecovery(workCtx, id, now, now.Add(s.cfg.Interval.Value())))
					}
				}
				return result, probeErr
			})
			stats.add(part)
			remaining -= part.Claimed
			if batchErr != nil {
				return stats, batchErr
			}
		case 1:
			allowed := make([]account.Provider, 0, 2)
			if s.cfg.Web {
				allowed = append(allowed, account.ProviderWeb)
			}
			if s.cfg.Console {
				allowed = append(allowed, account.ProviderConsole)
			}
			if len(allowed) == 0 {
				continue
			}
			part, windowErr := s.windows.RunBatch(runCtx, now, remaining, s.cfg.Concurrency, allowed, s.cfg.IncludeDisabled)
			stats.add(Stats{Scanned: part.Scanned, Claimed: part.Claimed, Recovered: part.Recovered, Failed: part.Failed, Skipped: part.Skipped, QuotaRecovered: part.Recovered})
			remaining -= part.Claimed
			if windowErr != nil {
				return stats, windowErr
			}
		case 2:
			if !s.cfg.Build || !s.cfg.SSOReauth {
				continue
			}
			ids, loadErr := s.auth.ListBuildReauthCandidates(runCtx, now, s.cfg.IncludeDisabled, min(remaining, s.cfg.ReauthBatchSize))
			if loadErr != nil {
				return stats, loadErr
			}
			part, batchErr := s.runAccounts(runCtx, ids, "authentication", func(workCtx context.Context, id uint64) (account.RecoveryResult, error) {
				result, authErr := s.auth.RecoverBuildAuthentication(workCtx, id, s.cfg.IncludeDisabled, s.cfg.ReauthBackoffBase.Value(), s.cfg.ReauthBackoffMax.Value())
				if result.Recovered {
					if store, ok := s.store.(activationStore); ok {
						writeCtx, done := context.WithTimeout(context.WithoutCancel(workCtx), 5*time.Second)
						_, prepareErr := store.PrepareRecoveredActivationQuota(writeCtx, id, time.Now().UTC())
						done()
						authErr = errors.Join(authErr, prepareErr)
					}
				}
				return result, authErr
			})
			stats.add(part)
			remaining -= part.Claimed
			if batchErr != nil {
				return stats, batchErr
			}
		}
	}
	return stats, runCtx.Err()
}

func (s *Service) runAccounts(ctx context.Context, ids []uint64, kind string, work func(context.Context, uint64) (account.RecoveryResult, error)) (Stats, error) {
	stats := Stats{Scanned: len(ids)}
	results, _, err := batch.Map(ctx, ids, batch.Options{Workers: s.cfg.Concurrency}, func(workCtx context.Context, id uint64) (account.RecoveryResult, error) {
		probeCtx, cancel := context.WithTimeout(workCtx, s.cfg.ProbeTimeout.Value())
		defer cancel()
		return work(probeCtx, id)
	})
	for index, result := range results {
		if result.Value.Claimed {
			stats.Claimed++
		}
		switch {
		case result.Value.Recovered:
			stats.Recovered++
			if kind == "authentication" {
				stats.Reauthenticated++
			} else {
				stats.QuotaRecovered++
			}
		case result.Err != nil && ctx.Err() == nil:
			stats.Failed++
		default:
			stats.Skipped++
		}
		if result.Value.Claimed || result.Err != nil {
			s.logger.Info("account_recovery_result", "account_id", ids[index], "kind", kind, "recovered", result.Value.Recovered, "reason", result.Value.Reason)
		}
	}
	return stats, err
}

func (s *Stats) add(other Stats) {
	s.Scanned += other.Scanned
	s.Claimed += other.Claimed
	s.Recovered += other.Recovered
	s.Failed += other.Failed
	s.Skipped += other.Skipped
	s.QuotaRecovered += other.QuotaRecovered
	s.Reauthenticated += other.Reauthenticated
}
