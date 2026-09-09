package quotarecovery

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	accountdomain "github.com/chenyme/grok2api/backend/internal/domain/account"
	"github.com/chenyme/grok2api/backend/internal/infra/runtime/memory"
)

type recoveryPolicySyncStub struct {
	quotaSyncStub
	probe func(context.Context, uint64, string, time.Time, []accountdomain.Provider, bool) (accountdomain.QuotaWindow, error)
}

func TestRunBatchReconciliationPreservesExistingBackoff(t *testing.T) {
	ctx := context.Background()
	now := time.Now().UTC()
	queue := memory.NewQuotaRecoveryQueue()
	var calls int
	syncer := &recoveryPolicySyncStub{quotaSyncStub: quotaSyncStub{due: []accountdomain.QuotaWindow{{AccountID: 7, Mode: "fast", Remaining: 0, ResetAt: &now}}}, probe: func(context.Context, uint64, string, time.Time, []accountdomain.Provider, bool) (accountdomain.QuotaWindow, error) {
		calls++
		return accountdomain.QuotaWindow{}, errors.New("timeout")
	}}
	service := NewService(testLogger(), queue, syncer, 2*time.Minute, 30*time.Minute)
	allowed := []accountdomain.Provider{accountdomain.ProviderWeb}
	first, err := service.RunBatch(ctx, now, 1, 1, allowed, false)
	if err != nil || first.Claimed != 1 || first.Failed != 1 {
		t.Fatalf("first=%+v err=%v", first, err)
	}
	second, err := service.RunBatch(ctx, now.Add(time.Minute), 1, 1, allowed, false)
	if err != nil || second.Claimed != 0 || calls != 1 {
		t.Fatalf("reconciliation reset backoff: second=%+v calls=%d err=%v", second, calls, err)
	}
	third, err := service.RunBatch(ctx, now.Add(2*time.Minute), 1, 1, allowed, false)
	if err != nil || third.Claimed != 1 || calls != 2 {
		t.Fatalf("due retry missing: third=%+v calls=%d err=%v", third, calls, err)
	}
}

func TestRunBatchConsoleMediaUsesPredictedRecoveryWindow(t *testing.T) {
	for _, mode := range []string{"console_image", "console_video"} {
		t.Run(mode, func(t *testing.T) {
			now := time.Now().UTC()
			queue := &quotaQueueStub{claimed: []accountdomain.QuotaRecoveryEvent{{AccountID: 1, Mode: mode, DueAt: now}}}
			syncer := &recoveryPolicySyncStub{probe: func(context.Context, uint64, string, time.Time, []accountdomain.Provider, bool) (accountdomain.QuotaWindow, error) {
				return accountdomain.QuotaWindow{Mode: mode, Remaining: 0}, nil
			}}
			service := NewService(testLogger(), queue, syncer, time.Minute, time.Hour)
			_, err := service.RunBatch(context.Background(), now, 1, 1, []accountdomain.Provider{accountdomain.ProviderConsole}, false)
			if err != nil || len(queue.rescheduled) != 1 || !queue.rescheduled[0].DueAt.Equal(now.Add(24*time.Hour)) {
				t.Fatalf("rescheduled=%+v err=%v", queue.rescheduled, err)
			}
		})
	}
}

type recoveryCursorSyncStub struct {
	recoveryPolicySyncStub
	cursors []uint64
	now     time.Time
}

func (s *recoveryCursorSyncStub) ListDueQuotaWindowsForRecoveryAfter(_ context.Context, _ time.Time, limit int, after *accountdomain.QuotaWindow, _ []accountdomain.Provider, _ bool) ([]accountdomain.QuotaWindow, error) {
	if after != nil {
		s.cursors = append(s.cursors, after.AccountID)
		return []accountdomain.QuotaWindow{{AccountID: 1001, Mode: "fast", ResetAt: &s.now}}, nil
	}
	s.cursors = append(s.cursors, 0)
	windows := make([]accountdomain.QuotaWindow, limit)
	for i := range windows {
		windows[i] = accountdomain.QuotaWindow{AccountID: uint64(i + 1), Mode: "fast", ResetAt: &s.now}
	}
	return windows, nil
}

func TestRunBatchReconciliationCursorAdvancesAndWraps(t *testing.T) {
	now := time.Now().UTC()
	syncer := &recoveryCursorSyncStub{now: now}
	queue := &quotaQueueStub{}
	service := NewService(testLogger(), queue, syncer, time.Minute, time.Hour)
	for range 3 {
		stats, err := service.RunBatch(context.Background(), now, 10, 1, []accountdomain.Provider{accountdomain.ProviderWeb}, false)
		if err != nil || stats.Claimed != 0 {
			t.Fatalf("stats=%+v err=%v", stats, err)
		}
	}
	if len(syncer.cursors) != 3 || syncer.cursors[0] != 0 || syncer.cursors[1] != 1000 || syncer.cursors[2] != 0 {
		t.Fatalf("cursors=%v", syncer.cursors)
	}
	if len(queue.ensured) != 2001 {
		t.Fatalf("ensured=%d", len(queue.ensured))
	}
}

func (s *recoveryPolicySyncStub) ProbeQuotaModeForRecovery(ctx context.Context, id uint64, mode string, now time.Time, allowed []accountdomain.Provider, includeDisabled bool) (accountdomain.QuotaWindow, error) {
	return s.probe(ctx, id, mode, now, allowed, includeDisabled)
}

type recoverySkipStub struct {
	permanent bool
	retryAt   time.Time
}

func (e recoverySkipStub) Error() string                        { return "skipped" }
func (e recoverySkipStub) QuotaRecoverySkip() (bool, time.Time) { return e.permanent, e.retryAt }

func TestRunBatchSharesBoundedBudgetAndWorkers(t *testing.T) {
	now := time.Now().UTC()
	queue := &quotaQueueStub{}
	for id := uint64(1); id <= 12; id++ {
		queue.claimed = append(queue.claimed, accountdomain.QuotaRecoveryEvent{AccountID: id, Mode: "fast", DueAt: now})
	}
	var active, maximum atomic.Int32
	syncer := &recoveryPolicySyncStub{probe: func(ctx context.Context, id uint64, mode string, gotNow time.Time, allowed []accountdomain.Provider, includeDisabled bool) (accountdomain.QuotaWindow, error) {
		if len(allowed) != 2 || !includeDisabled || !gotNow.Equal(now) {
			t.Error("policy arguments not forwarded")
		}
		current := active.Add(1)
		defer active.Add(-1)
		for old := maximum.Load(); current > old && !maximum.CompareAndSwap(old, current); old = maximum.Load() {
		}
		time.Sleep(10 * time.Millisecond)
		return accountdomain.QuotaWindow{Remaining: 1}, nil
	}}
	service := NewService(testLogger(), queue, syncer, time.Second, time.Minute)
	stats, err := service.RunBatch(context.Background(), now, 10, 2, []accountdomain.Provider{accountdomain.ProviderWeb, accountdomain.ProviderConsole}, true)
	if err != nil || stats.Claimed != 10 || stats.Scanned != 10 || stats.Recovered != 10 || stats.Failed != 0 || stats.Skipped != 0 || queue.claimLimit != 10 || maximum.Load() != 2 {
		t.Fatalf("stats=%+v max=%d err=%v", stats, maximum.Load(), err)
	}
}

func TestRunBatchClassifiesRecoverySkipAndFailure(t *testing.T) {
	now := time.Now().UTC()
	queue := &quotaQueueStub{}
	for id := uint64(1); id <= 4; id++ {
		queue.claimed = append(queue.claimed, accountdomain.QuotaRecoveryEvent{AccountID: id, Mode: "fast", DueAt: now, Attempts: 2})
	}
	syncer := &recoveryPolicySyncStub{probe: func(_ context.Context, id uint64, _ string, _ time.Time, _ []accountdomain.Provider, _ bool) (accountdomain.QuotaWindow, error) {
		switch id {
		case 1:
			return accountdomain.QuotaWindow{Remaining: 1}, nil
		case 2:
			return accountdomain.QuotaWindow{}, recoverySkipStub{permanent: true}
		case 3:
			return accountdomain.QuotaWindow{}, recoverySkipStub{retryAt: now.Add(5 * time.Minute)}
		default:
			return accountdomain.QuotaWindow{}, errors.New("network timeout with secret data")
		}
	}}
	service := NewService(testLogger(), queue, syncer, 30*time.Second, 30*time.Minute)
	stats, err := service.RunBatch(context.Background(), now, 4, 1, []accountdomain.Provider{accountdomain.ProviderWeb}, false)
	if err != nil || stats != (Stats{Scanned: 4, Claimed: 4, Recovered: 1, Failed: 1, Skipped: 2}) || queue.acked != 2 || len(queue.rescheduled) != 2 {
		t.Fatalf("stats=%+v err=%v acked=%d rescheduled=%+v", stats, err, queue.acked, queue.rescheduled)
	}
	if queue.rescheduled[0].Attempts != 2 || !queue.rescheduled[0].DueAt.Equal(now.Add(5*time.Minute)) {
		t.Fatal("temporary skip must preserve attempts and cooldown")
	}
	if queue.rescheduled[1].Attempts != 3 || !queue.rescheduled[1].DueAt.Equal(now.Add(2*time.Minute)) {
		t.Fatal("transport failure must preserve exponential backoff")
	}
}

func TestRunBatchCancellationLeavesClaimsAndPreventsOverlap(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	now := time.Now().UTC()
	queue := &quotaQueueStub{claimed: []accountdomain.QuotaRecoveryEvent{{AccountID: 1, Mode: "fast", DueAt: now}}}
	started := make(chan struct{})
	syncer := &recoveryPolicySyncStub{probe: func(ctx context.Context, _ uint64, _ string, _ time.Time, _ []accountdomain.Provider, _ bool) (accountdomain.QuotaWindow, error) {
		close(started)
		<-ctx.Done()
		return accountdomain.QuotaWindow{}, ctx.Err()
	}}
	service := NewService(testLogger(), queue, syncer, time.Second, time.Minute)
	done := make(chan error, 1)
	go func() {
		_, err := service.RunBatch(ctx, now, 1, 1, []accountdomain.Provider{accountdomain.ProviderWeb}, false)
		done <- err
	}()
	<-started
	stats, err := service.RunBatch(context.Background(), now, 1, 1, []accountdomain.Provider{accountdomain.ProviderWeb}, false)
	if err != nil || stats.Claimed != 0 {
		t.Fatalf("overlapping batch claimed work: %+v %v", stats, err)
	}
	cancel()
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("cancel error=%v", err)
	}
	if queue.acked != 0 || len(queue.rescheduled) != 0 {
		t.Fatal("cancel must leave leases for expiry")
	}
}
