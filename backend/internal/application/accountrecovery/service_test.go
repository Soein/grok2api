package accountrecovery

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/chenyme/grok2api/backend/internal/application/quotarecovery"
	"github.com/chenyme/grok2api/backend/internal/domain/account"
	"github.com/chenyme/grok2api/backend/internal/infra/config"
	"github.com/chenyme/grok2api/backend/internal/infra/runtime/memory"
)

type testBackend struct {
	mu            sync.Mutex
	ids           []uint64
	reauth        []uint64
	current, peak atomic.Int32
	entered       chan struct{}
	proceed       chan struct{}
	fail          bool
	providers     []account.Provider
}

func (b *testBackend) ListDueBuildQuotaRecoveryIDs(_ context.Context, _ time.Time, _ bool, limit int) ([]uint64, error) {
	return b.ids[:min(len(b.ids), limit)], nil
}
func (b *testBackend) ListBuildReauthCandidates(_ context.Context, _ time.Time, _ bool, limit int) ([]uint64, error) {
	return b.reauth[:min(len(b.reauth), limit)], nil
}
func (b *testBackend) ProbeBuildQuotaRecovery(ctx context.Context, _ uint64, _ []string, _ bool) (account.RecoveryResult, error) {
	n := b.current.Add(1)
	defer b.current.Add(-1)
	for old := b.peak.Load(); n > old && !b.peak.CompareAndSwap(old, n); old = b.peak.Load() {
	}
	if b.entered != nil {
		select {
		case b.entered <- struct{}{}:
		default:
		}
	}
	if b.proceed != nil {
		select {
		case <-b.proceed:
		case <-ctx.Done():
			return account.RecoveryResult{Claimed: true, Reason: "canceled"}, ctx.Err()
		}
	}
	if b.fail {
		return account.RecoveryResult{Claimed: true, Reason: "transport"}, errors.New("secret-should-not-appear")
	}
	return account.RecoveryResult{Claimed: true, Recovered: true}, nil
}
func (b *testBackend) RecoverBuildAuthentication(ctx context.Context, id uint64, _ bool, _ time.Duration, _ time.Duration) (account.RecoveryResult, error) {
	return b.ProbeBuildQuotaRecovery(ctx, id, nil, true)
}
func (b *testBackend) RunBatch(_ context.Context, _ time.Time, limit, workers int, allowed []account.Provider, _ bool) (quotarecovery.Stats, error) {
	b.mu.Lock()
	b.providers = append(b.providers, allowed...)
	b.mu.Unlock()
	return quotarecovery.Stats{}, nil
}

func testConfig() config.AccountRecoveryConfig {
	return config.AccountRecoveryConfig{Enabled: true, Interval: config.Duration(time.Minute), BatchSize: 3, Concurrency: 1, ProbeTimeout: config.Duration(10 * time.Second), Build: true, ReauthBatchSize: 1, ReauthBackoffBase: config.Duration(time.Hour), ReauthBackoffMax: config.Duration(24 * time.Hour), BuildModels: []string{"grok-4.5"}}
}
func testService(cfg config.AccountRecoveryConfig, b *testBackend) *Service {
	return NewService(slog.New(slog.NewTextHandler(io.Discard, nil)), cfg, b, b, b, b, memory.NewLockStore())
}

func TestPatrolDefaultOffDoesNotTouchDependencies(t *testing.T) {
	cfg := testConfig()
	cfg.Enabled = false
	s := NewService(slog.Default(), cfg, nil, nil, nil, nil, nil)
	stats, err := s.RunOnce(context.Background())
	if err != nil || stats.Scanned != 0 {
		t.Fatalf("disabled stats=%+v err=%v", stats, err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	s.Run(ctx)
}

func TestPatrolSharesBatchBudgetAcrossQuotaAndAuthentication(t *testing.T) {
	b := &testBackend{ids: []uint64{1, 2, 3, 4, 5}, reauth: []uint64{7, 8}}
	cfg := testConfig()
	cfg.SSOReauth = true
	s := testService(cfg, b)
	for range 4 {
		stats, err := s.RunOnce(context.Background())
		if err != nil || stats.Scanned > 3 || stats.Claimed > 3 || stats.Recovered != stats.Claimed {
			t.Fatalf("stats=%+v err=%v", stats, err)
		}
	}
	if b.peak.Load() > 1 {
		t.Fatalf("peak concurrency=%d", b.peak.Load())
	}
}

func TestPatrolDoesNotOverlapAndCancelsInflightWork(t *testing.T) {
	b := &testBackend{ids: []uint64{1, 2, 3}, entered: make(chan struct{}, 1), proceed: make(chan struct{})}
	s := testService(testConfig(), b)
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { _, err := s.RunOnce(ctx); done <- err }()
	select {
	case <-b.entered:
	case <-time.After(3 * time.Second):
		t.Fatal("probe did not start")
	}
	if _, err := s.RunOnce(context.Background()); !errors.Is(err, ErrBusy) {
		t.Fatalf("overlap err=%v", err)
	}
	cancel()
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("cancellation did not stop patrol")
	}
	if b.current.Load() != 0 {
		t.Fatal("probe leaked")
	}
	close(b.proceed)
	if _, err := s.RunOnce(context.Background()); err != nil {
		t.Fatal("round lock leaked", err)
	}
}

func TestPatrolProviderSwitchesAndFailureStats(t *testing.T) {
	b := &testBackend{ids: []uint64{1}, fail: true}
	cfg := testConfig()
	cfg.Web = true
	cfg.Console = false
	stats, err := testService(cfg, b).RunOnce(context.Background())
	if err != nil || stats.Scanned != 1 || stats.Claimed != 1 || stats.Failed != 1 || stats.Recovered != 0 {
		t.Fatalf("stats=%+v err=%v", stats, err)
	}
	if len(b.providers) != 1 || b.providers[0] != account.ProviderWeb {
		t.Fatalf("providers=%v", b.providers)
	}
}

func TestPatrolSharedLockPreventsTwoSchedulers(t *testing.T) {
	b := &testBackend{ids: []uint64{1}, entered: make(chan struct{}, 1), proceed: make(chan struct{})}
	lock := memory.NewLockStore()
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	one := NewService(logger, testConfig(), b, b, b, b, lock)
	two := NewService(logger, testConfig(), b, b, b, b, lock)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan struct{})
	go func() { defer close(done); _, _ = one.RunOnce(ctx) }()
	select {
	case <-b.entered:
	case <-time.After(3 * time.Second):
		t.Fatal("not started")
	}
	if _, err := two.RunOnce(context.Background()); !errors.Is(err, ErrBusy) {
		t.Fatalf("second scheduler=%v", err)
	}
	cancel()
	<-done
}
