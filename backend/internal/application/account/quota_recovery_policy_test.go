package account

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"path/filepath"
	"testing"
	"time"

	"github.com/chenyme/grok2api/backend/internal/application/quotarecovery"
	accountdomain "github.com/chenyme/grok2api/backend/internal/domain/account"
	"github.com/chenyme/grok2api/backend/internal/infra/persistence/relational"
	"github.com/chenyme/grok2api/backend/internal/infra/provider"
	"github.com/chenyme/grok2api/backend/internal/infra/runtime/memory"
)

type recoveryPolicyChangingLock struct{ change func() }

func (l recoveryPolicyChangingLock) Acquire(context.Context, string, time.Duration) (func(), bool, error) {
	l.change()
	return func() {}, true, nil
}

func TestQuotaRecoveryRebuildsLostQueueWithoutResetAt(t *testing.T) {
	ctx := context.Background()
	db, err := relational.OpenSQLite(ctx, filepath.Join(t.TempDir(), "recovery-restart.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = db.Close() })
	if err := db.InitializeSchema(ctx); err != nil {
		t.Fatal(err)
	}
	repo := relational.NewAccountRepository(db)
	value, _, err := repo.UpsertByIdentity(ctx, accountdomain.Credential{Provider: accountdomain.ProviderConsole, AuthType: accountdomain.AuthTypeSSO, Name: "console", SourceKey: "console", EncryptedAccessToken: "encrypted", Enabled: true, AuthStatus: accountdomain.AuthStatusActive})
	if err != nil {
		t.Fatal(err)
	}
	now := time.Now().UTC()
	old := now.Add(-25 * time.Hour)
	if err := repo.SaveQuotaWindows(ctx, value.ID, "", old, []accountdomain.QuotaWindow{{AccountID: value.ID, Mode: "console", Remaining: 0}}); err != nil {
		t.Fatal(err)
	}
	adapter := &rateLimitConsoleQuotaAdapter{remaining: 3}
	accounts := NewService(repo, nil, nil, nil, provider.NewRegistry(adapter), nil, memory.NewLockStore())
	// A new empty memory queue models the state after a process restart.
	queue := memory.NewQuotaRecoveryQueue()
	service := quotarecovery.NewService(slog.New(slog.NewTextHandler(io.Discard, nil)), queue, accounts, time.Minute, time.Hour)
	stats, err := service.RunBatch(ctx, now, 10, 1, []accountdomain.Provider{accountdomain.ProviderConsole}, false)
	if err != nil || stats.Claimed != 1 || stats.Recovered != 1 || adapter.calls.Load() != 1 {
		t.Fatalf("restart stats=%+v calls=%d err=%v", stats, adapter.calls.Load(), err)
	}
}

func TestQuotaRecoveryCredentialPolicy(t *testing.T) {
	now := time.Now().UTC()
	cooldown := now.Add(time.Minute)
	tests := []struct {
		name                     string
		provider                 accountdomain.Provider
		mode                     string
		enabled, includeDisabled bool
		auth                     accountdomain.AuthStatus
		cooldown                 *time.Time
		allowed                  []accountdomain.Provider
		skip                     bool
	}{
		{"web", accountdomain.ProviderWeb, "fast", true, false, accountdomain.AuthStatusActive, nil, []accountdomain.Provider{accountdomain.ProviderWeb}, false},
		{"console", accountdomain.ProviderConsole, "console_image", true, false, accountdomain.AuthStatusActive, nil, []accountdomain.Provider{accountdomain.ProviderConsole}, false},
		{"disabled", accountdomain.ProviderWeb, "fast", false, false, accountdomain.AuthStatusActive, nil, []accountdomain.Provider{accountdomain.ProviderWeb}, true},
		{"maintain disabled", accountdomain.ProviderWeb, "fast", false, true, accountdomain.AuthStatusActive, nil, []accountdomain.Provider{accountdomain.ProviderWeb}, false},
		{"reauth", accountdomain.ProviderWeb, "fast", true, true, accountdomain.AuthStatusReauthRequired, nil, []accountdomain.Provider{accountdomain.ProviderWeb}, true},
		{"cooldown", accountdomain.ProviderWeb, "fast", true, false, accountdomain.AuthStatusActive, &cooldown, []accountdomain.Provider{accountdomain.ProviderWeb}, true},
		{"provider switch", accountdomain.ProviderConsole, "console", true, false, accountdomain.AuthStatusActive, nil, []accountdomain.Provider{accountdomain.ProviderWeb}, true},
		{"build", accountdomain.ProviderBuild, "fast", true, true, accountdomain.AuthStatusActive, nil, []accountdomain.Provider{accountdomain.ProviderBuild}, true},
		{"wrong console mode", accountdomain.ProviderConsole, "weekly", true, false, accountdomain.AuthStatusActive, nil, []accountdomain.Provider{accountdomain.ProviderConsole}, true},
		{"wrong web mode", accountdomain.ProviderWeb, "console", true, false, accountdomain.AuthStatusActive, nil, []accountdomain.Provider{accountdomain.ProviderWeb}, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			value := accountdomain.Credential{Provider: tt.provider, Enabled: tt.enabled, AuthStatus: tt.auth, CooldownUntil: tt.cooldown}
			err := quotaRecoveryCredentialPolicy(value, tt.mode, now, tt.allowed, tt.includeDisabled)
			if (err != nil) != tt.skip {
				t.Fatalf("skip=%t err=%v", tt.skip, err)
			}
		})
	}
}

func TestQuotaRecoveryProbeChecksDueStateAndPreservesDisabled(t *testing.T) {
	ctx := context.Background()
	db, err := relational.OpenSQLite(ctx, filepath.Join(t.TempDir(), "recovery.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = db.Close() })
	if err := db.InitializeSchema(ctx); err != nil {
		t.Fatal(err)
	}
	repo := relational.NewAccountRepository(db)
	value, _, err := repo.UpsertByIdentity(ctx, accountdomain.Credential{Provider: accountdomain.ProviderConsole, AuthType: accountdomain.AuthTypeSSO, Name: "console", SourceKey: "console", EncryptedAccessToken: "encrypted", Enabled: false, AuthStatus: accountdomain.AuthStatusActive})
	if err != nil {
		t.Fatal(err)
	}
	value.Enabled = false
	if _, err := repo.Update(ctx, value); err != nil {
		t.Fatal(err)
	}
	now := time.Now().UTC()
	due := now.Add(-time.Minute)
	window := accountdomain.QuotaWindow{AccountID: value.ID, Mode: "console", Remaining: 0, ResetAt: &due}
	if err := repo.SaveQuotaWindows(ctx, value.ID, "", now, []accountdomain.QuotaWindow{window}); err != nil {
		t.Fatal(err)
	}
	adapter := &rateLimitConsoleQuotaAdapter{remaining: 5}
	service := NewService(repo, nil, nil, nil, provider.NewRegistry(adapter), nil, memory.NewLockStore())
	allowed := []accountdomain.Provider{accountdomain.ProviderConsole}
	if _, err := service.ProbeQuotaMode(ctx, value.ID, "console"); err == nil {
		t.Fatal("legacy probe accepted disabled account")
	}
	if _, err := service.ProbeQuotaModeForRecovery(ctx, value.ID, "console", now, allowed, false); err == nil {
		t.Fatal("disabled account accepted")
	}
	service.refreshLock = deniedQuotaRefreshLock{}
	if _, err := service.ProbeQuotaModeForRecovery(ctx, value.ID, "console", now, allowed, true); err == nil {
		t.Fatal("busy account accepted")
	}
	if adapter.calls.Load() != 0 {
		t.Fatal("ineligible probes reached upstream")
	}
	service.refreshLock = memory.NewLockStore()
	got, err := service.ProbeQuotaModeForRecovery(ctx, value.ID, "console", now, allowed, true)
	if err != nil || got.Remaining != 5 || adapter.calls.Load() != 1 {
		t.Fatalf("window=%+v calls=%d err=%v", got, adapter.calls.Load(), err)
	}
	stored, err := repo.Get(ctx, value.ID)
	if err != nil || stored.Enabled {
		t.Fatal("maintenance enabled disabled account")
	}
	if _, err := service.ProbeQuotaModeForRecovery(ctx, value.ID, "console", now, allowed, true); err == nil || adapter.calls.Load() != 1 {
		t.Fatal("already recovered account reprobed")
	}
	window.ResetAt = &now
	if err := repo.SaveQuotaWindows(ctx, value.ID, "", now, []accountdomain.QuotaWindow{window}); err != nil {
		t.Fatal(err)
	}
	adapter.err = context.DeadlineExceeded
	if _, err := service.ProbeQuotaModeForRecovery(ctx, value.ID, "console", now, allowed, true); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("timeout error=%v", err)
	}
	stored, err = repo.Get(ctx, value.ID)
	if err != nil || stored.AuthStatus != accountdomain.AuthStatusActive {
		t.Fatal("timeout marked authentication permanently invalid")
	}
	window.ResetAt = func() *time.Time { future := now.Add(time.Hour); return &future }()
	if err := repo.SaveQuotaWindows(ctx, value.ID, "", now, []accountdomain.QuotaWindow{window}); err != nil {
		t.Fatal(err)
	}
	before := adapter.calls.Load()
	if _, err := service.ProbeQuotaModeForRecovery(ctx, value.ID, "console", now, allowed, true); err == nil || adapter.calls.Load() != before {
		t.Fatal("future window probed")
	}
}

func TestQuotaRecoverySharesConsoleAndImagineRefreshLocks(t *testing.T) {
	for _, mode := range []string{"console", "console_image", "console_video"} {
		if got := quotaRecoveryRefreshLockKey(7, mode); got != consoleQuotaRefreshLockKey(7) {
			t.Fatalf("%s lock=%s", mode, got)
		}
	}
	for _, mode := range []string{accountdomain.QuotaModeWebImagePro, accountdomain.QuotaModeWebVideo720p, accountdomain.QuotaGroupWebImagine} {
		if got := quotaRecoveryRefreshLockKey(7, mode); got != "quota-refresh:7:"+accountdomain.QuotaGroupWebImagine {
			t.Fatalf("%s lock=%s", mode, got)
		}
	}
}

func TestQuotaRecoveryDoesNotOverlapBusinessRefreshAndRechecksPolicy(t *testing.T) {
	ctx := context.Background()
	db, err := relational.OpenSQLite(ctx, filepath.Join(t.TempDir(), "recovery-overlap.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = db.Close() })
	if err := db.InitializeSchema(ctx); err != nil {
		t.Fatal(err)
	}
	repo := relational.NewAccountRepository(db)
	value, _, err := repo.UpsertByIdentity(ctx, accountdomain.Credential{Provider: accountdomain.ProviderConsole, AuthType: accountdomain.AuthTypeSSO, Name: "console", SourceKey: "console", EncryptedAccessToken: "encrypted", Enabled: true, AuthStatus: accountdomain.AuthStatusActive})
	if err != nil {
		t.Fatal(err)
	}
	now := time.Now().UTC()
	past := now.Add(-time.Hour)
	old := now.Add(-time.Minute)
	windows := []accountdomain.QuotaWindow{{AccountID: value.ID, Mode: "console", Remaining: 0, ResetAt: &past}, {AccountID: value.ID, Mode: "console_image", Remaining: 0, ResetAt: &past}, {AccountID: value.ID, Mode: "console_video", Remaining: 0, ResetAt: &past}}
	if err := repo.SaveQuotaWindows(ctx, value.ID, "", old, windows); err != nil {
		t.Fatal(err)
	}
	adapter := &consoleQuotaSnapshotAdapter{fullStarted: make(chan struct{}, 1), fullRelease: make(chan struct{})}
	service := NewService(repo, nil, nil, nil, provider.NewRegistry(adapter), nil, memory.NewLockStore())
	service.QueueQuotaRefresh(value.ID, "console")
	request := <-service.quotaRefreshQueue
	done := make(chan struct{})
	go func() { service.runQuotaRefresh(ctx, request); close(done) }()
	select {
	case <-adapter.fullStarted:
	case <-time.After(time.Second):
		t.Fatal("business refresh did not start")
	}
	allowed := []accountdomain.Provider{accountdomain.ProviderConsole}
	for _, mode := range []string{"console", "console_image", "console_video"} {
		_, err := service.ProbeQuotaModeForRecovery(ctx, value.ID, mode, now, allowed, false)
		var skip *QuotaRecoverySkipError
		if !errors.As(err, &skip) || skip.Permanent || skip.Reason != "busy" {
			t.Fatalf("%s busy result=%v", mode, err)
		}
	}
	close(adapter.fullRelease)
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("business refresh did not finish")
	}
	if adapter.fullCalls.Load() != 1 {
		t.Fatal("background duplicated business refresh")
	}
	if err := repo.SaveQuotaWindows(ctx, value.ID, "", old, windows); err != nil {
		t.Fatal(err)
	}
	service.refreshLock = recoveryPolicyChangingLock{change: func() {
		value.Enabled = false
		if _, err := repo.Update(ctx, value); err != nil {
			t.Fatal(err)
		}
	}}
	_, err = service.ProbeQuotaModeForRecovery(ctx, value.ID, "console", now, allowed, false)
	var skip *QuotaRecoverySkipError
	if !errors.As(err, &skip) || skip.Reason != "disabled" || adapter.fullCalls.Load() != 1 {
		t.Fatalf("disabled during claim was probed: %v", err)
	}
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := service.ProbeQuotaModeForRecovery(canceled, value.ID, "console", now, allowed, true); !errors.Is(err, context.Canceled) || adapter.fullCalls.Load() != 1 {
		t.Fatalf("canceled probe result=%v", err)
	}
}
