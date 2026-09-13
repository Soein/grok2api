package gateway

import (
	"context"
	"fmt"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/chenyme/grok2api/backend/internal/domain/account"
	"github.com/chenyme/grok2api/backend/internal/infra/persistence/relational"
	"github.com/chenyme/grok2api/backend/internal/infra/runtime/memory"
	"github.com/chenyme/grok2api/backend/internal/repository"
)

func setupTestBuildOAuthSelector(t *testing.T, dbName string) (*relational.Database, *relational.AccountRepository, *Selector, *Selector, account.Credential) {
	t.Helper()
	ctx := context.Background()
	db, err := relational.OpenSQLite(ctx, filepath.Join(t.TempDir(), dbName))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = db.Close() })
	if err := db.InitializeSchema(ctx); err != nil {
		t.Fatal(err)
	}

	repo := relational.NewAccountRepository(db)
	acc, _, err := repo.UpsertByIdentity(ctx, account.Credential{
		Provider:              account.ProviderBuild,
		Name:                  "build-test-account",
		SourceKey:             "build-test-account",
		EncryptedAccessToken:  "initial-token",
		EncryptedRefreshToken: "initial-refresh",
		ExpiresAt:             time.Now().UTC().Add(time.Hour),
		Enabled:               true,
		AuthStatus:            account.AuthStatusActive,
		MaxConcurrent:         8,
		Priority:              100,
	})
	if err != nil {
		t.Fatal(err)
	}

	selectorA := NewSelector(repo, memory.NewConcurrencyLimiter(), memory.NewStickyStore(), nil, time.Hour, time.Second, time.Minute)
	selectorB := NewSelector(repo, memory.NewConcurrencyLimiter(), memory.NewStickyStore(), nil, time.Hour, time.Second, time.Minute)

	// Invalidation bus dispatching to both selectors
	repo.SetInvalidationObserver(func(_ context.Context, ev repository.InvalidationEvent) {
		selectorA.ApplyInvalidation(ev)
		selectorB.ApplyInvalidation(ev)
	})

	return db, repo, selectorA, selectorB, acc
}

func TestTokenRefreshCacheSelectorWarmCachePreservedAndMaterialHydrated(t *testing.T) {
	ctx := context.Background()
	_, repo, selectorA, selectorB, acc := setupTestBuildOAuthSelector(t, "warm-cache.db")

	leaseA, err := selectorA.Acquire(ctx, account.ProviderBuild, 0, "grok-3", "", "", nil, false)
	if err != nil {
		t.Fatalf("selectorA Acquire failed: %v", err)
	}
	if leaseA.Credential.EncryptedAccessToken != "initial-token" {
		t.Fatalf("expected initial-token, got %s", leaseA.Credential.EncryptedAccessToken)
	}
	leaseA.Release()

	leaseB, err := selectorB.Acquire(ctx, account.ProviderBuild, 0, "grok-3", "", "", nil, false)
	if err != nil {
		t.Fatalf("selectorB Acquire failed: %v", err)
	}
	if leaseB.Credential.EncryptedAccessToken != "initial-token" {
		t.Fatalf("expected initial-token, got %s", leaseB.Credential.EncryptedAccessToken)
	}
	leaseB.Release()

	selectorA.candidateMu.Lock()
	candidatesA := len(selectorA.candidates)
	versionA := selectorA.baseProviderVersion[account.ProviderBuild]
	selectorA.candidateMu.Unlock()

	selectorB.candidateMu.Lock()
	candidatesB := len(selectorB.candidates)
	versionB := selectorB.baseProviderVersion[account.ProviderBuild]
	selectorB.candidateMu.Unlock()

	if candidatesA == 0 || candidatesB == 0 {
		t.Fatalf("expected warmed candidate caches, got A=%d, B=%d", candidatesA, candidatesB)
	}

	newExpiry := time.Now().UTC().Add(3 * time.Hour)
	updated, err := repo.UpdateTokens(ctx, acc.ID, "updated-access-v2", "updated-refresh-v2", newExpiry, 0)
	if err != nil {
		t.Fatalf("UpdateTokens failed: %v", err)
	}
	if updated.LastRefreshAt == nil {
		t.Fatal("expected non-nil LastRefreshAt from UpdateTokens")
	}
	if updated.RefreshDueAt == nil {
		t.Fatal("expected non-nil RefreshDueAt from UpdateTokens")
	}

	selectorA.candidateMu.Lock()
	newCandidatesA := len(selectorA.candidates)
	newVersionA := selectorA.baseProviderVersion[account.ProviderBuild]
	selectorA.candidateMu.Unlock()

	selectorB.candidateMu.Lock()
	newCandidatesB := len(selectorB.candidates)
	newVersionB := selectorB.baseProviderVersion[account.ProviderBuild]
	selectorB.candidateMu.Unlock()

	if newVersionA != versionA || newVersionB != versionB {
		t.Fatalf("base provider version advanced on ordinary renewal: A: %d -> %d, B: %d -> %d", versionA, newVersionA, versionB, newVersionB)
	}
	if newCandidatesA != candidatesA || newCandidatesB != candidatesB {
		t.Fatalf("candidate cache was wiped on ordinary renewal: A: %d -> %d, B: %d -> %d", candidatesA, newCandidatesA, candidatesB, newCandidatesB)
	}

	nextLeaseA, err := selectorA.Acquire(ctx, account.ProviderBuild, 0, "grok-3", "", "", nil, false)
	if err != nil {
		t.Fatalf("selectorA Acquire after renewal failed: %v", err)
	}
	defer nextLeaseA.Release()

	if nextLeaseA.Credential.EncryptedAccessToken != "updated-access-v2" {
		t.Fatalf("selectorA did not hydrate latest token: got %s, want updated-access-v2", nextLeaseA.Credential.EncryptedAccessToken)
	}
	if nextLeaseA.Credential.EncryptedRefreshToken != "updated-refresh-v2" {
		t.Fatalf("selectorA did not hydrate latest refresh token: got %s, want updated-refresh-v2", nextLeaseA.Credential.EncryptedRefreshToken)
	}
	if !nextLeaseA.Credential.ExpiresAt.Equal(newExpiry) {
		t.Fatalf("selectorA did not hydrate latest expiry: got %v, want %v", nextLeaseA.Credential.ExpiresAt, newExpiry)
	}
	if nextLeaseA.Credential.LastRefreshAt == nil || !nextLeaseA.Credential.LastRefreshAt.Equal(*updated.LastRefreshAt) {
		t.Fatalf("selectorA did not hydrate latest LastRefreshAt: got %v, want %v", nextLeaseA.Credential.LastRefreshAt, updated.LastRefreshAt)
	}
	if nextLeaseA.Credential.RefreshDueAt == nil || !nextLeaseA.Credential.RefreshDueAt.Equal(*updated.RefreshDueAt) {
		t.Fatalf("selectorA did not hydrate latest RefreshDueAt: got %v, want %v", nextLeaseA.Credential.RefreshDueAt, updated.RefreshDueAt)
	}

	nextLeaseB, err := selectorB.Acquire(ctx, account.ProviderBuild, 0, "grok-3", "", "", nil, false)
	if err != nil {
		t.Fatalf("selectorB Acquire after renewal failed: %v", err)
	}
	defer nextLeaseB.Release()

	if nextLeaseB.Credential.EncryptedAccessToken != "updated-access-v2" {
		t.Fatalf("selectorB did not hydrate latest token: got %s, want updated-access-v2", nextLeaseB.Credential.EncryptedAccessToken)
	}
	if nextLeaseB.Credential.EncryptedRefreshToken != "updated-refresh-v2" {
		t.Fatalf("selectorB did not hydrate latest refresh token: got %s, want updated-refresh-v2", nextLeaseB.Credential.EncryptedRefreshToken)
	}
	if !nextLeaseB.Credential.ExpiresAt.Equal(newExpiry) {
		t.Fatalf("selectorB did not hydrate latest expiry: got %v, want %v", nextLeaseB.Credential.ExpiresAt, newExpiry)
	}
	if nextLeaseB.Credential.LastRefreshAt == nil || !nextLeaseB.Credential.LastRefreshAt.Equal(*updated.LastRefreshAt) {
		t.Fatalf("selectorB did not hydrate latest LastRefreshAt: got %v, want %v", nextLeaseB.Credential.LastRefreshAt, updated.LastRefreshAt)
	}
	if nextLeaseB.Credential.RefreshDueAt == nil || !nextLeaseB.Credential.RefreshDueAt.Equal(*updated.RefreshDueAt) {
		t.Fatalf("selectorB did not hydrate latest RefreshDueAt: got %v, want %v", nextLeaseB.Credential.RefreshDueAt, updated.RefreshDueAt)
	}
}

func TestTokenRefreshCacheSelectorRoutingMutationInvalidatesCache(t *testing.T) {
	ctx := context.Background()
	_, repo, selectorA, _, acc := setupTestBuildOAuthSelector(t, "mutation-invalidates.db")

	lease, err := selectorA.Acquire(ctx, account.ProviderBuild, 0, "grok-3", "", "", nil, false)
	if err != nil {
		t.Fatal(err)
	}
	lease.Release()

	selectorA.candidateMu.Lock()
	versionBefore := selectorA.baseProviderVersion[account.ProviderBuild]
	candidatesBefore := len(selectorA.candidates)
	selectorA.candidateMu.Unlock()

	if candidatesBefore == 0 {
		t.Fatal("expected cached candidates")
	}

	_, err = repo.UpdateTokens(ctx, acc.ID, "mutated-token", "mutated-refresh", time.Now().UTC().Add(time.Hour), 1)
	if err != nil {
		t.Fatal(err)
	}

	selectorA.candidateMu.Lock()
	versionAfter := selectorA.baseProviderVersion[account.ProviderBuild]
	candidatesAfter := len(selectorA.candidates)
	selectorA.candidateMu.Unlock()

	if versionAfter <= versionBefore {
		t.Fatalf("expected version to advance on routing mutation: before=%d, after=%d", versionBefore, versionAfter)
	}
	if candidatesAfter != 0 {
		t.Fatalf("expected candidate cache to be cleared on routing mutation, got %d", candidatesAfter)
	}
}

func TestTokenRefreshCacheSelectorConcurrentRefreshAndDisable(t *testing.T) {
	ctx := context.Background()
	_, repo, selectorA, _, acc := setupTestBuildOAuthSelector(t, "concurrent-disable.db")

	const numRefreshIterations = 30
	writeErrCh := make(chan error, numRefreshIterations+1)

	var wg sync.WaitGroup
	start := make(chan struct{})

	wg.Add(1)
	go func() {
		defer wg.Done()
		<-start
		for i := 0; i < numRefreshIterations; i++ {
			_, err := repo.UpdateTokens(ctx, acc.ID, "concurrent-token", "concurrent-refresh", time.Now().UTC().Add(time.Hour), 0)
			if err != nil {
				writeErrCh <- fmt.Errorf("refresh iter %d: %w", i, err)
			}
			time.Sleep(time.Millisecond)
		}
	}()

	wg.Add(1)
	go func() {
		defer wg.Done()
		<-start
		time.Sleep(10 * time.Millisecond)
		disabled := false
		_, err := repo.UpdateMany(ctx, account.ProviderBuild, []uint64{acc.ID}, repository.AccountUpdates{Enabled: &disabled})
		if err != nil {
			writeErrCh <- fmt.Errorf("disable: %w", err)
		}
	}()

	const numAcquirers = 4
	for a := 0; a < numAcquirers; a++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			for i := 0; i < 30; i++ {
				lease, err := selectorA.Acquire(ctx, account.ProviderBuild, 0, "grok-3", "", "", nil, false)
				if err == nil && lease != nil {
					if !lease.Credential.Enabled {
						t.Errorf("selector acquired a disabled account!")
					}
					lease.Release()
				}
				time.Sleep(time.Millisecond)
			}
		}()
	}

	close(start)
	wg.Wait()
	close(writeErrCh)

	for err := range writeErrCh {
		t.Fatalf("concurrent write failed: %v", err)
	}

	stored, err := repo.Get(ctx, acc.ID)
	if err != nil {
		t.Fatalf("failed to read back account: %v", err)
	}
	if stored.Enabled {
		t.Fatalf("expected account Enabled=false after disable, got true")
	}

	lease, err := selectorA.Acquire(ctx, account.ProviderBuild, 0, "grok-3", "", "", nil, false)
	if err == nil && lease != nil {
		lease.Release()
		t.Fatal("expected no account available after disable")
	}
}
