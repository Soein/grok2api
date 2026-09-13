package gateway

import (
	"context"
	"errors"
	"path/filepath"
	"testing"
	"time"

	"github.com/chenyme/grok2api/backend/internal/domain/account"
	"github.com/chenyme/grok2api/backend/internal/infra/egress"
	"github.com/chenyme/grok2api/backend/internal/infra/persistence/relational"
	"github.com/chenyme/grok2api/backend/internal/infra/runtime/memory"
	"github.com/chenyme/grok2api/backend/internal/repository"
)

func TestSelectorChurnUsesAuthoritativeResultAfterFirstConflict(t *testing.T) {
	for _, stage := range []string{"after_load", "before_store"} {
		t.Run(stage, func(t *testing.T) {
			repo := newLayeredRepositoryFixture()
			// The original layered candidate was revoked while its snapshot loaded.
			repo.combined = nil
			selector := NewSelector(repo, nil, nil, nil, time.Hour, time.Second, time.Minute)
			ctx, timing := egress.WithPreflightTiming(context.Background(), time.Now())
			invalidate := func() {
				selector.ApplyInvalidation(repository.InvalidationEvent{Kind: repository.InvalidationAccountStateChanged, Provider: account.ProviderBuild})
			}
			if stage == "after_load" {
				repo.baseHook = invalidate
			} else {
				// Hold filtering until assembly completed, then revoke the snapshot.
				selector.configMu.Lock()
			}
			done := make(chan error, 1)
			go func() {
				_, err := selector.beginSelectionSession(ctx, account.ProviderBuild, 0, "model-a", "", "", nil, false)
				done <- err
			}()
			if stage == "before_store" {
				deadline := time.Now().Add(5 * time.Second)
				for timing.Snapshot().Stages["candidate_assemble"].Count == 0 && time.Now().Before(deadline) {
					time.Sleep(time.Millisecond)
				}
				assembled := timing.Snapshot().Stages["candidate_assemble"].Count
				invalidate()
				selector.configMu.Unlock()
				if assembled != 1 {
					t.Fatal("assembly did not reach the publication boundary")
				}
			}
			select {
			case err := <-done:
				var unavailable *SelectionUnavailableError
				if !errors.As(err, &unavailable) || unavailable.Reason != SelectionNoAccounts {
					t.Fatalf("revoked candidate remained selectable: %v", err)
				}
			case <-time.After(5 * time.Second):
				t.Fatal("selection did not finish")
			}
			if repo.baseCalls != 1 || repo.combinedCalls != 1 {
				t.Fatalf("amplified full loads: base=%d combined=%d", repo.baseCalls, repo.combinedCalls)
			}
			if len(selector.candidates) != 0 {
				t.Fatal("fallback was cached despite unstable versions")
			}
			if timing.Snapshot().Counters["candidate_version_retry_"+stage] != 1 {
				t.Fatalf("wrong conflict attribution: %+v", timing.Snapshot())
			}
		})
	}
}

func TestSelectorModelQuotaInvalidatesOverlayWithoutRebuildingBase(t *testing.T) {
	ctx := context.Background()
	db, err := relational.OpenSQLite(ctx, filepath.Join(t.TempDir(), "quota-scope.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if err = db.InitializeSchema(ctx); err != nil {
		t.Fatal(err)
	}
	repo := relational.NewAccountRepository(db)
	value, _, err := repo.UpsertByIdentity(ctx, account.Credential{Provider: account.ProviderBuild, Name: "scoped", SourceKey: "scoped", Enabled: true, AuthStatus: account.AuthStatusActive, EncryptedAccessToken: "encrypted", ExpiresAt: time.Now().Add(time.Hour), MaxConcurrent: 1})
	if err != nil {
		t.Fatal(err)
	}
	selector := NewSelector(repo, memory.NewConcurrencyLimiter(), memory.NewStickyStore(), nil, time.Hour, time.Second, time.Minute)
	// The selector must invalidate locally even when no observer is installed.
	if _, err = selector.beginSelectionSession(ctx, account.ProviderBuild, 0, "model-a", "", "", nil, false); err != nil {
		t.Fatal(err)
	}
	version := selector.routingBaseVersion(account.ProviderBuild)
	selector.MarkModelQuotaExhausted(ctx, value, nil, "model-a", time.Hour)
	if selector.routingBaseVersion(account.ProviderBuild) != version {
		t.Fatal("model quota unnecessarily invalidated the account base")
	}
	if _, err = selector.beginSelectionSession(ctx, account.ProviderBuild, 0, "model-a", "", "", nil, false); err == nil {
		t.Fatal("exhausted model remained selectable")
	}
	if _, err = selector.beginSelectionSession(ctx, account.ProviderBuild, 0, "model-b", "", "", nil, false); err != nil {
		t.Fatalf("unrelated model became unavailable: %v", err)
	}
}

func TestSelectorLinkReconciliationInvalidatesPeerProviderCaches(t *testing.T) {
	ctx := context.Background()
	db, err := relational.OpenSQLite(ctx, filepath.Join(t.TempDir(), "link-scope.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if err = db.InitializeSchema(ctx); err != nil {
		t.Fatal(err)
	}
	repo := relational.NewAccountRepository(db)
	var webID uint64
	for _, provider := range []account.Provider{account.ProviderBuild, account.ProviderConsole, account.ProviderWeb} {
		value, _, err := repo.UpsertByIdentity(ctx, account.Credential{Provider: provider, Name: string(provider), SourceKey: string(provider), UserID: "shared-owner", Enabled: true, AuthStatus: account.AuthStatusActive, EncryptedAccessToken: "encrypted", ExpiresAt: time.Now().Add(time.Hour), MaxConcurrent: 1})
		if err != nil {
			t.Fatal(err)
		}
		if provider == account.ProviderWeb {
			webID = value.ID
		}
	}
	selector := NewSelector(repo, nil, nil, nil, time.Hour, time.Second, time.Minute)
	for _, provider := range []account.Provider{account.ProviderBuild, account.ProviderConsole} {
		if _, err := selector.loadCandidates(ctx, provider, 0, "model-a", "", time.Now()); err != nil {
			t.Fatal(err)
		}
	}
	beforeBuild := selector.routingBaseVersion(account.ProviderBuild)
	beforeConsole := selector.routingBaseVersion(account.ProviderConsole)
	repo.SetInvalidationObserver(func(_ context.Context, event repository.InvalidationEvent) { selector.ApplyInvalidation(event) })
	if err = repo.ReconcileProviderLinks(ctx, webID); err != nil {
		t.Fatal(err)
	}
	if selector.routingBaseVersion(account.ProviderBuild) == beforeBuild || selector.routingBaseVersion(account.ProviderConsole) == beforeConsole {
		t.Fatal("link change only invalidated its initiating Web account, leaving peer caches stale")
	}
}
