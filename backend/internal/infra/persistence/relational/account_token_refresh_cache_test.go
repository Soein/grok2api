package relational

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/chenyme/grok2api/backend/internal/domain/account"
	"github.com/chenyme/grok2api/backend/internal/repository"
	"gorm.io/gorm"
)

func createTestBuildOAuthAccount(t *testing.T, db *Database, name string) account.Credential {
	t.Helper()
	repo := NewAccountRepository(db)
	acc, _, err := repo.UpsertByIdentity(context.Background(), account.Credential{
		Provider:              account.ProviderBuild,
		Name:                  name,
		SourceKey:             name,
		EncryptedAccessToken:  "initial-access-token",
		EncryptedRefreshToken: "initial-refresh-token",
		ExpiresAt:             time.Now().UTC().Add(time.Hour),
		Enabled:               true,
		AuthStatus:            account.AuthStatusActive,
		MaxConcurrent:         8,
		Priority:              100,
	})
	if err != nil {
		t.Fatalf("failed to create test account: %v", err)
	}
	return acc
}

func TestTokenRefreshCacheOrdinaryRenewalEmitsNoInvalidation(t *testing.T) {
	db := openTestDatabase(t)
	repo := NewAccountRepository(db)
	acc := createTestBuildOAuthAccount(t, db, "ordinary-build-account")

	var mu sync.Mutex
	var events []repository.InvalidationEvent
	repo.SetInvalidationObserver(func(_ context.Context, e repository.InvalidationEvent) {
		mu.Lock()
		defer mu.Unlock()
		events = append(events, e)
	})

	expires := time.Now().UTC().Add(2 * time.Hour)
	updated, err := repo.UpdateTokens(context.Background(), acc.ID, "new-access-token", "new-refresh-token", expires, 0)
	if err != nil {
		t.Fatalf("UpdateTokens failed: %v", err)
	}

	if updated.EncryptedAccessToken != "new-access-token" {
		t.Fatalf("access token not updated: %s", updated.EncryptedAccessToken)
	}
	if updated.EncryptedRefreshToken != "new-refresh-token" {
		t.Fatalf("refresh token not updated: %s", updated.EncryptedRefreshToken)
	}

	mu.Lock()
	defer mu.Unlock()
	if len(events) != 0 {
		t.Fatalf("ordinary renewal emitted invalidation events: %+v", events)
	}
}

func TestTokenRefreshCacheBotFlagBothDirectionsNotifies(t *testing.T) {
	db := openTestDatabase(t)
	repo := NewAccountRepository(db)
	acc := createTestBuildOAuthAccount(t, db, "bot-flag-test-account")

	tests := []struct {
		name       string
		setInitial int
		newFlag    int
	}{
		{"0 to 1", 0, 1},
		{"1 to 0", 1, 0},
		{"0 to 2", 0, 2},
		{"2 to 0", 2, 0},
		{"1 to 2", 1, 2},
		{"2 to 1", 2, 1},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if err := db.db.Model(&accountCredentialModel{}).Where("account_id = ?", acc.ID).Update("build_bot_flag_source", tc.setInitial).Error; err != nil {
				t.Fatal(err)
			}

			var mu sync.Mutex
			var events []repository.InvalidationEvent
			repo.SetInvalidationObserver(func(_ context.Context, e repository.InvalidationEvent) {
				mu.Lock()
				defer mu.Unlock()
				events = append(events, e)
			})

			expires := time.Now().UTC().Add(time.Hour)
			_, err := repo.UpdateTokens(context.Background(), acc.ID, "token-x", "refresh-x", expires, tc.newFlag)
			if err != nil {
				t.Fatalf("UpdateTokens failed: %v", err)
			}

			mu.Lock()
			defer mu.Unlock()
			if len(events) != 1 {
				t.Fatalf("expected 1 invalidation event for bot flag change %s, got %d: %+v", tc.name, len(events), events)
			}
			ev := events[0]
			if ev.Kind != repository.InvalidationAccountCredentialChanged || ev.Provider != account.ProviderBuild || ev.AccountID != acc.ID {
				t.Fatalf("unexpected invalidation event: %+v", ev)
			}
		})
	}
}

func TestTokenRefreshCacheNonActiveToActiveNotifies(t *testing.T) {
	db := openTestDatabase(t)
	repo := NewAccountRepository(db)
	acc := createTestBuildOAuthAccount(t, db, "non-active-test-account")

	if err := db.db.Model(&accountModel{}).Where("id = ?", acc.ID).Update("auth_status", string(account.AuthStatusReauthRequired)).Error; err != nil {
		t.Fatal(err)
	}

	var mu sync.Mutex
	var events []repository.InvalidationEvent
	repo.SetInvalidationObserver(func(_ context.Context, e repository.InvalidationEvent) {
		mu.Lock()
		defer mu.Unlock()
		events = append(events, e)
	})

	expires := time.Now().UTC().Add(time.Hour)
	_, err := repo.UpdateTokens(context.Background(), acc.ID, "token-active", "refresh-active", expires, 0)
	if err != nil {
		t.Fatalf("UpdateTokens failed: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	if len(events) != 1 {
		t.Fatalf("expected 1 invalidation event for reauthRequired -> active, got %d: %+v", len(events), events)
	}
	ev := events[0]
	if ev.Kind != repository.InvalidationAccountCredentialChanged || ev.Provider != account.ProviderBuild || ev.AccountID != acc.ID {
		t.Fatalf("unexpected invalidation event: %+v", ev)
	}
}

func TestTokenRefreshCacheNonemptyLastErrorToEmptyNotifies(t *testing.T) {
	db := openTestDatabase(t)
	repo := NewAccountRepository(db)
	acc := createTestBuildOAuthAccount(t, db, "last-error-test-account")

	if err := db.db.Model(&accountModel{}).Where("id = ?", acc.ID).Update("last_error", "temporary_gateway_timeout").Error; err != nil {
		t.Fatal(err)
	}

	var mu sync.Mutex
	var events []repository.InvalidationEvent
	repo.SetInvalidationObserver(func(_ context.Context, e repository.InvalidationEvent) {
		mu.Lock()
		defer mu.Unlock()
		events = append(events, e)
	})

	expires := time.Now().UTC().Add(time.Hour)
	_, err := repo.UpdateTokens(context.Background(), acc.ID, "token-cleared-err", "refresh-cleared-err", expires, 0)
	if err != nil {
		t.Fatalf("UpdateTokens failed: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	if len(events) != 1 {
		t.Fatalf("expected 1 invalidation event for last_error cleared, got %d: %+v", len(events), events)
	}
	ev := events[0]
	if ev.Kind != repository.InvalidationAccountCredentialChanged || ev.Provider != account.ProviderBuild || ev.AccountID != acc.ID {
		t.Fatalf("unexpected invalidation event: %+v", ev)
	}
}

func TestTokenRefreshCachePermanentFailureToClearedNotifies(t *testing.T) {
	db := openTestDatabase(t)
	repo := NewAccountRepository(db)
	acc := createTestBuildOAuthAccount(t, db, "perm-fail-test-account")

	if err := db.db.Model(&accountCredentialModel{}).Where("account_id = ?", acc.ID).Update("refresh_permanent", true).Error; err != nil {
		t.Fatal(err)
	}

	var mu sync.Mutex
	var events []repository.InvalidationEvent
	repo.SetInvalidationObserver(func(_ context.Context, e repository.InvalidationEvent) {
		mu.Lock()
		defer mu.Unlock()
		events = append(events, e)
	})

	expires := time.Now().UTC().Add(time.Hour)
	_, err := repo.UpdateTokens(context.Background(), acc.ID, "token-cleared-perm", "refresh-cleared-perm", expires, 0)
	if err != nil {
		t.Fatalf("UpdateTokens failed: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	if len(events) != 1 {
		t.Fatalf("expected 1 invalidation event for refresh_permanent cleared, got %d: %+v", len(events), events)
	}
	ev := events[0]
	if ev.Kind != repository.InvalidationAccountCredentialChanged || ev.Provider != account.ProviderBuild || ev.AccountID != acc.ID {
		t.Fatalf("unexpected invalidation event: %+v", ev)
	}
}

func TestTokenRefreshCacheNonBuildNonOAuthDisabledMissingNotSuppressed(t *testing.T) {
	db := openTestDatabase(t)
	repo := NewAccountRepository(db)

	t.Run("Web account", func(t *testing.T) {
		webAcc, _, err := repo.UpsertByIdentity(context.Background(), account.Credential{
			Provider:             account.ProviderWeb,
			Name:                 "web-account",
			SourceKey:            "web-account",
			EncryptedAccessToken: "web-token",
			ExpiresAt:            time.Now().UTC().Add(time.Hour),
			Enabled:              true,
			AuthStatus:           account.AuthStatusActive,
		})
		if err != nil {
			t.Fatal(err)
		}

		var events []repository.InvalidationEvent
		repo.SetInvalidationObserver(func(_ context.Context, e repository.InvalidationEvent) {
			events = append(events, e)
		})

		_, err = repo.UpdateTokens(context.Background(), webAcc.ID, "web-new", "", time.Now().UTC().Add(time.Hour), 0)
		if err != nil {
			t.Fatal(err)
		}
		if len(events) != 1 || events[0].Provider != account.ProviderWeb || events[0].AccountID != webAcc.ID {
			t.Fatalf("expected invalidation for web account, got %+v", events)
		}
	})

	t.Run("Console account", func(t *testing.T) {
		consoleAcc, _, err := repo.UpsertByIdentity(context.Background(), account.Credential{
			Provider:             account.ProviderConsole,
			Name:                 "console-account",
			SourceKey:            "console-account",
			EncryptedAccessToken: "console-token",
			ExpiresAt:            time.Now().UTC().Add(time.Hour),
			Enabled:              true,
			AuthStatus:           account.AuthStatusActive,
		})
		if err != nil {
			t.Fatal(err)
		}

		var events []repository.InvalidationEvent
		repo.SetInvalidationObserver(func(_ context.Context, e repository.InvalidationEvent) {
			events = append(events, e)
		})

		_, err = repo.UpdateTokens(context.Background(), consoleAcc.ID, "console-new", "", time.Now().UTC().Add(time.Hour), 0)
		if err != nil {
			t.Fatal(err)
		}
		if len(events) != 1 || events[0].Provider != account.ProviderConsole || events[0].AccountID != consoleAcc.ID {
			t.Fatalf("expected invalidation for console account, got %+v", events)
		}
	})

	t.Run("Non-OAuth credential", func(t *testing.T) {
		acc := createTestBuildOAuthAccount(t, db, "non-oauth-account")
		if err := db.db.Model(&accountCredentialModel{}).Where("account_id = ?", acc.ID).Updates(map[string]any{
			"auth_type":         string(account.AuthTypeSSO),
			"encrypted_refresh": "",
		}).Error; err != nil {
			t.Fatal(err)
		}

		var events []repository.InvalidationEvent
		repo.SetInvalidationObserver(func(_ context.Context, e repository.InvalidationEvent) {
			events = append(events, e)
		})

		_, err := repo.UpdateTokens(context.Background(), acc.ID, "sso-new", "", time.Now().UTC().Add(time.Hour), 0)
		if err != nil {
			t.Fatal(err)
		}
		if len(events) != 1 || events[0].Provider != account.ProviderBuild || events[0].AccountID != acc.ID {
			t.Fatalf("expected invalidation for non-oauth account, got %+v", events)
		}
	})

	t.Run("Disabled account", func(t *testing.T) {
		acc := createTestBuildOAuthAccount(t, db, "disabled-account")
		if err := db.db.Model(&accountModel{}).Where("id = ?", acc.ID).Update("enabled", false).Error; err != nil {
			t.Fatal(err)
		}

		var events []repository.InvalidationEvent
		repo.SetInvalidationObserver(func(_ context.Context, e repository.InvalidationEvent) {
			events = append(events, e)
		})

		_, err := repo.UpdateTokens(context.Background(), acc.ID, "dis-new", "dis-refresh", time.Now().UTC().Add(time.Hour), 0)
		if err != nil {
			t.Fatal(err)
		}
		if len(events) != 1 || events[0].Provider != account.ProviderBuild || events[0].AccountID != acc.ID {
			t.Fatalf("expected invalidation for disabled account, got %+v", events)
		}
	})

	t.Run("Missing credential row", func(t *testing.T) {
		acc := createTestBuildOAuthAccount(t, db, "missing-cred-account")
		if err := db.db.Where("account_id = ?", acc.ID).Delete(&accountCredentialModel{}).Error; err != nil {
			t.Fatal(err)
		}

		var events []repository.InvalidationEvent
		repo.SetInvalidationObserver(func(_ context.Context, e repository.InvalidationEvent) {
			events = append(events, e)
		})

		_, err := repo.UpdateTokens(context.Background(), acc.ID, "miss-new", "miss-refresh", time.Now().UTC().Add(time.Hour), 0)
		if err != nil {
			t.Fatal(err)
		}
		if len(events) != 1 || events[0].Provider != account.ProviderBuild || events[0].AccountID != acc.ID {
			t.Fatalf("expected invalidation for missing credential row, got %+v", events)
		}
	})
}

func TestTokenRefreshCacheTransactionFailureAndReadBackFailure(t *testing.T) {
	db := openTestDatabase(t)
	repo := NewAccountRepository(db)

	t.Run("Missing parent account failure rolls back and emits nothing", func(t *testing.T) {
		var events []repository.InvalidationEvent
		repo.SetInvalidationObserver(func(_ context.Context, e repository.InvalidationEvent) {
			events = append(events, e)
		})

		nonExistentID := uint64(99999999)
		_, err := repo.UpdateTokens(context.Background(), nonExistentID, "tok", "ref", time.Now().UTC().Add(time.Hour), 0)
		if err == nil {
			t.Fatal("expected error for non-existent account")
		}
		if len(events) != 0 {
			t.Fatalf("transaction failure emitted events: %+v", events)
		}
	})

	t.Run("Injected parent update failure rolls back credential update and emits nothing", func(t *testing.T) {
		acc := createTestBuildOAuthAccount(t, db, "tx-parent-fail-account")
		injectedErr := errors.New("injected parent update failure")
		failParentUpdate := false
		var credUpdated bool
		callbackName := "test:inject_parent_update_failure"
		if err := db.db.Callback().Update().Before("gorm:update").Register(callbackName, func(tx *gorm.DB) {
			if !failParentUpdate {
				return
			}
			if tx.Statement.Table == "account_credentials" {
				credUpdated = true
			} else if credUpdated && (tx.Statement.Table == "provider_accounts" || (tx.Statement.Schema != nil && tx.Statement.Schema.Table == "provider_accounts")) {
				tx.AddError(injectedErr)
			}
		}); err != nil {
			t.Fatal(err)
		}
		defer func() {
			_ = db.db.Callback().Update().Remove(callbackName)
		}()

		var events []repository.InvalidationEvent
		repo.SetInvalidationObserver(func(_ context.Context, e repository.InvalidationEvent) {
			events = append(events, e)
		})

		failParentUpdate = true
		newExpiry := time.Now().UTC().Add(2 * time.Hour)
		_, err := repo.UpdateTokens(context.Background(), acc.ID, "attempted-token", "attempted-refresh", newExpiry, 0)
		failParentUpdate = false

		if !errors.Is(err, injectedErr) {
			t.Fatalf("expected injectedErr, got: %v", err)
		}
		if !credUpdated {
			t.Fatal("credential update was not executed before parent update")
		}
		if len(events) != 0 {
			t.Fatalf("transaction failure emitted invalidation events: %+v", events)
		}

		stored, err := repo.Get(context.Background(), acc.ID)
		if err != nil {
			t.Fatalf("failed to read back stored account: %v", err)
		}
		if stored.EncryptedAccessToken != acc.EncryptedAccessToken {
			t.Fatalf("expected token %q to be preserved, got %q", acc.EncryptedAccessToken, stored.EncryptedAccessToken)
		}
		if stored.EncryptedRefreshToken != acc.EncryptedRefreshToken {
			t.Fatalf("expected refresh token %q to be preserved, got %q", acc.EncryptedRefreshToken, stored.EncryptedRefreshToken)
		}
		if !stored.ExpiresAt.Equal(acc.ExpiresAt) {
			t.Fatalf("expected expiry %v to be preserved, got %v", acc.ExpiresAt, stored.ExpiresAt)
		}
	})

	t.Run("Read-back failure notifies and returns error", func(t *testing.T) {
		acc := createTestBuildOAuthAccount(t, db, "readback-fail-account")

		injectedErr := errors.New("injected read-back query failure")
		armed := false
		failQuery := false
		updateCbName := "test:arm_readback_failure"
		queryCbName := "test:inject_readback_failure"

		if err := db.db.Callback().Update().After("gorm:update").Register(updateCbName, func(tx *gorm.DB) {
			if armed && (tx.Statement.Table == "provider_accounts" || (tx.Statement.Schema != nil && tx.Statement.Schema.Table == "provider_accounts")) {
				failQuery = true
			}
		}); err != nil {
			t.Fatal(err)
		}
		defer func() {
			_ = db.db.Callback().Update().Remove(updateCbName)
		}()

		if err := db.db.Callback().Query().Before("gorm:query").Register(queryCbName, func(tx *gorm.DB) {
			if failQuery {
				tx.AddError(injectedErr)
			}
		}); err != nil {
			t.Fatal(err)
		}
		defer func() {
			_ = db.db.Callback().Query().Remove(queryCbName)
		}()

		var events []repository.InvalidationEvent
		repo.SetInvalidationObserver(func(_ context.Context, e repository.InvalidationEvent) {
			events = append(events, e)
		})

		armed = true
		newExpiry := time.Now().UTC().Add(time.Hour)
		_, err := repo.UpdateTokens(context.Background(), acc.ID, "committed-token", "committed-refresh", newExpiry, 0)
		armed = false
		failQuery = false

		if !errors.Is(err, injectedErr) {
			t.Fatalf("expected injectedErr, got: %v", err)
		}
		if len(events) != 1 {
			t.Fatalf("expected 1 fallback invalidation event on read-back failure, got %d: %+v", len(events), events)
		}
		ev := events[0]
		if ev.Kind != repository.InvalidationAccountCredentialChanged || ev.AccountID != acc.ID {
			t.Fatalf("unexpected readback fallback event: %+v", ev)
		}
		if ev.Provider != "" {
			t.Fatalf("expected empty Provider for broad fallback invalidation, got: %q", ev.Provider)
		}

		stored, err := repo.Get(context.Background(), acc.ID)
		if err != nil {
			t.Fatalf("failed to read back committed account: %v", err)
		}
		if stored.EncryptedAccessToken != "committed-token" {
			t.Fatalf("expected committed-token, got: %s", stored.EncryptedAccessToken)
		}
		if stored.EncryptedRefreshToken != "committed-refresh" {
			t.Fatalf("expected committed-refresh, got: %s", stored.EncryptedRefreshToken)
		}
		if !stored.ExpiresAt.Equal(newExpiry) {
			t.Fatalf("expected committed expiry %v, got %v", newExpiry, stored.ExpiresAt)
		}
	})
}

func TestTokenRefreshCacheEmptyRefreshTokenPreservesOldToken(t *testing.T) {
	db := openTestDatabase(t)
	repo := NewAccountRepository(db)
	acc := createTestBuildOAuthAccount(t, db, "empty-refresh-account")

	expires := time.Now().UTC().Add(3 * time.Hour)
	updated, err := repo.UpdateTokens(context.Background(), acc.ID, "brand-new-access", "", expires, 0)
	if err != nil {
		t.Fatal(err)
	}
	if updated.EncryptedAccessToken != "brand-new-access" {
		t.Fatalf("access token not updated: %s", updated.EncryptedAccessToken)
	}
	if updated.EncryptedRefreshToken != "initial-refresh-token" {
		t.Fatalf("refresh token was overwritten by empty string, got: %s", updated.EncryptedRefreshToken)
	}

	stored, err := repo.Get(context.Background(), acc.ID)
	if err != nil {
		t.Fatal(err)
	}
	if stored.EncryptedRefreshToken != "initial-refresh-token" {
		t.Fatalf("stored refresh token was overwritten: %s", stored.EncryptedRefreshToken)
	}
}

func TestTokenRefreshCacheConcurrentOrdinaryRefreshAndRoutingMutation(t *testing.T) {
	db := openTestDatabase(t)
	repo := NewAccountRepository(db)
	acc := createTestBuildOAuthAccount(t, db, "concurrent-mutation-account")

	var mu sync.Mutex
	var events []repository.InvalidationEvent
	repo.SetInvalidationObserver(func(_ context.Context, e repository.InvalidationEvent) {
		mu.Lock()
		defer mu.Unlock()
		events = append(events, e)
	})

	const numWorkers = 8
	const numIterations = 20
	const numMutations = 5
	errCh := make(chan error, numWorkers*numIterations+numMutations)

	var wg sync.WaitGroup
	start := make(chan struct{})

	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			<-start
			for i := 0; i < numIterations; i++ {
				token := fmt.Sprintf("access-%d-%d", workerID, i)
				_, err := repo.UpdateTokens(context.Background(), acc.ID, token, "refresh-static", time.Now().UTC().Add(time.Hour), 0)
				if err != nil {
					errCh <- fmt.Errorf("worker %d iter %d: %w", workerID, i, err)
				}
			}
		}(w)
	}

	wg.Add(1)
	go func() {
		defer wg.Done()
		<-start
		for i := 0; i < numMutations; i++ {
			time.Sleep(2 * time.Millisecond)
			flag := (i % 2) + 1
			_, err := repo.UpdateTokens(context.Background(), acc.ID, "bot-mutated-token", "refresh-static", time.Now().UTC().Add(time.Hour), flag)
			if err != nil {
				errCh <- fmt.Errorf("mutator iter %d: %w", i, err)
			}
		}
	}()

	close(start)
	wg.Wait()
	close(errCh)

	for err := range errCh {
		t.Fatalf("concurrent UpdateTokens write failed: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	if len(events) == 0 {
		t.Fatal("expected invalidations for bot flag mutations, got 0")
	}
	for _, ev := range events {
		if ev.Kind != repository.InvalidationAccountCredentialChanged || ev.Provider != account.ProviderBuild || ev.AccountID != acc.ID {
			t.Fatalf("unexpected event in concurrent run: %+v", ev)
		}
	}
}
