package relational

import (
	"context"
	"errors"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/chenyme/grok2api/backend/internal/domain/account"
	"github.com/chenyme/grok2api/backend/internal/repository"
	"gorm.io/gorm"
)

func createCapabilityInvalidationAccount(t *testing.T, db *Database, provider account.Provider) uint64 {
	t.Helper()
	row := accountModel{IdentityKey: testIdentityKey(string(provider)), Provider: string(provider), Name: string(provider), SourceKey: string(provider), AuthStatus: string(account.AuthStatusActive)}
	if err := db.db.Create(&row).Error; err != nil {
		t.Fatal(err)
	}
	return row.ID
}

func TestModelCapabilitySameSetOnlyRefreshesSyncState(t *testing.T) {
	db := openTestDatabase(t)
	repo := NewModelRepository(db)
	id := createCapabilityInvalidationAccount(t, db, account.ProviderWeb)
	old := time.Now().UTC().Truncate(time.Second)
	if err := repo.ReplaceAccountCapabilities(context.Background(), id, []string{"model-a", "model-b"}, old); err != nil {
		t.Fatal(err)
	}
	if err := repo.MarkAccountCapabilitySyncFailed(context.Background(), id, old.Add(time.Minute), "temporary error"); err != nil {
		t.Fatal(err)
	}
	writes, events := 0, 0
	countWrites := func(tx *gorm.DB) {
		if tx.Statement.Table == "account_model_capabilities" {
			writes++
		}
	}
	if err := db.db.Callback().Create().Before("gorm:create").Register("test:capability_create", countWrites); err != nil {
		t.Fatal(err)
	}
	if err := db.db.Callback().Delete().Before("gorm:delete").Register("test:capability_delete", countWrites); err != nil {
		t.Fatal(err)
	}
	repo.SetInvalidationObserver(func(context.Context, repository.InvalidationEvent) { events++ })
	newer := old.Add(time.Hour)
	if err := repo.ReplaceAccountCapabilities(context.Background(), id, []string{" model-b ", "model-a", "model-b", " "}, newer); err != nil {
		t.Fatal(err)
	}
	if events != 0 || writes != 0 {
		t.Fatalf("unchanged initialized set: events=%d capability writes=%d", events, writes)
	}
	var state accountModelSyncStateModel
	if err := db.db.First(&state, id).Error; err != nil {
		t.Fatal(err)
	}
	if !state.LastAttemptAt.Equal(newer) || state.LastSuccessAt == nil || !state.LastSuccessAt.Equal(newer) || state.LastError != "" {
		t.Fatalf("sync state not refreshed: %+v", state)
	}
}

func TestModelCapabilityInitializationAndChangesInvalidateRealProvider(t *testing.T) {
	for _, provider := range []account.Provider{account.ProviderBuild, account.ProviderWeb, account.ProviderConsole} {
		t.Run(string(provider), func(t *testing.T) {
			db := openTestDatabase(t)
			repo := NewModelRepository(db)
			id := createCapabilityInvalidationAccount(t, db, provider)
			now := time.Now().UTC()
			if err := repo.MarkAccountCapabilitySyncFailed(context.Background(), id, now, "not initialized"); err != nil {
				t.Fatal(err)
			}
			var events []repository.InvalidationEvent
			repo.SetInvalidationObserver(func(_ context.Context, event repository.InvalidationEvent) { events = append(events, event) })
			cases := []struct {
				values []string
				events int
			}{{nil, 1}, {[]string{" "}, 0}, {[]string{"model-a"}, 1}, {[]string{"model-b", "model-a"}, 1}, {[]string{"model-b"}, 1}, {nil, 1}, {nil, 0}}
			for index, tc := range cases {
				events = nil
				if err := repo.ReplaceAccountCapabilities(context.Background(), id, tc.values, now.Add(time.Duration(index+1)*time.Minute)); err != nil {
					t.Fatal(err)
				}
				if len(events) != tc.events {
					t.Fatalf("step %d events=%+v want %d", index, events, tc.events)
				}
				for _, event := range events {
					if event.Kind != repository.InvalidationAccountCapabilityChanged || event.Provider != provider || event.AccountID != id {
						t.Fatalf("incorrect scope: %+v", event)
					}
				}
			}
		})
	}
}

func TestModelCapabilityReplacementRollbackDoesNotPublish(t *testing.T) {
	db := openTestDatabase(t)
	repo := NewModelRepository(db)
	id := createCapabilityInvalidationAccount(t, db, account.ProviderBuild)
	old := time.Now().UTC().Truncate(time.Second)
	if err := repo.ReplaceAccountCapabilities(context.Background(), id, []string{"old-model"}, old); err != nil {
		t.Fatal(err)
	}
	injected := errors.New("sync state write failed")
	if err := db.db.Callback().Create().Before("gorm:create").Register("test:capability_sync_failure", func(tx *gorm.DB) {
		if tx.Statement.Table == "account_model_sync_states" {
			tx.AddError(injected)
		}
	}); err != nil {
		t.Fatal(err)
	}
	events := 0
	repo.SetInvalidationObserver(func(context.Context, repository.InvalidationEvent) { events++ })
	if err := repo.ReplaceAccountCapabilities(context.Background(), id, []string{"new-model"}, old.Add(time.Hour)); !errors.Is(err, injected) {
		t.Fatalf("error=%v", err)
	}
	if events != 0 {
		t.Fatal("rolled-back replacement invalidated routing")
	}
	var models []string
	if err := db.db.Model(&accountModelCapabilityModel{}).Where("account_id = ?", id).Pluck("upstream_model", &models).Error; err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(models, []string{"old-model"}) {
		t.Fatalf("rollback lost old models: %v", models)
	}
	var state accountModelSyncStateModel
	if err := db.db.First(&state, id).Error; err != nil {
		t.Fatal(err)
	}
	if state.LastSuccessAt == nil || !state.LastSuccessAt.Equal(old) {
		t.Fatalf("rollback changed sync time: %+v", state)
	}
}

func TestModelCapabilityConcurrentSameSetInitializesOnce(t *testing.T) {
	db := openTestDatabase(t)
	repo := NewModelRepository(db)
	id := createCapabilityInvalidationAccount(t, db, account.ProviderConsole)
	var mu sync.Mutex
	events := 0
	repo.SetInvalidationObserver(func(context.Context, repository.InvalidationEvent) { mu.Lock(); events++; mu.Unlock() })
	start := make(chan struct{})
	outcomes := make(chan error, 8)
	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			outcomes <- repo.ReplaceAccountCapabilities(context.Background(), id, []string{"model-a"}, time.Now().UTC())
		}()
	}
	close(start)
	wg.Wait()
	close(outcomes)
	for err := range outcomes {
		if err != nil {
			t.Fatal(err)
		}
	}
	if events != 1 {
		t.Fatalf("same-set concurrent initialization emitted %d events", events)
	}
}

func TestModelCapabilityFirstEmptySuccessInitializesMissingState(t *testing.T) {
	db := openTestDatabase(t)
	repo := NewModelRepository(db)
	id := createCapabilityInvalidationAccount(t, db, account.ProviderBuild)
	events := 0
	repo.SetInvalidationObserver(func(context.Context, repository.InvalidationEvent) { events++ })
	if err := repo.ReplaceAccountCapabilities(context.Background(), id, nil, time.Now()); err != nil {
		t.Fatal(err)
	}
	known, err := repo.HasSuccessfulAccountSync(context.Background(), id)
	if err != nil || !known || events != 1 {
		t.Fatalf("known=%v events=%d error=%v", known, events, err)
	}
}

func TestModelCapabilityReadFailureLeavesOldSnapshotAndNoNotification(t *testing.T) {
	for _, table := range []string{"provider_accounts", "account_model_sync_states", "account_model_capabilities"} {
		t.Run(table, func(t *testing.T) {
			db := openTestDatabase(t)
			repo := NewModelRepository(db)
			id := createCapabilityInvalidationAccount(t, db, account.ProviderConsole)
			if err := repo.ReplaceAccountCapabilities(context.Background(), id, []string{"old-model"}, time.Now()); err != nil {
				t.Fatal(err)
			}
			injected := errors.New("comparison read unavailable")
			fail := true
			if err := db.db.Callback().Query().Before("gorm:query").Register("test:capability_read_failure", func(tx *gorm.DB) {
				if fail && tx.Statement.Table == table {
					tx.AddError(injected)
				}
			}); err != nil {
				t.Fatal(err)
			}
			events := 0
			repo.SetInvalidationObserver(func(context.Context, repository.InvalidationEvent) { events++ })
			if err := repo.ReplaceAccountCapabilities(context.Background(), id, []string{"new-model"}, time.Now()); !errors.Is(err, injected) {
				t.Fatalf("error=%v", err)
			}
			if events != 0 {
				t.Fatal("failed read notified an uncommitted change")
			}
			fail = false
			var models []string
			if err := db.db.Model(&accountModelCapabilityModel{}).Where("account_id = ?", id).Pluck("upstream_model", &models).Error; err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(models, []string{"old-model"}) {
				t.Fatalf("failed read changed models: %v", models)
			}
		})
	}
}

func TestModelCapabilityMissingAccountStillFailsWithoutNotification(t *testing.T) {
	db := openTestDatabase(t)
	repo := NewModelRepository(db)
	events := 0
	repo.SetInvalidationObserver(func(context.Context, repository.InvalidationEvent) { events++ })
	if err := repo.ReplaceAccountCapabilities(context.Background(), 999, nil, time.Now()); err == nil {
		t.Fatal("nonexistent account accepted")
	}
	if events != 0 {
		t.Fatal("failed foreign key write emitted invalidation")
	}
}
