package relational

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/chenyme/grok2api/backend/internal/domain/account"
	"github.com/chenyme/grok2api/backend/internal/repository"
	"gorm.io/gorm"
)

func TestAccountInvalidationResolvesUncachedProvider(t *testing.T) {
	db := openTestDatabase(t)
	repo := NewAccountRepository(db)
	for _, provider := range []account.Provider{account.ProviderWeb, account.ProviderConsole} {
		t.Run(string(provider), func(t *testing.T) {
			repo.SetInvalidationObserver(nil)
			value, _, err := repo.UpsertByIdentity(context.Background(), account.Credential{Provider: provider, Name: string(provider), SourceKey: string(provider), EncryptedAccessToken: testEncryptedToken, AuthStatus: account.AuthStatusActive})
			if err != nil {
				t.Fatal(err)
			}
			var events []repository.InvalidationEvent
			repo.SetInvalidationObserver(func(_ context.Context, event repository.InvalidationEvent) { events = append(events, event) })
			if err := repo.SaveBilling(context.Background(), account.Billing{AccountID: value.ID, SyncedAt: time.Now()}); err != nil {
				t.Fatal(err)
			}
			if err := repo.SaveQuotaWindows(context.Background(), value.ID, account.WebTier("auto"), time.Now(), nil); err != nil {
				t.Fatal(err)
			}
			if len(events) != 2 {
				t.Fatalf("events = %+v", events)
			}
			for _, event := range events {
				if event.Provider != provider || event.AccountID != value.ID {
					t.Fatalf("unscoped event: %+v", event)
				}
			}
			if events[0].Kind != repository.InvalidationAccountBillingChanged || events[1].Kind != repository.InvalidationAccountQuotaChanged {
				t.Fatalf("event kinds changed: %+v", events)
			}
		})
	}
}

func TestAccountInvalidationLookupIsMinimalAndPreservesFallback(t *testing.T) {
	db := openTestDatabase(t)
	repo := NewAccountRepository(db)
	value, _, err := repo.UpsertByIdentity(context.Background(), account.Credential{Provider: account.ProviderWeb, Name: "scope", SourceKey: "scope", EncryptedAccessToken: testEncryptedToken, AuthStatus: account.AuthStatusActive})
	if err != nil {
		t.Fatal(err)
	}
	var selects [][]string
	var queryFailure error
	if err := db.db.Callback().Query().Before("gorm:query").Register("test:invalidation_provider_query", func(tx *gorm.DB) {
		selects = append(selects, append([]string(nil), tx.Statement.Selects...))
		if queryFailure != nil {
			tx.AddError(queryFailure)
		}
	}); err != nil {
		t.Fatal(err)
	}
	event := repository.InvalidationEvent{Kind: repository.InvalidationAccountBillingChanged, AccountID: value.ID}
	repo.notifyInvalidation(context.Background(), event)
	if len(selects) != 0 {
		t.Fatal("observer disabled but query executed")
	}
	var received []repository.InvalidationEvent
	repo.SetInvalidationObserver(func(_ context.Context, event repository.InvalidationEvent) { received = append(received, event) })
	explicit := event
	explicit.Provider = account.ProviderConsole
	repo.notifyInvalidation(context.Background(), explicit)
	if len(selects) != 0 || received[0].Provider != account.ProviderConsole {
		t.Fatal("explicit provider replaced or queried")
	}
	repo.notifyInvalidation(context.Background(), event)
	if len(selects) != 1 || !reflect.DeepEqual(selects[0], []string{"provider"}) || received[1].Provider != account.ProviderWeb {
		t.Fatalf("lookup selects=%v events=%+v", selects, received)
	}
	missing := event
	missing.AccountID = value.ID + 999
	repo.notifyInvalidation(context.Background(), missing)
	if received[2] != missing {
		t.Fatalf("missing row discarded or changed event: %+v", received[2])
	}
	queryFailure = errors.New("test database unavailable")
	repo.notifyInvalidation(context.Background(), event)
	if len(received) != 4 || received[3] != event {
		t.Fatalf("lookup failure discarded event: %+v", received)
	}
	global := repository.InvalidationEvent{Kind: repository.InvalidationAccountStateChanged}
	queries := len(selects)
	repo.notifyInvalidation(context.Background(), global)
	if len(selects) != queries || received[4] != global {
		t.Fatal("global event unnecessarily queried or changed")
	}
}

func TestAccountObservedModelInvalidationOnlyForChangedValue(t *testing.T) {
	db := openTestDatabase(t)
	repo := NewAccountRepository(db)
	old := time.Now().UTC().Truncate(time.Second)
	tests := []struct {
		name, stored, incoming string
		storedAt               *time.Time
		incomingAt             time.Time
		updated, invalidated   bool
	}{
		{"same-expired", "grok-4.5", "grok-4.5", &old, old.Add(31 * time.Minute), true, false},
		{"same-boundary", "grok-4.5", "grok-4.5", &old, old.Add(30 * time.Minute), true, false},
		{"same-not-expired", "grok-4.5", "grok-4.5", &old, old.Add(29 * time.Minute), false, false},
		{"same-stale", "grok-4.5", "grok-4.5", &old, old.Add(-time.Hour), false, false},
		{"different-newer", "grok-4.5", "grok-4.6", &old, old.Add(time.Minute), true, true},
		{"different-same-time", "grok-4.5", "grok-4.6", &old, old, true, true},
		{"different-stale", "grok-4.5", "grok-4.6", &old, old.Add(-time.Minute), false, false},
		{"to-build-free", "grok-4.5", "grok-4.5-build-free", &old, old.Add(time.Minute), true, true},
		{"from-build-free", "grok-4.5-build-free", "grok-4.5", &old, old.Add(time.Minute), true, true},
		{"same-null-time", "grok-4.5", "grok-4.5", nil, old, false, false},
		{"different-null-time", "", "grok-4.5", nil, old, true, true},
	}
	for index, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			row := accountModel{IdentityKey: testIdentityKey(tc.name), Provider: string(account.ProviderBuild), Name: tc.name, SourceKey: fmt.Sprint(index), ObservedModel: tc.stored, ObservedModelAt: tc.storedAt, AuthStatus: string(account.AuthStatusActive)}
			if err := db.db.Create(&row).Error; err != nil {
				t.Fatal(err)
			}
			var events []repository.InvalidationEvent
			repo.SetInvalidationObserver(func(_ context.Context, event repository.InvalidationEvent) { events = append(events, event) })
			updated, err := repo.UpdateObservedModelIfNewer(context.Background(), row.ID, tc.incoming, tc.incomingAt)
			if err != nil || updated != tc.updated {
				t.Fatalf("updated=%v error=%v", updated, err)
			}
			if (len(events) > 0) != tc.invalidated {
				t.Fatalf("events=%+v want invalidation=%v", events, tc.invalidated)
			}
			if tc.invalidated && (len(events) != 1 || events[0].Kind != repository.InvalidationAccountStateChanged || events[0].Provider != account.ProviderBuild || events[0].AccountID != row.ID) {
				t.Fatalf("event=%+v", events)
			}
			var stored accountModel
			if err := db.db.First(&stored, row.ID).Error; err != nil {
				t.Fatal(err)
			}
			wantModel, wantTime := tc.stored, tc.storedAt
			if tc.updated {
				wantModel, wantTime = tc.incoming, &tc.incomingAt
			}
			if stored.ObservedModel != wantModel || (stored.ObservedModelAt == nil) != (wantTime == nil) || wantTime != nil && !stored.ObservedModelAt.Equal(*wantTime) {
				t.Fatalf("stored model=%s time=%v", stored.ObservedModel, stored.ObservedModelAt)
			}
		})
	}
}

func TestAccountObservedModelRefreshDoesNotOverwriteConcurrentNewerModel(t *testing.T) {
	db := openTestDatabase(t)
	repo := NewAccountRepository(db)
	old := time.Now().UTC().Truncate(time.Second)
	row := accountModel{IdentityKey: testIdentityKey("observed-concurrent"), Provider: string(account.ProviderBuild), Name: "concurrent", SourceKey: "concurrent", ObservedModel: "grok-4.5", ObservedModelAt: &old, AuthStatus: string(account.AuthStatusActive)}
	if err := db.db.Create(&row).Error; err != nil {
		t.Fatal(err)
	}
	var eventsMu sync.Mutex
	var events []repository.InvalidationEvent
	repo.SetInvalidationObserver(func(_ context.Context, event repository.InvalidationEvent) {
		eventsMu.Lock()
		events = append(events, event)
		eventsMu.Unlock()
	})
	start := make(chan struct{})
	errors := make(chan error, 12)
	var wg sync.WaitGroup
	newer := old.Add(2 * time.Hour)
	for i := range 12 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			model, observedAt := "grok-4.5", old.Add(time.Hour)
			if i%2 == 1 {
				model, observedAt = "grok-4.5-build-free", newer
			}
			_, err := repo.UpdateObservedModelIfNewer(context.Background(), row.ID, model, observedAt)
			errors <- err
		}()
	}
	close(start)
	wg.Wait()
	close(errors)
	for err := range errors {
		if err != nil {
			t.Fatal(err)
		}
	}
	var stored accountModel
	if err := db.db.First(&stored, row.ID).Error; err != nil {
		t.Fatal(err)
	}
	if stored.ObservedModel != "grok-4.5-build-free" || stored.ObservedModelAt == nil || !stored.ObservedModelAt.Equal(newer) {
		t.Fatalf("stale refresh replaced newer model: %s at %v", stored.ObservedModel, stored.ObservedModelAt)
	}
	if len(events) == 0 {
		t.Fatal("changed model did not invalidate routing")
	}
	for _, event := range events {
		if event.Kind != repository.InvalidationAccountStateChanged || event.Provider != account.ProviderBuild || event.AccountID != row.ID {
			t.Fatalf("incorrect concurrent event: %+v", event)
		}
	}
}
