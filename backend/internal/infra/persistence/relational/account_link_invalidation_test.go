package relational

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/chenyme/grok2api/backend/internal/domain/account"
	"github.com/chenyme/grok2api/backend/internal/repository"
	"gorm.io/gorm"
)

func createLinkInvalidationAccount(t *testing.T, db *Database, provider account.Provider, name, userID string) uint64 {
	t.Helper()
	row := accountModel{IdentityKey: testIdentityKey(name), Provider: string(provider), Name: name, SourceKey: name, UserID: userID, AuthStatus: string(account.AuthStatusActive)}
	if err := db.db.Create(&row).Error; err != nil {
		t.Fatal(err)
	}
	return row.ID
}

func TestAccountLinkInvalidationReportsCommittedChangesOnceGlobally(t *testing.T) {
	db := openTestDatabase(t)
	repo := NewAccountRepository(db)
	web := createLinkInvalidationAccount(t, db, account.ProviderWeb, "notify-web", "linked-user")
	console := createLinkInvalidationAccount(t, db, account.ProviderConsole, "notify-console", "linked-user")
	build := createLinkInvalidationAccount(t, db, account.ProviderBuild, "notify-build", "linked-user")
	var events []repository.InvalidationEvent
	repo.SetInvalidationObserver(func(_ context.Context, event repository.InvalidationEvent) { events = append(events, event) })
	if err := repo.ReconcileProviderLinks(context.Background(), web); err != nil {
		t.Fatal(err)
	}
	if len(events) != 1 || events[0].Kind != repository.InvalidationAccountCredentialChanged || events[0].Provider != "" || events[0].AccountID != 0 {
		t.Fatalf("cross-provider event = %+v", events)
	}
	var consoleLink webConsoleAccountLinkModel
	var buildLink accountProviderLinkModel
	if err := db.db.First(&consoleLink).Error; err != nil {
		t.Fatal(err)
	}
	if err := db.db.First(&buildLink).Error; err != nil {
		t.Fatal(err)
	}
	if consoleLink.WebAccountID != web || consoleLink.ConsoleAccountID != console || buildLink.WebAccountID != web || buildLink.BuildAccountID != build {
		t.Fatal("expected links were not created")
	}
	events = nil
	for _, id := range []uint64{web, console, build} {
		if err := repo.ReconcileProviderLinks(context.Background(), id); err != nil {
			t.Fatal(err)
		}
	}
	if len(events) != 0 {
		t.Fatalf("unchanged links invalidated routing: %+v", events)
	}
}

func TestAccountLinkInvalidationSkipsNoopReconciliation(t *testing.T) {
	for _, scenario := range []string{"missing-candidates", "ambiguous-candidates", "existing-conflicts"} {
		t.Run(scenario, func(t *testing.T) {
			db := openTestDatabase(t)
			repo := NewAccountRepository(db)
			web := createLinkInvalidationAccount(t, db, account.ProviderWeb, "noop-web", "user")
			if scenario == "ambiguous-candidates" {
				createLinkInvalidationAccount(t, db, account.ProviderConsole, "console-a", "user")
				createLinkInvalidationAccount(t, db, account.ProviderConsole, "console-b", "user")
				createLinkInvalidationAccount(t, db, account.ProviderBuild, "build-a", "user")
				createLinkInvalidationAccount(t, db, account.ProviderBuild, "build-b", "user")
			}
			if scenario == "existing-conflicts" {
				otherWeb := createLinkInvalidationAccount(t, db, account.ProviderWeb, "other-web", "other-user")
				console := createLinkInvalidationAccount(t, db, account.ProviderConsole, "console", "user")
				build := createLinkInvalidationAccount(t, db, account.ProviderBuild, "build", "user")
				if err := db.db.Create(&webConsoleAccountLinkModel{WebAccountID: otherWeb, ConsoleAccountID: console, CreatedAt: time.Now()}).Error; err != nil {
					t.Fatal(err)
				}
				if err := db.db.Create(&accountProviderLinkModel{WebAccountID: otherWeb, BuildAccountID: build, CreatedAt: time.Now()}).Error; err != nil {
					t.Fatal(err)
				}
			}
			events := 0
			repo.SetInvalidationObserver(func(context.Context, repository.InvalidationEvent) { events++ })
			if err := repo.ReconcileProviderLinks(context.Background(), web); err != nil {
				t.Fatal(err)
			}
			if events != 0 {
				t.Fatalf("no-op reconciliation emitted %d events", events)
			}
			for _, table := range []string{"web_console_account_links", "account_provider_links"} {
				var count int64
				if err := db.db.Table(table).Where("web_account_id = ?", web).Count(&count).Error; err != nil {
					t.Fatal(err)
				}
				if count != 0 {
					t.Fatalf("no-op created a link in %s", table)
				}
			}
		})
	}
}

func TestAccountLinkInvalidationDoesNotPublishRolledBackChanges(t *testing.T) {
	db := openTestDatabase(t)
	repo := NewAccountRepository(db)
	web := createLinkInvalidationAccount(t, db, account.ProviderWeb, "rollback-web", "rollback-user")
	createLinkInvalidationAccount(t, db, account.ProviderConsole, "rollback-console", "rollback-user")
	createLinkInvalidationAccount(t, db, account.ProviderBuild, "rollback-build", "rollback-user")
	injected := errors.New("second link insert failed")
	sawFirstInsert := false
	if err := db.db.Callback().Create().Before("gorm:create").Register("test:fail_second_link", func(tx *gorm.DB) {
		if tx.Statement.Table != "account_provider_links" {
			return
		}
		var count int64
		if err := tx.Session(&gorm.Session{NewDB: true}).Model(&webConsoleAccountLinkModel{}).Where("web_account_id = ?", web).Count(&count).Error; err != nil {
			tx.AddError(err)
			return
		}
		sawFirstInsert = count == 1
		tx.AddError(injected)
	}); err != nil {
		t.Fatal(err)
	}
	events := 0
	repo.SetInvalidationObserver(func(context.Context, repository.InvalidationEvent) { events++ })
	if err := repo.ReconcileProviderLinks(context.Background(), web); !errors.Is(err, injected) {
		t.Fatalf("error=%v", err)
	}
	if !sawFirstInsert {
		t.Fatal("failure did not occur after the first link insert")
	}
	if events != 0 {
		t.Fatal("rolled-back changes published an event")
	}
	for _, table := range []string{"web_console_account_links", "account_provider_links"} {
		var count int64
		if err := db.db.Table(table).Count(&count).Error; err != nil {
			t.Fatal(err)
		}
		if count != 0 {
			t.Fatalf("rollback left a row in %s", table)
		}
	}
}
