package relational

import (
	"context"
	"path/filepath"
	"testing"
	"time"

	account "github.com/chenyme/grok2api/backend/internal/domain/account"
)

func TestQuotaWindowRecoveryFiltersBeforeLimit(t *testing.T) {
	ctx := context.Background()
	db, err := OpenSQLite(ctx, filepath.Join(t.TempDir(), "quota-window-recovery.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = db.Close() })
	if err := db.InitializeSchema(ctx); err != nil {
		t.Fatal(err)
	}
	repo := NewAccountRepository(db)
	now := time.Now().UTC()
	past := now.Add(-time.Hour)
	future := now.Add(time.Hour)
	create := func(name string, p account.Provider, enabled bool, auth account.AuthStatus, cooldown *time.Time, mode string, remaining int, reset *time.Time) uint64 {
		value, _, err := repo.UpsertByIdentity(ctx, account.Credential{Provider: p, AuthType: account.AuthTypeSSO, Name: name, SourceKey: name, EncryptedAccessToken: "encrypted", Enabled: true, AuthStatus: auth})
		if err != nil {
			t.Fatal(err)
		}
		value.Enabled = enabled
		value.AuthStatus = auth
		if _, err := repo.Update(ctx, value); err != nil {
			t.Fatal(err)
		}
		if cooldown != nil {
			if err := repo.UpdateHealth(ctx, value.ID, p, 1, cooldown, "", false); err != nil {
				t.Fatal(err)
			}
		}
		if err := repo.SaveQuotaWindows(ctx, value.ID, "", now, []account.QuotaWindow{{AccountID: value.ID, Mode: mode, Remaining: remaining, ResetAt: reset}}); err != nil {
			t.Fatal(err)
		}
		return value.ID
	}
	disabled := create("disabled", account.ProviderWeb, false, account.AuthStatusActive, nil, "fast", 0, &past)
	create("reauth", account.ProviderWeb, true, account.AuthStatusReauthRequired, nil, "fast", 0, &past)
	create("cooldown", account.ProviderWeb, true, account.AuthStatusActive, &future, "fast", 0, &past)
	create("build", account.ProviderBuild, true, account.AuthStatusActive, nil, "fast", 0, &past)
	create("future", account.ProviderWeb, true, account.AuthStatusActive, nil, "fast", 0, &future)
	create("recovered", account.ProviderWeb, true, account.AuthStatusActive, nil, "fast", 1, &past)
	create("wrong-mode", account.ProviderConsole, true, account.AuthStatusActive, nil, "weekly", 0, &past)
	web := create("web", account.ProviderWeb, true, account.AuthStatusActive, nil, "expert", 0, &past)
	console := create("console", account.ProviderConsole, true, account.AuthStatusActive, nil, "console_video", 0, &past)
	values, err := repo.ListDueQuotaWindowsForRecovery(ctx, now, 1, []account.Provider{account.ProviderWeb, account.ProviderConsole}, false)
	if err != nil || len(values) != 1 || values[0].AccountID != web {
		t.Fatalf("bounded eligible values=%+v err=%v", values, err)
	}
	values, err = repo.ListDueQuotaWindowsForRecovery(ctx, now, 10, []account.Provider{account.ProviderConsole}, true)
	if err != nil || len(values) != 1 || values[0].AccountID != console {
		t.Fatalf("provider filter values=%+v err=%v", values, err)
	}
	values, err = repo.ListDueQuotaWindowsForRecovery(ctx, now, 10, []account.Provider{account.ProviderWeb}, true)
	if err != nil || len(values) != 2 || values[0].AccountID != disabled || values[1].AccountID != web {
		t.Fatalf("include disabled values=%+v err=%v", values, err)
	}
	values, err = repo.ListDueQuotaWindowsForRecovery(ctx, now, 10, nil, true)
	if err != nil || len(values) != 0 {
		t.Fatalf("empty providers values=%+v err=%v", values, err)
	}
	if err := repo.SaveQuotaWindows(ctx, web, "", now, []account.QuotaWindow{{AccountID: web, Mode: "heavy", Remaining: 0, ResetAt: &past}}); err != nil {
		t.Fatal(err)
	}
	cursor := account.QuotaWindow{AccountID: web, Mode: "expert", ResetAt: &past}
	values, err = repo.ListDueQuotaWindowsForRecoveryAfter(ctx, now, 1, &cursor, []account.Provider{account.ProviderWeb, account.ProviderConsole}, false)
	if err != nil || len(values) != 1 || values[0].AccountID != web || values[0].Mode != "heavy" {
		t.Fatalf("same-account cursor values=%+v err=%v", values, err)
	}
	cursor = values[0]
	values, err = repo.ListDueQuotaWindowsForRecoveryAfter(ctx, now, 1, &cursor, []account.Provider{account.ProviderWeb, account.ProviderConsole}, false)
	if err != nil || len(values) != 1 || values[0].AccountID != console {
		t.Fatalf("next-account cursor values=%+v err=%v", values, err)
	}
	nilResetTests := []struct {
		name     string
		provider account.Provider
		mode     string
		age      time.Duration
		seconds  int
		want     bool
	}{
		{"console-predicted", account.ProviderConsole, "console", 24 * time.Hour, 0, true},
		{"console-too-soon", account.ProviderConsole, "console", 23 * time.Hour, 0, false},
		{"web-duration", account.ProviderWeb, "fast", time.Hour, 3600, true},
		{"web-duration-pending", account.ProviderWeb, "fast", 30 * time.Minute, 3600, false},
		{"web-fallback", account.ProviderWeb, "expert", 5 * time.Minute, 0, true},
		{"web-fallback-pending", account.ProviderWeb, "expert", 4 * time.Minute, 0, false},
	}
	for _, tt := range nilResetTests {
		id := create(tt.name, tt.provider, true, account.AuthStatusActive, nil, tt.mode, 0, nil)
		synced := now.Add(-tt.age)
		if err := repo.SaveQuotaWindows(ctx, id, "", synced, []account.QuotaWindow{{AccountID: id, Mode: tt.mode, Remaining: 0, WindowSeconds: tt.seconds}}); err != nil {
			t.Fatal(err)
		}
		values, err := repo.ListDueQuotaWindowsForRecovery(ctx, now, 100, []account.Provider{tt.provider}, false)
		if err != nil {
			t.Fatal(err)
		}
		found := false
		for _, value := range values {
			if value.AccountID == id {
				found = true
			}
		}
		if found != tt.want {
			t.Errorf("%s selected=%t want=%t", tt.name, found, tt.want)
		}
	}
}
