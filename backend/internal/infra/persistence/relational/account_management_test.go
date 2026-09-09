package relational

import (
	"context"
	"testing"
	"time"

	"github.com/chenyme/grok2api/backend/internal/domain/account"
	"github.com/chenyme/grok2api/backend/internal/repository"
)

func TestAccountManagementPatchPreservesCredentialAndIdentity(t *testing.T) {
	r, _, b := reauthPair(t)
	ctx := context.Background()
	now := time.Now().UTC()
	if _, err := r.UpdateTokens(ctx, b.ID, "rotated-access", "rotated-refresh", now.Add(time.Hour), 0); err != nil {
		t.Fatal(err)
	}
	before, err := r.Get(ctx, b.ID)
	if err != nil {
		t.Fatal(err)
	}
	name := "renamed"
	priority := 23
	cookie := "cookie-update"
	after, err := r.UpdateAccountManagement(ctx, b.ID, repository.ManagementAccountUpdates{Name: &name, Priority: &priority, EncryptedCloudflareCookie: &cookie})
	if err != nil {
		t.Fatal(err)
	}
	if after.Name != name || after.Priority != priority || after.EncryptedCloudflareCookie != cookie {
		t.Fatal("explicit patch not applied")
	}
	if after.EncryptedAccessToken != before.EncryptedAccessToken || after.EncryptedRefreshToken != before.EncryptedRefreshToken || after.OIDCClientID != before.OIDCClientID || after.AuthStatus != before.AuthStatus || after.Enabled != before.Enabled || after.SourceKey != before.SourceKey || after.UserID != before.UserID || after.Email != before.Email || !after.RefreshDueAt.Equal(*before.RefreshDueAt) || !after.LastRefreshAt.Equal(*before.LastRefreshAt) {
		t.Fatal("patch changed credential, identity, schedule, or omitted enablement")
	}
}

func TestAccountReauthStateCASPreservesSettingsTokensAndAge(t *testing.T) {
	r, _, b := reauthPair(t)
	ctx := context.Background()
	now := time.Now().UTC()
	if _, err := r.UpdateTokens(ctx, b.ID, "rotated-access", "rotated-refresh", now.Add(time.Hour), 0); err != nil {
		t.Fatal(err)
	}
	marked, err := r.MarkAccountReauthRequired(ctx, b, "stale rejection", now)
	if err != nil || marked {
		t.Fatalf("stale marked=%v err=%v", marked, err)
	}
	current, _ := r.Get(ctx, b.ID)
	marked, err = r.MarkAccountReauthRequired(ctx, current, "current rejection", now)
	if err != nil || !marked {
		t.Fatalf("current marked=%v err=%v", marked, err)
	}
	first, _ := r.Get(ctx, b.ID)
	if first.AuthStatus != account.AuthStatusReauthRequired || first.ReauthMarkedAt == nil || !first.ReauthMarkedAt.Equal(now) || first.EncryptedRefreshToken != "rotated-refresh" || first.Enabled || first.Priority != b.Priority {
		t.Fatal("narrow rejection changed operational or credential state")
	}
	marked, err = r.MarkAccountReauthRequired(ctx, first, "repeat rejection", now.Add(time.Hour))
	if err != nil || !marked {
		t.Fatalf("repeat marked=%v err=%v", marked, err)
	}
	last, _ := r.Get(ctx, b.ID)
	if !last.ReauthMarkedAt.Equal(*first.ReauthMarkedAt) {
		t.Fatal("repeat rejection reset cleanup age")
	}
}
