package relational

import (
	"context"
	"path/filepath"
	"testing"
	"time"

	"github.com/chenyme/grok2api/backend/internal/domain/account"
)

func reauthPair(t *testing.T) (*AccountRepository, account.Credential, account.Credential) {
	t.Helper()
	ctx := context.Background()
	db, err := OpenSQLite(ctx, filepath.Join(t.TempDir(), "reauth.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = db.Close() })
	if err = db.InitializeSchema(ctx); err != nil {
		t.Fatal(err)
	}
	r := NewAccountRepository(db)
	web, _, err := r.UpsertByIdentity(ctx, account.Credential{Provider: account.ProviderWeb, AuthType: account.AuthTypeSSO, Name: "web", SourceKey: "web", UserID: "same-user", EncryptedAccessToken: "sso", AuthStatus: account.AuthStatusActive})
	if err != nil {
		t.Fatal(err)
	}
	build, _, err := r.UpsertByIdentity(ctx, account.Credential{Provider: account.ProviderBuild, AuthType: account.AuthTypeOAuth, Name: "build", SourceKey: "build", UserID: "same-user", EncryptedAccessToken: "old-access", EncryptedRefreshToken: "old-refresh", AuthStatus: account.AuthStatusReauthRequired, Enabled: false, Priority: 17})
	if err != nil {
		t.Fatal(err)
	}
	if err = r.LinkWebToBuild(ctx, web.ID, build.ID); err != nil {
		t.Fatal(err)
	}
	web.Enabled = false
	build.Enabled = false
	if _, err := r.Update(ctx, web); err != nil {
		t.Fatal(err)
	}
	if _, err := r.Update(ctx, build); err != nil {
		t.Fatal(err)
	}
	web, _ = r.Get(ctx, web.ID)
	build, _ = r.Get(ctx, build.ID)
	return r, web, build
}

func TestBuildReauthCandidateClaimAndBackoff(t *testing.T) {
	r, web, build := reauthPair(t)
	ctx := context.Background()
	now := time.Now().UTC()
	lease := now.Add(2 * time.Minute)
	ids, err := r.ListBuildReauthCandidates(ctx, now, false, 10)
	if err != nil || len(ids) != 0 {
		t.Fatalf("disabled candidates: %v %v", ids, err)
	}
	ids, err = r.ListBuildReauthCandidates(ctx, now, true, 10)
	if err != nil || len(ids) != 1 {
		t.Fatalf("candidates: %v %v", ids, err)
	}
	claimed, err := r.ClaimBuildReauth(ctx, build, web, now, lease, true)
	if err != nil || !claimed {
		t.Fatalf("claim %v %v", claimed, err)
	}
	claimed, err = r.ClaimBuildReauth(ctx, build, web, now, lease, true)
	if err != nil || claimed {
		t.Fatalf("duplicate claim %v %v", claimed, err)
	}
	retry := now.Add(time.Hour)
	if err := r.FailBuildReauth(ctx, build, lease, retry, "sso_transient"); err != nil {
		t.Fatal(err)
	}
	ids, err = r.ListBuildReauthCandidates(ctx, now.Add(5*time.Minute), true, 10)
	if err != nil || len(ids) != 0 {
		t.Fatalf("backoff %v %v", ids, err)
	}
	got, _ := r.Get(ctx, build.ID)
	if got.RefreshFailureCount != 1 || got.RefreshDueAt == nil || !got.RefreshDueAt.Equal(retry) {
		t.Fatalf("backoff not persisted")
	}
}

func TestBuildReauthCASPreservesAccountAndClientID(t *testing.T) {
	r, web, build := reauthPair(t)
	ctx := context.Background()
	ok, err := r.SaveBuildReauth(ctx, build, web, "new-access", "new-refresh", "new-client", time.Now().Add(time.Hour), 0)
	if err != nil || !ok {
		t.Fatalf("save %v %v", ok, err)
	}
	got, _ := r.Get(ctx, build.ID)
	if got.Enabled || got.Priority != 17 || got.UserID != build.UserID || got.SourceKey != build.SourceKey || got.OIDCClientID != "new-client" || got.AuthStatus != account.AuthStatusActive {
		t.Fatalf("state changed unexpectedly")
	}
	ok, err = r.SaveBuildReauth(ctx, build, web, "stale-access", "stale-refresh", "stale-client", time.Now().Add(time.Hour), 0)
	if err != nil || ok {
		t.Fatalf("stale CAS %v %v", ok, err)
	}
	got, _ = r.Get(ctx, build.ID)
	if got.EncryptedRefreshToken != "new-refresh" {
		t.Fatal("overwrote newer token")
	}
}

func TestBuildReauthRejectsChangedLinkIdentityAndSSO(t *testing.T) {
	for _, change := range []string{"identity", "sso", "link"} {
		t.Run(change, func(t *testing.T) {
			r, w, b := reauthPair(t)
			ctx := context.Background()
			switch change {
			case "identity":
				if err := r.db.db.Model(&accountModel{}).Where("id = ?", b.ID).Update("user_id", "changed-user").Error; err != nil {
					t.Fatal(err)
				}
			case "sso":
				if err := r.db.db.Model(&accountCredentialModel{}).Where("account_id = ?", w.ID).Update("encrypted_primary", "new-sso").Error; err != nil {
					t.Fatal(err)
				}
			case "link":
				if err := r.db.db.Where("web_account_id = ?", w.ID).Delete(&accountProviderLinkModel{}).Error; err != nil {
					t.Fatal(err)
				}
			}
			saved, err := r.SaveBuildReauth(ctx, b, w, "new", "new-refresh", "new-client", time.Now().Add(time.Hour), 0)
			if err != nil || saved {
				t.Fatalf("changed %s CAS saved=%v err=%v", change, saved, err)
			}
			got, _ := r.Get(ctx, b.ID)
			if got.EncryptedRefreshToken != b.EncryptedRefreshToken {
				t.Fatal("overwrote token")
			}
		})
	}
}

func TestBuildReauthSSORejectionDoesNotPoisonReplacement(t *testing.T) {
	r, w, _ := reauthPair(t)
	ctx := context.Background()
	if err := r.db.db.Model(&accountCredentialModel{}).Where("account_id = ?", w.ID).Update("encrypted_primary", "replacement-sso").Error; err != nil {
		t.Fatal(err)
	}
	if err := r.RejectReauthSSO(ctx, w); err != nil {
		t.Fatal(err)
	}
	got, _ := r.Get(ctx, w.ID)
	if got.AuthStatus != account.AuthStatusActive {
		t.Fatal("stale SSO rejection poisoned replacement")
	}
	if err := r.RejectReauthSSO(ctx, got); err != nil {
		t.Fatal(err)
	}
	got, _ = r.Get(ctx, w.ID)
	if got.AuthStatus != account.AuthStatusReauthRequired {
		t.Fatal("current SSO rejection not recorded")
	}
}

func TestBuildReauthConcurrentClaimAndSave(t *testing.T) {
	r, w, b := reauthPair(t)
	ctx := context.Background()
	now := time.Now().UTC()
	start := make(chan struct{})
	outcomes := make(chan bool, 2)
	errs := make(chan error, 2)
	for i := 0; i < 2; i++ {
		go func() {
			<-start
			ok, err := r.ClaimBuildReauth(ctx, b, w, now, now.Add(2*time.Minute), true)
			outcomes <- ok
			errs <- err
		}()
	}
	close(start)
	claimed := 0
	for i := 0; i < 2; i++ {
		if <-outcomes {
			claimed++
		}
		if err := <-errs; err != nil {
			t.Fatal(err)
		}
	}
	if claimed != 1 {
		t.Fatalf("claimed=%d", claimed)
	}
	start = make(chan struct{})
	for i := 0; i < 2; i++ {
		go func() {
			<-start
			ok, err := r.SaveBuildReauth(ctx, b, w, "new-access", "new-refresh", "client", now.Add(time.Hour), 0)
			outcomes <- ok
			errs <- err
		}()
	}
	close(start)
	saved := 0
	for i := 0; i < 2; i++ {
		if <-outcomes {
			saved++
		}
		if err := <-errs; err != nil {
			t.Fatal(err)
		}
	}
	if saved != 1 {
		t.Fatalf("saved=%d", saved)
	}
}
