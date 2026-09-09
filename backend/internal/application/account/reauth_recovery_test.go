package account

import (
	"context"
	"encoding/base64"
	"errors"
	"path/filepath"
	"strconv"
	"sync/atomic"
	"testing"
	"time"

	accountdomain "github.com/chenyme/grok2api/backend/internal/domain/account"
	"github.com/chenyme/grok2api/backend/internal/infra/persistence/relational"
	"github.com/chenyme/grok2api/backend/internal/infra/provider"
	"github.com/chenyme/grok2api/backend/internal/infra/runtime/memory"
	"github.com/chenyme/grok2api/backend/internal/infra/security"
)

type reauthAdapter struct {
	calls  atomic.Int64
	seed   provider.CredentialSeed
	err    error
	during func(context.Context)
}

func (*reauthAdapter) Provider() accountdomain.Provider { return accountdomain.ProviderWeb }
func (a *reauthAdapter) ConvertToBuild(ctx context.Context, _ accountdomain.Credential) (provider.CredentialSeed, error) {
	a.calls.Add(1)
	if a.during != nil {
		a.during(ctx)
	}
	return a.seed, a.err
}
func newReauthService(t *testing.T) (*Service, accountdomain.Credential, accountdomain.Credential, *reauthAdapter) {
	t.Helper()
	ctx := context.Background()
	db, err := relational.OpenSQLite(ctx, filepath.Join(t.TempDir(), "reauth.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = db.Close() })
	if err = db.InitializeSchema(ctx); err != nil {
		t.Fatal(err)
	}
	cipher, err := security.NewCipher(base64.StdEncoding.EncodeToString(make([]byte, 32)))
	if err != nil {
		t.Fatal(err)
	}
	sso, _ := cipher.Encrypt("sso")
	repo := relational.NewAccountRepository(db)
	web, _, err := repo.UpsertByIdentity(ctx, accountdomain.Credential{Provider: accountdomain.ProviderWeb, AuthType: accountdomain.AuthTypeSSO, Name: "web", SourceKey: "web", UserID: "user", Email: " USER@example.com ", EncryptedAccessToken: sso, AuthStatus: accountdomain.AuthStatusActive})
	if err != nil {
		t.Fatal(err)
	}
	build, _, err := repo.UpsertByIdentity(ctx, accountdomain.Credential{Provider: accountdomain.ProviderBuild, AuthType: accountdomain.AuthTypeOAuth, Name: "build", SourceKey: "build", UserID: "user", Email: "user@example.com", EncryptedAccessToken: "old", EncryptedRefreshToken: "old-refresh", OIDCClientID: "old-client", AuthStatus: accountdomain.AuthStatusReauthRequired, Priority: 13})
	if err != nil {
		t.Fatal(err)
	}
	if err = repo.LinkWebToBuild(ctx, web.ID, build.ID); err != nil {
		t.Fatal(err)
	}
	web.Enabled = false
	build.Enabled = false
	if _, err := repo.Update(ctx, web); err != nil {
		t.Fatal(err)
	}
	if _, err := repo.Update(ctx, build); err != nil {
		t.Fatal(err)
	}
	web, _ = repo.Get(ctx, web.ID)
	build, _ = repo.Get(ctx, build.ID)
	a := &reauthAdapter{seed: provider.CredentialSeed{UserID: "user", Email: "user@example.com", AccessToken: "new", RefreshToken: "new-refresh", OIDCClientID: "new-client", ExpiresAt: time.Now().UTC().Add(time.Hour)}}
	s := NewService(repo, relational.NewAuditRepository(db), nil, nil, provider.NewRegistry(a), cipher, memory.NewLockStore())
	return s, web, build, a
}

func TestRecoverBuildAuthenticationPreservesDisabledAndCanceledSuccess(t *testing.T) {
	s, _, b, a := newReauthService(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	a.during = func(context.Context) { cancel() }
	result, err := s.RecoverBuildAuthentication(ctx, b.ID, true, time.Hour, 24*time.Hour)
	if err != nil || !result.Claimed || !result.Recovered {
		t.Fatalf("result=%+v err=%v", result, err)
	}
	got, _ := s.accounts.Get(context.Background(), b.ID)
	access, _ := s.cipher.Decrypt(got.EncryptedAccessToken)
	if got.Enabled || got.Priority != 13 || got.OIDCClientID != "new-client" || got.AuthStatus != accountdomain.AuthStatusActive || access != "new" {
		t.Fatal("new credentials or operational fields incorrect")
	}
}

func TestRecoverBuildAuthenticationRejectsSeedIdentityConflict(t *testing.T) {
	for _, id := range []string{"different-user", ""} {
		t.Run(id, func(t *testing.T) {
			s, _, b, a := newReauthService(t)
			a.seed.UserID = id
			result, err := s.RecoverBuildAuthentication(context.Background(), b.ID, true, time.Hour, 24*time.Hour)
			if err == nil || result.Recovered || !result.Claimed {
				t.Fatalf("result=%+v err=%v", result, err)
			}
			got, _ := s.accounts.Get(context.Background(), b.ID)
			if got.EncryptedRefreshToken != b.EncryptedRefreshToken || got.AuthStatus != accountdomain.AuthStatusReauthRequired || got.RefreshDueAt == nil {
				t.Fatal("conflicting identity overwrote credential or lacked backoff")
			}
		})
	}
}

func TestRecoverBuildAuthenticationTransientBackoffAndDisabledFilter(t *testing.T) {
	s, w, b, a := newReauthService(t)
	a.err = context.DeadlineExceeded
	result, err := s.RecoverBuildAuthentication(context.Background(), b.ID, false, time.Hour, 24*time.Hour)
	if err != nil || !result.Skipped || a.calls.Load() != 0 {
		t.Fatal("disabled account attempted")
	}
	result, err = s.RecoverBuildAuthentication(context.Background(), b.ID, true, time.Hour, 24*time.Hour)
	if err == nil || !result.Claimed || result.Recovered {
		t.Fatalf("result=%+v err=%v", result, err)
	}
	result, err = s.RecoverBuildAuthentication(context.Background(), b.ID, true, time.Hour, 24*time.Hour)
	if err != nil || !result.Skipped || a.calls.Load() != 1 {
		t.Fatal("persistent backoff did not suppress retry")
	}
	got, _ := s.accounts.Get(context.Background(), w.ID)
	if got.AuthStatus != accountdomain.AuthStatusActive {
		t.Fatal("timeout marked SSO permanently invalid")
	}
}

func TestRecoverBuildAuthenticationDeduplicatesManualConversion(t *testing.T) {
	s, w, b, a := newReauthService(t)
	started := make(chan struct{})
	proceed := make(chan struct{})
	a.during = func(context.Context) { close(started); <-proceed }
	done := make(chan error, 1)
	go func() {
		_, err := s.RecoverBuildAuthentication(context.Background(), b.ID, true, time.Hour, 24*time.Hour)
		done <- err
	}()
	<-started
	_, _, _, err := s.convertWebAccountToBuild(context.Background(), w.ID, BuildConversionAll)
	if !errors.Is(err, ErrConversionBusy) {
		t.Fatalf("manual error=%v", err)
	}
	result, err := s.RecoverBuildAuthentication(context.Background(), b.ID, true, time.Hour, 24*time.Hour)
	if err != nil || !result.Skipped {
		t.Fatalf("duplicate result=%+v err=%v", result, err)
	}
	close(proceed)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if a.calls.Load() != 1 {
		t.Fatal("duplicated conversion")
	}
}

func TestRecoverBuildAuthenticationMarksOnlyExplicitUnauthorized(t *testing.T) {
	s, w, b, a := newReauthService(t)
	a.err = provider.ErrUnauthorized
	result, err := s.RecoverBuildAuthentication(context.Background(), b.ID, true, time.Hour, 24*time.Hour)
	if err == nil || !result.Claimed || result.Recovered {
		t.Fatalf("result=%+v err=%v", result, err)
	}
	got, _ := s.accounts.Get(context.Background(), w.ID)
	if got.AuthStatus != accountdomain.AuthStatusReauthRequired {
		t.Fatal("SSO unauthorized not recorded")
	}
	ids, err := s.ListBuildReauthCandidates(context.Background(), time.Now().Add(48*time.Hour), true, 10)
	if err != nil || len(ids) != 0 {
		t.Fatalf("rejected SSO remained eligible: %v %v", ids, err)
	}
}

func TestRecoverBuildAuthenticationCanceledBeforeAttempt(t *testing.T) {
	s, _, b, a := newReauthService(t)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	result, err := s.RecoverBuildAuthentication(ctx, b.ID, true, time.Hour, 24*time.Hour)
	if !errors.Is(err, context.Canceled) || result.Claimed || a.calls.Load() != 0 {
		t.Fatalf("result=%+v err=%v", result, err)
	}
}

func TestReauthBackoffBounds(t *testing.T) {
	for _, tt := range []struct {
		count int
		want  time.Duration
	}{{0, time.Hour}, {1, 2 * time.Hour}, {5, 24 * time.Hour}, {1000, 24 * time.Hour}} {
		if got := reauthBackoff(tt.count, time.Hour, 24*time.Hour); got != tt.want {
			t.Fatalf("count=%d got=%v want=%v", tt.count, got, tt.want)
		}
	}
}

func TestRecoverBuildAuthenticationInterruptedAttemptPersistsBackoff(t *testing.T) {
	s, w, b, a := newReauthService(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	a.err = context.Canceled
	a.during = func(context.Context) { cancel() }
	result, err := s.RecoverBuildAuthentication(ctx, b.ID, true, time.Hour, 24*time.Hour)
	if err == nil || !result.Claimed || result.Recovered {
		t.Fatalf("result=%+v err=%v", result, err)
	}
	got, _ := s.accounts.Get(context.Background(), b.ID)
	if got.RefreshDueAt == nil || got.RefreshDueAt.Before(time.Now().Add(59*time.Minute)) || got.RefreshFailureCount != 1 {
		t.Fatal("interrupted attempt lost durable retry")
	}
	web, _ := s.accounts.Get(context.Background(), w.ID)
	if web.AuthStatus != accountdomain.AuthStatusActive {
		t.Fatal("cancellation rejected valid SSO")
	}
}

func TestManualLinkedConversionPreservesDisabledAndRejectsConflict(t *testing.T) {
	s, w, b, a := newReauthService(t)
	id, created, skipped, err := s.convertWebAccountToBuild(context.Background(), w.ID, BuildConversionAll)
	if err != nil || id != b.ID || created || skipped {
		t.Fatalf("manual result id=%d created=%v skipped=%v err=%v", id, created, skipped, err)
	}
	got, _ := s.accounts.Get(context.Background(), b.ID)
	if got.Enabled || got.OIDCClientID != "new-client" {
		t.Fatal("manual conversion modified enablement or lost client ID")
	}
	a.seed.UserID = "other-user"
	_, _, _, err = s.convertWebAccountToBuild(context.Background(), w.ID, BuildConversionAll)
	if err == nil {
		t.Fatal("manual conversion accepted conflicting identity")
	}
}

func TestRecoverBuildAuthenticationSharesOAuthRefreshLock(t *testing.T) {
	s, _, b, a := newReauthService(t)
	release, acquired, err := s.refreshLock.Acquire(context.Background(), "credential-refresh:"+strconv.FormatUint(b.ID, 10), 2*time.Minute)
	if err != nil || !acquired {
		t.Fatalf("lock %v %v", acquired, err)
	}
	defer release()
	result, err := s.RecoverBuildAuthentication(context.Background(), b.ID, true, time.Hour, 24*time.Hour)
	if err != nil || !result.Skipped || result.Claimed || a.calls.Load() != 0 {
		t.Fatalf("result=%+v err=%v", result, err)
	}
}

func TestRecoverBuildAuthenticationDoesNotOverwriteConcurrentImport(t *testing.T) {
	s, _, b, a := newReauthService(t)
	a.during = func(context.Context) {
		if _, err := s.accounts.UpdateTokens(context.Background(), b.ID, "imported-access", "imported-refresh", time.Now().Add(2*time.Hour), 0); err != nil {
			t.Error(err)
		}
	}
	result, err := s.RecoverBuildAuthentication(context.Background(), b.ID, true, time.Hour, 24*time.Hour)
	if err == nil || !result.Claimed || result.Recovered || result.Reason != "credential_changed" {
		t.Fatalf("result=%+v err=%v", result, err)
	}
	got, _ := s.accounts.Get(context.Background(), b.ID)
	if got.EncryptedRefreshToken != "imported-refresh" || got.AuthStatus != accountdomain.AuthStatusActive || got.RefreshFailureCount != 0 {
		t.Fatal("stale reauthorization changed newer imported credentials")
	}
}
