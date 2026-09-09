package account

import (
	"context"
	"sync"
	"testing"
	"time"

	accountdomain "github.com/chenyme/grok2api/backend/internal/domain/account"
	"github.com/chenyme/grok2api/backend/internal/repository"
)

type interleavedManagementRepository struct {
	repository.AccountRepository
	once      sync.Once
	afterRead func()
}

func (r *interleavedManagementRepository) Get(ctx context.Context, id uint64) (accountdomain.Credential, error) {
	value, err := r.AccountRepository.Get(ctx, id)
	if err == nil {
		r.once.Do(r.afterRead)
	}
	return value, err
}
func (r *interleavedManagementRepository) UpdateAccountManagement(ctx context.Context, id uint64, patch repository.ManagementAccountUpdates) (accountdomain.Credential, error) {
	return r.AccountRepository.(repository.AccountManagementWriter).UpdateAccountManagement(ctx, id, patch)
}
func (r *interleavedManagementRepository) MarkAccountReauthRequired(ctx context.Context, value accountdomain.Credential, reason string, now time.Time) (bool, error) {
	return r.AccountRepository.(repository.AccountAuthenticationStateWriter).MarkAccountReauthRequired(ctx, value, reason, now)
}
func (r *interleavedManagementRepository) RevokeRecoveryActivations(ctx context.Context, p accountdomain.Provider, ids []uint64) error {
	return r.AccountRepository.(repository.RecoveryActivationRevoker).RevokeRecoveryActivations(ctx, p, ids)
}

func TestManagementUpdatePreservesConcurrentCredentialRotation(t *testing.T) {
	for _, rotation := range []string{"oauth", "sso"} {
		for _, field := range []string{"disable", "name"} {
			t.Run(rotation+"_"+field, func(t *testing.T) {
				s, w, b, _ := newReauthService(t)
				ctx := context.Background()
				original := s.accounts
				s.accounts = &interleavedManagementRepository{AccountRepository: original, afterRead: func() {
					var err error
					if rotation == "oauth" {
						_, err = original.UpdateTokens(ctx, b.ID, "rotated-access", "rotated-refresh", time.Now().Add(time.Hour), 0)
					} else {
						var ok bool
						ok, err = original.(repository.ReauthRepository).SaveBuildReauth(ctx, b, w, "rotated-access", "rotated-refresh", "rotated-client", time.Now().Add(time.Hour), 0)
						if !ok && err == nil {
							t.Error("SSO rotation did not save")
						}
					}
					if err != nil {
						t.Error(err)
					}
				}}
				disabled := false
				name := "new management name"
				input := UpdateInput{Name: &name}
				if field == "disable" {
					input = UpdateInput{Enabled: &disabled}
				}
				if _, err := s.Update(ctx, b.ID, input); err != nil {
					t.Fatal(err)
				}
				got, _ := original.Get(ctx, b.ID)
				if got.EncryptedAccessToken != "rotated-access" || got.EncryptedRefreshToken != "rotated-refresh" || got.AuthStatus != accountdomain.AuthStatusActive {
					t.Fatal("management update reverted credential rotation")
				}
				if rotation == "sso" && got.OIDCClientID != "rotated-client" {
					t.Fatal("management update reverted OAuth client")
				}
				if field == "disable" && got.Enabled {
					t.Fatal("explicit disabled state lost")
				}
				if field == "name" && got.Name != name {
					t.Fatal("explicit name update lost")
				}
			})
		}
	}
}

func TestMarkReauthRequiredDoesNotRevertConcurrentRotation(t *testing.T) {
	s, _, b, _ := newReauthService(t)
	ctx := context.Background()
	original := s.accounts
	s.accounts = &interleavedManagementRepository{AccountRepository: original, afterRead: func() {
		if _, err := original.UpdateTokens(ctx, b.ID, "rotated-access", "rotated-refresh", time.Now().Add(time.Hour), 0); err != nil {
			t.Error(err)
		}
	}}
	if err := s.MarkReauthRequired(ctx, b.ID, "old credential rejected"); err != nil {
		t.Fatal(err)
	}
	got, _ := original.Get(ctx, b.ID)
	if got.EncryptedRefreshToken != "rotated-refresh" || got.AuthStatus != accountdomain.AuthStatusActive {
		t.Fatal("stale rejection reverted rotated credentials or invalidated new token")
	}
}

func TestStaleSSORejectionDoesNotInvalidateReplacement(t *testing.T) {
	s, w, _, _ := newReauthService(t)
	ctx := context.Background()
	fresh := w
	fresh.EncryptedAccessToken = "new-sso"
	if _, err := s.accounts.Update(ctx, fresh); err != nil {
		t.Fatal(err)
	}
	if err := s.markSSOCredentialRejected(ctx, w, "old SSO rejected"); err != nil {
		t.Fatal(err)
	}
	got, _ := s.accounts.Get(ctx, w.ID)
	if got.EncryptedAccessToken != "new-sso" || got.AuthStatus != accountdomain.AuthStatusActive {
		t.Fatal("old SSO response invalidated the replacement")
	}
}
