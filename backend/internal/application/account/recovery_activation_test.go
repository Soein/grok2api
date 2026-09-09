package account

import (
	"context"
	"errors"
	"testing"

	accountdomain "github.com/chenyme/grok2api/backend/internal/domain/account"
	"github.com/chenyme/grok2api/backend/internal/repository"
)

type activationRevocationRepository struct {
	repository.AccountRepository
	revoked           bool
	revokeErr         error
	writes            int
	writeBeforeRevoke bool
}

func (r *activationRevocationRepository) RevokeRecoveryActivations(_ context.Context, p accountdomain.Provider, ids []uint64) error {
	if p == accountdomain.ProviderBuild && len(ids) > 0 {
		r.revoked = true
	}
	return r.revokeErr
}
func (r *activationRevocationRepository) Update(ctx context.Context, value accountdomain.Credential) (accountdomain.Credential, error) {
	r.writes++
	r.writeBeforeRevoke = !r.revoked
	return r.AccountRepository.Update(ctx, value)
}
func (r *activationRevocationRepository) UpdateMany(ctx context.Context, p accountdomain.Provider, ids []uint64, updates repository.AccountUpdates) (int64, error) {
	r.writes++
	r.writeBeforeRevoke = !r.revoked
	return r.AccountRepository.UpdateMany(ctx, p, ids, updates)
}

func TestRecoveryActivationAdministratorDisableRevokesEvenIfAlreadyDisabled(t *testing.T) {
	for _, batch := range []bool{false, true} {
		t.Run(map[bool]string{false: "single", true: "batch"}[batch], func(t *testing.T) {
			s, _, b, _ := newReauthService(t)
			tracked := &activationRevocationRepository{AccountRepository: s.accounts}
			s.accounts = tracked
			disabled := false
			var err error
			if batch {
				_, err = s.BatchUpdate(context.Background(), accountdomain.ProviderBuild, []uint64{b.ID}, UpdateInput{Enabled: &disabled})
			} else {
				_, err = s.Update(context.Background(), b.ID, UpdateInput{Enabled: &disabled})
			}
			if err != nil {
				t.Fatal(err)
			}
			if !tracked.revoked || tracked.writes != 1 || tracked.writeBeforeRevoke {
				t.Fatalf("revoked=%v writes=%d early=%v", tracked.revoked, tracked.writes, tracked.writeBeforeRevoke)
			}
			got, _ := s.accounts.Get(context.Background(), b.ID)
			if got.Enabled {
				t.Fatal("explicit disable did not remain disabled")
			}
		})
	}
}

func TestRecoveryActivationAdministratorRevokeFailsClosed(t *testing.T) {
	for _, batch := range []bool{false, true} {
		t.Run(map[bool]string{false: "single", true: "batch"}[batch], func(t *testing.T) {
			s, _, b, _ := newReauthService(t)
			tracked := &activationRevocationRepository{AccountRepository: s.accounts, revokeErr: errors.New("revoke_storage_failed")}
			s.accounts = tracked
			enabled := true
			var err error
			if batch {
				_, err = s.BatchUpdate(context.Background(), accountdomain.ProviderBuild, []uint64{b.ID}, UpdateInput{Enabled: &enabled})
			} else {
				_, err = s.Update(context.Background(), b.ID, UpdateInput{Enabled: &enabled})
			}
			if err == nil || tracked.writes != 0 {
				t.Fatalf("err=%v writes=%d", err, tracked.writes)
			}
			got, _ := s.accounts.Get(context.Background(), b.ID)
			if got.Enabled {
				t.Fatal("failed revoke allowed enablement update")
			}
		})
	}
}

func TestRecoveryActivationAutomaticAuthenticationChangeKeepsAuthorization(t *testing.T) {
	s, _, b, _ := newReauthService(t)
	tracked := &activationRevocationRepository{AccountRepository: s.accounts}
	s.accounts = tracked
	if err := s.MarkReauthRequired(context.Background(), b.ID, "reauthentication required"); err != nil {
		t.Fatal(err)
	}
	if tracked.revoked {
		t.Fatal("automatic authentication update revoked recovery authorization")
	}
	priority := 18
	if _, err := s.Update(context.Background(), b.ID, UpdateInput{Priority: &priority}); err != nil {
		t.Fatal(err)
	}
	if tracked.revoked {
		t.Fatal("routing preference update revoked recovery authorization")
	}
}
