package relational

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/chenyme/grok2api/backend/internal/domain/account"
	"github.com/chenyme/grok2api/backend/internal/repository"
)

func authorizeRecoveryActivation(t *testing.T, r *AccountRepository, b account.Credential) {
	t.Helper()
	if err := r.db.db.Create(&accountRecoveryActivationModel{AccountID: b.ID, SourceKey: b.SourceKey, UserID: b.UserID, Email: b.Email, CreatedAt: time.Now().UTC()}).Error; err != nil {
		t.Fatal(err)
	}
}

func TestRecoveryActivationRequiresAuthorizationAndQuotaCompletion(t *testing.T) {
	r, _, b := reauthPair(t)
	ctx := context.Background()
	now := time.Now().UTC()
	if _, err := r.UpdateTokens(ctx, b.ID, "fresh-access", "fresh-refresh", now.Add(time.Hour), 0); err != nil {
		t.Fatal(err)
	}
	queued, err := r.PrepareRecoveredActivationQuota(ctx, b.ID, now)
	if err != nil || queued {
		t.Fatalf("unauthorized queued=%v err=%v", queued, err)
	}
	authorizeRecoveryActivation(t, r, b)
	queued, err = r.PrepareRecoveredActivationQuota(ctx, b.ID, now)
	if err != nil || !queued {
		t.Fatalf("authorized queued=%v err=%v", queued, err)
	}
	got, _ := r.Get(ctx, b.ID)
	if got.Enabled {
		t.Fatal("authentication alone enabled account")
	}
	pending, err := r.ListPendingRecoveryActivationIDs(ctx, 10)
	if err != nil || len(pending) != 0 {
		t.Fatalf("already queued pending=%v err=%v", pending, err)
	}
	lease := now.Add(5 * time.Minute)
	claimed, err := r.ClaimQuotaProbe(ctx, b.ID, now, lease)
	if err != nil || !claimed {
		t.Fatalf("claim=%v err=%v", claimed, err)
	}
	completed, err := r.CompleteQuotaProbe(ctx, b.ID, lease, true, now.Add(time.Minute))
	if err != nil || !completed {
		t.Fatalf("complete=%v err=%v", completed, err)
	}
	got, _ = r.Get(ctx, b.ID)
	if !got.Enabled {
		t.Fatal("authorized quota recovery did not enable")
	}
	var grants int64
	r.db.db.Model(&accountRecoveryActivationModel{}).Where("account_id = ?", b.ID).Count(&grants)
	if grants != 0 {
		t.Fatal("activation grant not consumed")
	}
	if _, err := r.GetQuotaRecovery(ctx, b.ID); !errors.Is(err, repository.ErrNotFound) {
		t.Fatalf("quota state=%v", err)
	}
}

func TestRecoveryActivationNoGrantNeverEnables(t *testing.T) {
	r, _, b := reauthPair(t)
	ctx := context.Background()
	now := time.Now().UTC()
	lease := now.Add(5 * time.Minute)
	if _, err := r.UpdateTokens(ctx, b.ID, "fresh-access", "fresh-refresh", now.Add(time.Hour), 0); err != nil {
		t.Fatal(err)
	}
	if err := r.SaveQuotaRecovery(ctx, account.QuotaRecovery{AccountID: b.ID, Kind: account.QuotaRecoveryKindFree, Status: account.QuotaRecoveryStatusProbing, NextProbeAt: &lease, UpdatedAt: now}); err != nil {
		t.Fatal(err)
	}
	done, err := r.CompleteQuotaProbe(ctx, b.ID, lease, true, now)
	if err != nil || !done {
		t.Fatalf("complete=%v err=%v", done, err)
	}
	got, _ := r.Get(ctx, b.ID)
	if got.Enabled {
		t.Fatal("unapproved disabled account enabled")
	}
}

func TestRecoveryActivationPendingSurvivesRestartAndPreservesExistingQuota(t *testing.T) {
	r, _, b := reauthPair(t)
	ctx := context.Background()
	now := time.Now().UTC()
	authorizeRecoveryActivation(t, r, b)
	pending, err := r.ListPendingRecoveryActivationIDs(ctx, 10)
	if err != nil || len(pending) != 0 {
		t.Fatalf("reauth pending=%v err=%v", pending, err)
	}
	if _, err := r.UpdateTokens(ctx, b.ID, "fresh-access", "fresh-refresh", now.Add(time.Hour), 0); err != nil {
		t.Fatal(err)
	}
	restarted := NewAccountRepository(r.db)
	pending, err = restarted.ListPendingRecoveryActivationIDs(ctx, 10)
	if err != nil || len(pending) != 1 || pending[0] != b.ID {
		t.Fatalf("restart pending=%v err=%v", pending, err)
	}
	due := now.Add(3 * time.Hour)
	if err := r.SaveQuotaRecovery(ctx, account.QuotaRecovery{AccountID: b.ID, Kind: account.QuotaRecoveryKindPaid, Status: account.QuotaRecoveryStatusExhausted, NextProbeAt: &due, UpdatedAt: now}); err != nil {
		t.Fatal(err)
	}
	queued, err := restarted.PrepareRecoveredActivationQuota(ctx, b.ID, now)
	if err != nil || queued {
		t.Fatalf("existing quota replaced: %v %v", queued, err)
	}
	quota, _ := r.GetQuotaRecovery(ctx, b.ID)
	if quota.Kind != account.QuotaRecoveryKindPaid || !quota.NextProbeAt.Equal(due) {
		t.Fatal("existing retry state changed")
	}
}

func TestRecoveryActivationRejectsRevokedIdentityChangedExpiredAndNewerLease(t *testing.T) {
	for _, condition := range []string{"revoked", "identity", "expired", "newer_lease", "authentication", "cooldown"} {
		t.Run(condition, func(t *testing.T) {
			r, _, b := reauthPair(t)
			ctx := context.Background()
			now := time.Now().UTC()
			lease := now.Add(5 * time.Minute)
			authorizeRecoveryActivation(t, r, b)
			if _, err := r.UpdateTokens(ctx, b.ID, "fresh-access", "fresh-refresh", now.Add(time.Hour), 0); err != nil {
				t.Fatal(err)
			}
			if err := r.SaveQuotaRecovery(ctx, account.QuotaRecovery{AccountID: b.ID, Kind: account.QuotaRecoveryKindFree, Status: account.QuotaRecoveryStatusProbing, NextProbeAt: &lease, UpdatedAt: now}); err != nil {
				t.Fatal(err)
			}
			suppliedLease := lease
			completeAt := now
			switch condition {
			case "revoked":
				if err := r.RevokeRecoveryActivations(ctx, account.ProviderBuild, []uint64{b.ID}); err != nil {
					t.Fatal(err)
				}
			case "identity":
				if err := r.db.db.Model(&accountModel{}).Where("id = ?", b.ID).Update("user_id", "replaced-user").Error; err != nil {
					t.Fatal(err)
				}
			case "expired":
				completeAt = lease.Add(time.Second)
			case "newer_lease":
				suppliedLease = lease.Add(-time.Minute)
			case "authentication":
				if err := r.db.db.Model(&accountModel{}).Where("id = ?", b.ID).Update("auth_status", account.AuthStatusReauthRequired).Error; err != nil {
					t.Fatal(err)
				}
			case "cooldown":
				if err := r.db.db.Model(&accountModel{}).Where("id = ?", b.ID).Update("cooldown_until", lease).Error; err != nil {
					t.Fatal(err)
				}
			}
			completed, err := r.CompleteQuotaProbe(ctx, b.ID, suppliedLease, true, completeAt)
			if err != nil {
				t.Fatal(err)
			}
			if (condition == "expired" || condition == "newer_lease") && completed {
				t.Fatal("non-owning quota lease completed")
			}
			got, _ := r.Get(ctx, b.ID)
			if got.Enabled {
				t.Fatal("ineligible activation enabled account")
			}
		})
	}
}

func TestRecoveryActivationUsesPaidBillingAndCascadesDelete(t *testing.T) {
	r, _, b := reauthPair(t)
	ctx := context.Background()
	now := time.Now().UTC()
	authorizeRecoveryActivation(t, r, b)
	if _, err := r.UpdateTokens(ctx, b.ID, "fresh-access", "fresh-refresh", now.Add(time.Hour), 0); err != nil {
		t.Fatal(err)
	}
	if err := r.SaveBilling(ctx, account.Billing{AccountID: b.ID, PlanName: "SuperGrok", MonthlyLimit: 100, SyncedAt: now}); err != nil {
		t.Fatal(err)
	}
	queued, err := r.PrepareRecoveredActivationQuota(ctx, b.ID, now)
	if err != nil || !queued {
		t.Fatalf("queue=%v err=%v", queued, err)
	}
	quota, _ := r.GetQuotaRecovery(ctx, b.ID)
	if quota.Kind != account.QuotaRecoveryKindPaid {
		t.Fatal("paid authorization queued free probe")
	}
	if err := r.Delete(ctx, b.ID); err != nil {
		t.Fatal(err)
	}
	var count int64
	r.db.db.Model(&accountRecoveryActivationModel{}).Count(&count)
	if count != 0 {
		t.Fatal("deleted account retained activation")
	}
}

func TestRecoveryActivationCompletionRollsBackOnGrantConsumptionFailure(t *testing.T) {
	r, _, b := reauthPair(t)
	ctx := context.Background()
	now := time.Now().UTC()
	lease := now.Add(5 * time.Minute)
	authorizeRecoveryActivation(t, r, b)
	if _, err := r.UpdateTokens(ctx, b.ID, "fresh-access", "fresh-refresh", now.Add(time.Hour), 0); err != nil {
		t.Fatal(err)
	}
	if err := r.SaveQuotaRecovery(ctx, account.QuotaRecovery{AccountID: b.ID, Kind: account.QuotaRecoveryKindFree, Status: account.QuotaRecoveryStatusProbing, NextProbeAt: &lease, UpdatedAt: now}); err != nil {
		t.Fatal(err)
	}
	if err := r.db.db.Exec("CREATE TRIGGER fail_activation_delete BEFORE DELETE ON account_recovery_activations BEGIN SELECT RAISE(ABORT, 'injected_activation_failure'); END").Error; err != nil {
		t.Fatal(err)
	}
	done, err := r.CompleteQuotaProbe(ctx, b.ID, lease, true, now)
	if err == nil || done {
		t.Fatalf("completion=%v err=%v", done, err)
	}
	got, _ := r.Get(ctx, b.ID)
	if got.Enabled {
		t.Fatal("failed grant consumption partially enabled account")
	}
	if _, err := r.GetQuotaRecovery(ctx, b.ID); err != nil {
		t.Fatal("failed grant consumption deleted quota lease")
	}
}

func TestRecoveryActivationChangedIdentityCannotQueueAfterRestart(t *testing.T) {
	r, _, b := reauthPair(t)
	ctx := context.Background()
	now := time.Now().UTC()
	authorizeRecoveryActivation(t, r, b)
	if _, err := r.UpdateTokens(ctx, b.ID, "fresh-access", "fresh-refresh", now.Add(time.Hour), 0); err != nil {
		t.Fatal(err)
	}
	if err := r.db.db.Model(&accountModel{}).Where("id = ?", b.ID).Update("email", "replaced@example.com").Error; err != nil {
		t.Fatal(err)
	}
	pending, err := r.ListPendingRecoveryActivationIDs(ctx, 10)
	if err != nil || len(pending) != 0 {
		t.Fatalf("pending=%v err=%v", pending, err)
	}
	queued, err := r.PrepareRecoveredActivationQuota(ctx, b.ID, now)
	if err != nil || queued {
		t.Fatalf("queue=%v err=%v", queued, err)
	}
}
