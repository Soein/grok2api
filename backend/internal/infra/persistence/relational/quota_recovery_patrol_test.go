package relational

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/chenyme/grok2api/backend/internal/domain/account"
	"github.com/chenyme/grok2api/backend/internal/repository"
)

func quotaPatrolRepository(t *testing.T) *AccountRepository {
	t.Helper()
	db, err := OpenSQLite(context.Background(), filepath.Join(t.TempDir(), "patrol.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = db.Close() })
	if err := db.InitializeSchema(context.Background()); err != nil {
		t.Fatal(err)
	}
	return NewAccountRepository(db)
}

func addQuotaPatrolAccount(t *testing.T, repo *AccountRepository, id int, enabled bool, auth account.AuthStatus, due time.Time) account.Credential {
	t.Helper()
	value, _, err := repo.UpsertByIdentity(context.Background(), account.Credential{
		Provider: account.ProviderBuild, AuthType: account.AuthTypeOAuth, Name: fmt.Sprintf("patrol-%d", id), SourceKey: fmt.Sprintf("patrol-%d", id),
		Enabled: enabled, AuthStatus: auth, EncryptedAccessToken: "encrypted", Priority: 1, MaxConcurrent: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !enabled {
		value.Enabled = false
		value, err = repo.Update(context.Background(), value)
		if err != nil {
			t.Fatal(err)
		}
	}
	if err := repo.SaveQuotaRecovery(context.Background(), account.QuotaRecovery{AccountID: value.ID, Kind: account.QuotaRecoveryKindFree, Status: account.QuotaRecoveryStatusExhausted, NextProbeAt: &due, UpdatedAt: time.Now().UTC()}); err != nil {
		t.Fatal(err)
	}
	return value
}

func TestQuotaPatrolSelectsDueAndRespectsDisabledPolicy(t *testing.T) {
	repo := quotaPatrolRepository(t)
	now := time.Now().UTC()
	one := addQuotaPatrolAccount(t, repo, 1, true, account.AuthStatusActive, now.Add(-time.Hour))
	disabled := addQuotaPatrolAccount(t, repo, 2, false, account.AuthStatusActive, now.Add(-time.Minute))
	addQuotaPatrolAccount(t, repo, 3, true, account.AuthStatusReauthRequired, now.Add(-time.Minute))
	addQuotaPatrolAccount(t, repo, 4, true, account.AuthStatusActive, now.Add(time.Minute))
	cooling := addQuotaPatrolAccount(t, repo, 5, true, account.AuthStatusActive, now.Add(-time.Minute))
	until := now.Add(time.Hour)
	cooling.CooldownUntil = &until
	if _, err := repo.Update(context.Background(), cooling); err != nil {
		t.Fatal(err)
	}
	ids, err := repo.ListDueBuildQuotaRecoveryIDs(context.Background(), now, false, 10)
	if err != nil || !slices.Equal(ids, []uint64{one.ID}) {
		t.Fatalf("ids=%v err=%v", ids, err)
	}
	ids, err = repo.ListDueBuildQuotaRecoveryIDs(context.Background(), now, true, 10)
	if err != nil || !slices.Equal(ids, []uint64{one.ID, disabled.ID}) {
		t.Fatalf("include disabled ids=%v err=%v", ids, err)
	}
	ids, err = repo.ListDueBuildQuotaRecoveryIDs(context.Background(), now, true, 1)
	if err != nil || len(ids) != 1 {
		t.Fatalf("bounded ids=%v err=%v", ids, err)
	}
}

func TestQuotaPatrolClaimIsAtomicAndCompletionIsFenced(t *testing.T) {
	repo := quotaPatrolRepository(t)
	ctx := context.Background()
	now := time.Now().UTC().Truncate(time.Second)
	value := addQuotaPatrolAccount(t, repo, 1, true, account.AuthStatusActive, now.Add(-time.Minute))
	lease := now.Add(5 * time.Minute)
	var claims atomic.Int32
	var workers sync.WaitGroup
	for range 8 {
		workers.Go(func() {
			claimed, err := repo.ClaimQuotaProbe(ctx, value.ID, now, lease)
			if err != nil {
				t.Error(err)
			}
			if claimed {
				claims.Add(1)
			}
		})
	}
	workers.Wait()
	if claims.Load() != 1 {
		t.Fatalf("claims=%d", claims.Load())
	}
	if ok, err := repo.CompleteQuotaProbe(ctx, value.ID, lease.Add(-time.Second), true, now); err != nil || ok {
		t.Fatalf("stale completion=%v err=%v", ok, err)
	}
	if ok, err := repo.CompleteQuotaProbe(ctx, value.ID, lease, false, now); err != nil || !ok {
		t.Fatalf("failed completion=%v err=%v", ok, err)
	}
	recovery, err := repo.GetQuotaRecovery(ctx, value.ID)
	if err != nil || recovery.NextProbeAt == nil || !recovery.NextProbeAt.Equal(lease) {
		t.Fatalf("failure lost retry lease: %+v err=%v", recovery, err)
	}
	if ok, err := repo.CompleteQuotaProbe(ctx, value.ID, lease, true, now); err != nil || !ok {
		t.Fatalf("completion=%v err=%v", ok, err)
	}
	if _, err := repo.GetQuotaRecovery(ctx, value.ID); !errors.Is(err, repository.ErrNotFound) {
		t.Fatalf("recovery not cleared: %v", err)
	}
}

func TestQuotaPatrolCandidateLoadsDisabledWithoutChangingState(t *testing.T) {
	repo := quotaPatrolRepository(t)
	ctx := context.Background()
	now := time.Now().UTC()
	value := addQuotaPatrolAccount(t, repo, 1, false, account.AuthStatusActive, now.Add(-time.Minute))
	if err := repo.db.db.Create(&accountModelSyncStateModel{AccountID: value.ID, LastAttemptAt: now, LastSuccessAt: &now}).Error; err != nil {
		t.Fatal(err)
	}
	if err := repo.db.db.Create(&accountModelCapabilityModel{AccountID: value.ID, UpstreamModel: "grok-4.5"}).Error; err != nil {
		t.Fatal(err)
	}
	until := now.Add(time.Hour)
	if err := repo.UpsertModelQuotaBlock(ctx, account.ModelQuotaBlock{AccountID: value.ID, UpstreamModel: "grok-4.5", Reason: "model_quota_depleted", CooldownUntil: until, UpdatedAt: now}); err != nil {
		t.Fatal(err)
	}
	candidate, err := repo.GetQuotaRecoveryCandidate(ctx, value.ID, 0, "grok-4.5", "")
	if err != nil {
		t.Fatal(err)
	}
	if candidate.Credential.Enabled || candidate.Credential.EncryptedAccessToken == "" || !candidate.SupportsModel || !candidate.ModelCapabilityKnown || candidate.ModelQuotaBlock == nil || candidate.QuotaRecovery == nil {
		t.Fatal("candidate missing required state or enabled a disabled account")
	}
}

func TestQuotaPatrolClaimedUpdateCannotOverwriteNewClaim(t *testing.T) {
	repo := quotaPatrolRepository(t)
	ctx := context.Background()
	now := time.Now().UTC().Truncate(time.Second)
	value := addQuotaPatrolAccount(t, repo, 1, true, account.AuthStatusActive, now.Add(-time.Minute))
	oldLease := now.Add(time.Minute)
	newLease := now.Add(10 * time.Minute)
	if ok, err := repo.ClaimQuotaProbe(ctx, value.ID, now, oldLease); err != nil || !ok {
		t.Fatal("claim failed", err)
	}
	if ok, err := repo.ClaimQuotaProbe(ctx, value.ID, oldLease.Add(time.Second), newLease); err != nil || !ok {
		t.Fatal("reclaim failed", err)
	}
	retry := now.Add(24 * time.Hour)
	updated, err := repo.SaveClaimedQuotaRecovery(ctx, oldLease, account.QuotaRecovery{AccountID: value.ID, Kind: account.QuotaRecoveryKindPaid, Status: account.QuotaRecoveryStatusExhausted, NextProbeAt: &retry, UpdatedAt: now})
	if err != nil || updated {
		t.Fatalf("stale update=%v err=%v", updated, err)
	}
	state, err := repo.GetQuotaRecovery(ctx, value.ID)
	if err != nil || state.NextProbeAt == nil || !state.NextProbeAt.Equal(newLease) {
		t.Fatal("new claim was overwritten", err)
	}
}

func TestQuotaPatrolDefersSkippedCandidateWithoutOverwritingActiveLease(t *testing.T) {
	repo := quotaPatrolRepository(t)
	ctx := context.Background()
	now := time.Now().UTC().Truncate(time.Second)
	value := addQuotaPatrolAccount(t, repo, 1, true, account.AuthStatusActive, now.Add(-time.Minute))
	if err := repo.DeferUnclaimedQuotaRecovery(ctx, value.ID, now, now.Add(time.Minute)); err != nil {
		t.Fatal(err)
	}
	if ids, err := repo.ListDueBuildQuotaRecoveryIDs(ctx, now, false, 10); err != nil || len(ids) != 0 {
		t.Fatal("skipped head was not delayed", ids, err)
	}
	claimAt := now.Add(2 * time.Minute)
	lease := now.Add(7 * time.Minute)
	if claimed, err := repo.ClaimQuotaProbe(ctx, value.ID, claimAt, lease); err != nil || !claimed {
		t.Fatal("claim failed", err)
	}
	if err := repo.DeferUnclaimedQuotaRecovery(ctx, value.ID, claimAt, claimAt.Add(time.Minute)); err != nil {
		t.Fatal(err)
	}
	state, err := repo.GetQuotaRecovery(ctx, value.ID)
	if err != nil || state.NextProbeAt == nil || !state.NextProbeAt.Equal(lease) {
		t.Fatal("active lease overwritten", err)
	}
}
