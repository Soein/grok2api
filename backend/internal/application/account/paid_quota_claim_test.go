package account

import (
	"context"
	"testing"
	"time"

	accountdomain "github.com/chenyme/grok2api/backend/internal/domain/account"
)

func TestPaidQuotaClaimDoesNotClearNewerLease(t *testing.T) {
	now := time.Now().UTC().Truncate(time.Second)
	service, credential, adapter := newCredentialRefreshTestService(t, now)
	adapter.billing = accountdomain.Billing{MonthlyLimit: 100, Used: 0, BillingPeriodEnd: now.Add(time.Hour).Format(time.RFC3339)}
	currentLease := now.Add(5 * time.Minute)
	if err := service.accounts.SaveQuotaRecovery(context.Background(), accountdomain.QuotaRecovery{AccountID: credential.ID, Kind: accountdomain.QuotaRecoveryKindPaid, Status: accountdomain.QuotaRecoveryStatusProbing, NextProbeAt: &currentLease, UpdatedAt: now}); err != nil {
		t.Fatal(err)
	}
	recovered, err := service.ProbePaidQuotaClaimed(context.Background(), credential, now.Add(time.Minute))
	if recovered || err == nil {
		t.Fatalf("stale paid probe succeeded: %v %v", recovered, err)
	}
	state, err := service.accounts.GetQuotaRecovery(context.Background(), credential.ID)
	if err != nil || state.NextProbeAt == nil || !state.NextProbeAt.Equal(currentLease) {
		t.Fatal("newer lease was lost", err)
	}
}

func TestPaidQuotaClaimRetainsOriginalExhaustionBackoff(t *testing.T) {
	now := time.Now().UTC().Truncate(time.Second)
	service, credential, adapter := newCredentialRefreshTestService(t, now)
	adapter.billing = accountdomain.Billing{MonthlyLimit: 100, Used: 100, BillingPeriodEnd: now.Add(-time.Hour).Format(time.RFC3339)}
	lease := now.Add(5 * time.Minute)
	if err := service.accounts.SaveQuotaRecovery(context.Background(), accountdomain.QuotaRecovery{AccountID: credential.ID, Kind: accountdomain.QuotaRecoveryKindPaid, Status: accountdomain.QuotaRecoveryStatusProbing, NextProbeAt: &lease, UpdatedAt: now}); err != nil {
		t.Fatal(err)
	}
	if recovered, err := service.ProbePaidQuotaClaimed(context.Background(), credential, lease); err != nil || recovered {
		t.Fatalf("result=%v %v", recovered, err)
	}
	state, err := service.accounts.GetQuotaRecovery(context.Background(), credential.ID)
	if err != nil || state.Status != accountdomain.QuotaRecoveryStatusExhausted || state.NextProbeAt.Before(now.Add(14*time.Minute)) {
		t.Fatal("paid backoff missing", err)
	}
}
