package account

import (
	"context"
	"errors"
	"time"

	accountdomain "github.com/chenyme/grok2api/backend/internal/domain/account"
	"github.com/chenyme/grok2api/backend/internal/repository"
)

type claimedQuotaRepository interface {
	CompleteQuotaProbe(context.Context, uint64, time.Time, bool, time.Time) (bool, error)
	SaveClaimedQuotaRecovery(context.Context, time.Time, accountdomain.QuotaRecovery) (bool, error)
}

// ProbePaidQuotaClaimed applies the existing billing recovery rules only while
// the supplied quota lease is current. It does not send a model request.
func (s *Service) ProbePaidQuotaClaimed(ctx context.Context, value accountdomain.Credential, leaseUntil time.Time) (bool, error) {
	if _, ok := s.accounts.(claimedQuotaRepository); !ok {
		return false, ErrUnsupported
	}
	state, err := s.accounts.GetQuotaRecovery(ctx, value.ID)
	if err != nil {
		return false, err
	}
	if state.Status != accountdomain.QuotaRecoveryStatusProbing || state.NextProbeAt == nil || !state.NextProbeAt.Equal(leaseUntil) || !leaseUntil.After(s.now()) {
		return false, repository.ErrConflict
	}
	return s.probePaidQuota(ctx, value, &leaseUntil)
}

func (s *Service) probePaidQuota(ctx context.Context, value accountdomain.Credential, leaseUntil *time.Time) (bool, error) {
	latest, billing, err := s.fetchAndSaveBilling(ctx, value.ID)
	writeCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), credentialStateWriteTimeout)
	defer cancel()
	if err != nil {
		now := time.Now().UTC()
		next := now.Add(paidProbeRetryInterval)
		writeErr := s.savePaidQuotaRecovery(writeCtx, leaseUntil, accountdomain.QuotaRecovery{AccountID: value.ID, Kind: accountdomain.QuotaRecoveryKindPaid, Status: accountdomain.QuotaRecoveryStatusExhausted, NextProbeAt: &next, UpdatedAt: now})
		return false, errors.Join(err, writeErr)
	}
	if ctx.Err() != nil {
		return false, ctx.Err()
	}
	if err := s.reconcilePaidQuotaRecoveryWithLease(writeCtx, latest, billing, true, leaseUntil); err != nil {
		return false, err
	}
	return !billing.IsExhausted(latest.MinimumRemaining), nil
}

func (s *Service) savePaidQuotaRecovery(ctx context.Context, leaseUntil *time.Time, value accountdomain.QuotaRecovery) error {
	if leaseUntil == nil {
		return s.accounts.SaveQuotaRecovery(ctx, value)
	}
	store, ok := s.accounts.(claimedQuotaRepository)
	if !ok {
		return ErrUnsupported
	}
	updated, err := store.SaveClaimedQuotaRecovery(ctx, *leaseUntil, value)
	if err != nil {
		return err
	}
	if !updated {
		return repository.ErrConflict
	}
	return nil
}

func (s *Service) completePaidQuotaRecovery(ctx context.Context, id uint64, leaseUntil *time.Time) error {
	if leaseUntil == nil {
		return s.accounts.ClearQuotaRecovery(ctx, id)
	}
	store, ok := s.accounts.(claimedQuotaRepository)
	if !ok {
		return ErrUnsupported
	}
	updated, err := store.CompleteQuotaProbe(ctx, id, *leaseUntil, true, time.Now().UTC())
	if err != nil {
		return err
	}
	if !updated {
		return repository.ErrConflict
	}
	return nil
}
