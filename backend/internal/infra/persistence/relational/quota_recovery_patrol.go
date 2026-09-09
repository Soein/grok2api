package relational

import (
	"context"
	"errors"
	"slices"
	"strings"
	"time"

	"github.com/chenyme/grok2api/backend/internal/domain/account"
	"github.com/chenyme/grok2api/backend/internal/repository"
	"gorm.io/gorm"
)

// ListDueBuildQuotaRecoveryIDs reads only due recovery rows; disabled accounts
// participate only when maintenance is explicitly allowed. It never enables them.
func (r *AccountRepository) ListDueBuildQuotaRecoveryIDs(ctx context.Context, now time.Time, includeDisabled bool, limit int) ([]uint64, error) {
	if limit <= 0 {
		return nil, nil
	}
	query := r.db.db.WithContext(ctx).Table("account_quota_recovery AS recovery").Select("recovery.account_id").
		Joins("JOIN provider_accounts AS account ON account.id = recovery.account_id").
		Where("account.provider = ? AND account.auth_status = ?", account.ProviderBuild, account.AuthStatusActive).
		Where("recovery.status IN ? AND recovery.next_probe_at IS NOT NULL AND recovery.next_probe_at <= ?", []string{"exhausted", "probing"}, now.UTC()).
		Where("account.cooldown_until IS NULL OR account.cooldown_until <= ?", now.UTC())
	if !includeDisabled {
		query = query.Where("account.enabled = ?", true)
	}
	var ids []uint64
	err := query.Order("recovery.next_probe_at ASC, recovery.account_id ASC").Limit(min(limit, 1000)).Scan(&ids).Error
	return ids, err
}

// DeferUnclaimedQuotaRecovery prevents locally ineligible head candidates from
// starving later accounts. It cannot change a live claim or a newer backoff.
func (r *AccountRepository) DeferUnclaimedQuotaRecovery(ctx context.Context, id uint64, now, retryAt time.Time) error {
	if !retryAt.After(now) {
		return nil
	}
	result := r.db.db.WithContext(ctx).Model(&quotaRecoveryModel{}).
		Where("account_id = ? AND status IN ? AND next_probe_at <= ?", id, []string{"exhausted", "probing"}, now.UTC()).
		Updates(map[string]any{"status": account.QuotaRecoveryStatusExhausted, "next_probe_at": retryAt.UTC(), "updated_at": now.UTC()})
	if result.Error == nil && result.RowsAffected > 0 {
		r.notifyInvalidation(ctx, repository.InvalidationEvent{Kind: repository.InvalidationAccountRecoveryChanged, AccountID: id})
	}
	return result.Error
}

// GetQuotaRecoveryCandidate loads one maintenance candidate including its
// current credentials. Callers must enforce their disabled-account policy.
// A route's explicit account bindings remain authoritative for maintenance.
func (r *AccountRepository) GetQuotaRecoveryCandidate(ctx context.Context, accountID, modelRouteID uint64, upstreamModel, quotaMode string) (account.RoutingCandidate, error) {
	value, err := r.Get(ctx, accountID)
	if err != nil {
		return account.RoutingCandidate{}, err
	}
	candidate := account.RoutingCandidate{Credential: value}
	if billing, err := r.GetBilling(ctx, accountID); err == nil {
		candidate.Billing = &billing
	} else if !errors.Is(err, repository.ErrNotFound) {
		return candidate, err
	}
	if recovery, err := r.GetQuotaRecovery(ctx, accountID); err == nil {
		candidate.QuotaRecovery = &recovery
	} else if !errors.Is(err, repository.ErrNotFound) {
		return candidate, err
	}
	if quotaMode != "" {
		var window quotaWindowModel
		err := r.db.db.WithContext(ctx).Where("account_id = ? AND mode = ?", accountID, quotaMode).Take(&window).Error
		if err == nil {
			converted := toRoutingQuotaWindowDomain(window)
			candidate.QuotaWindow = &converted
		} else if !errors.Is(err, gorm.ErrRecordNotFound) {
			return candidate, err
		}
	}
	now := time.Now().UTC()
	var leaseBlocks []accountEgressLeaseBlockModel
	query := r.db.db.WithContext(ctx).Where("account_id = ? AND cooldown_until > ?", accountID, now)
	if value.EgressNodeID != 0 {
		query = query.Where("node_id = ?", value.EgressNodeID)
	}
	if err := query.Order("cooldown_until DESC").Limit(1).Find(&leaseBlocks).Error; err != nil {
		return candidate, err
	}
	if len(leaseBlocks) != 0 {
		block := egressLeaseBlockFromModel(leaseBlocks[0])
		candidate.EgressLeaseBlock = &block
	}
	upstreamModel = strings.TrimSpace(upstreamModel)
	if upstreamModel == "" {
		return candidate, nil
	}
	bound, err := r.listRoutingBoundAccountIDs(ctx, value.Provider, modelRouteID, upstreamModel)
	if err != nil {
		return candidate, err
	}
	if len(bound) > 0 && !slices.Contains(bound, accountID) {
		return candidate, repository.ErrNotFound
	}
	var known, supported int64
	if err := r.db.db.WithContext(ctx).Model(&accountModelSyncStateModel{}).Where("account_id = ? AND last_success_at IS NOT NULL", accountID).Count(&known).Error; err != nil {
		return candidate, err
	}
	if err := r.db.db.WithContext(ctx).Model(&accountModelCapabilityModel{}).Where("account_id = ? AND upstream_model = ?", accountID, upstreamModel).Count(&supported).Error; err != nil {
		return candidate, err
	}
	candidate.ModelCapabilityKnown, candidate.SupportsModel = known > 0, supported > 0
	if len(bound) > 0 || (value.Provider == account.ProviderConsole && quotaMode != "") || (value.Provider == account.ProviderWeb && account.IsWebImagineQuotaMode(quotaMode)) {
		candidate.ModelCapabilityKnown, candidate.SupportsModel = true, true
	}
	var blocks []accountModelQuotaBlockModel
	if err := r.db.db.WithContext(ctx).Where("account_id = ? AND upstream_model = ? AND cooldown_until > ?", accountID, upstreamModel, now).Limit(1).Find(&blocks).Error; err != nil {
		return candidate, err
	}
	if len(blocks) != 0 {
		row := blocks[0]
		candidate.ModelQuotaBlock = &account.ModelQuotaBlock{AccountID: row.AccountID, UpstreamModel: row.UpstreamModel, Reason: row.Reason, CooldownUntil: row.CooldownUntil.UTC(), UpdatedAt: row.UpdatedAt.UTC()}
	}
	return candidate, nil
}

// CompleteQuotaProbe accepts only the still-live lease that observed the result.
// A failed/incomplete stream keeps its original retry time; a newer claim or
// authoritative exhaustion update is never overwritten by a late finalizer.
func (r *AccountRepository) CompleteQuotaProbe(ctx context.Context, id uint64, leaseUntil time.Time, recovered bool, now time.Time) (bool, error) {
	completed, activated := false, false
	err := r.db.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		query := tx.Where("account_id = ? AND status = ? AND next_probe_at = ? AND next_probe_at > ?", id, account.QuotaRecoveryStatusProbing, leaseUntil.UTC(), now.UTC())
		var result *gorm.DB
		if recovered {
			result = query.Delete(&quotaRecoveryModel{})
		} else {
			result = query.Model(&quotaRecoveryModel{}).Update("updated_at", now.UTC())
		}
		if result.Error != nil {
			return result.Error
		}
		completed = result.RowsAffected == 1
		if completed && recovered {
			var activationErr error
			activated, activationErr = activateRecoveredAccount(tx, id, now)
			return activationErr
		}
		return nil
	})
	if err != nil {
		return false, err
	}
	if completed && recovered {
		r.notifyInvalidation(ctx, repository.InvalidationEvent{Kind: repository.InvalidationAccountRecoveryChanged, AccountID: id})
		if activated {
			r.notifyInvalidation(ctx, repository.InvalidationEvent{Kind: repository.InvalidationAccountStateChanged, Provider: account.ProviderBuild, AccountID: id})
		}
	}
	return completed, nil
}

// SaveClaimedQuotaRecovery persists existing backoff semantics only while the
// caller still owns the probe. It cannot replace another request's recovery.
func (r *AccountRepository) SaveClaimedQuotaRecovery(ctx context.Context, leaseUntil time.Time, value account.QuotaRecovery) (bool, error) {
	result := r.db.db.WithContext(ctx).Model(&quotaRecoveryModel{}).
		Where("account_id = ? AND status = ? AND next_probe_at = ? AND next_probe_at > ?", value.AccountID, account.QuotaRecoveryStatusProbing, leaseUntil.UTC(), value.UpdatedAt.UTC()).
		Updates(map[string]any{"kind": value.Kind, "status": value.Status, "confirmed_used": value.ConfirmedUsed, "confirmed_limit": value.ConfirmedLimit, "exhausted_at": value.ExhaustedAt, "next_probe_at": value.NextProbeAt, "last_confirmed_at": value.LastConfirmedAt, "updated_at": value.UpdatedAt})
	if result.Error == nil && result.RowsAffected == 1 {
		r.notifyInvalidation(ctx, repository.InvalidationEvent{Kind: repository.InvalidationAccountRecoveryChanged, AccountID: value.AccountID})
	}
	return result.RowsAffected == 1, result.Error
}
