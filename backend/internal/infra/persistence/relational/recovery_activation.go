package relational

import (
	"context"
	"errors"
	"time"

	"github.com/chenyme/grok2api/backend/internal/domain/account"
	"github.com/chenyme/grok2api/backend/internal/repository"
	"gorm.io/gorm"
	"gorm.io/gorm/clause"
)

// Entries are inserted only from an administrator-reviewed migration manifest.
// There is deliberately no automatic enrollment based on source prefixes or
// authentication state; a disabled account without this grant stays disabled.
type accountRecoveryActivationModel struct {
	AccountID uint64        `gorm:"primaryKey"`
	SourceKey string        `gorm:"size:512;not null;check:chk_recovery_activation_source,length(trim(source_key)) BETWEEN 1 AND 512"`
	UserID    string        `gorm:"size:255;not null;default:''"`
	Email     string        `gorm:"size:255;not null;default:''"`
	CreatedAt time.Time     `gorm:"not null"`
	Account   *accountModel `gorm:"foreignKey:AccountID;references:ID;constraint:OnUpdate:CASCADE,OnDelete:CASCADE"`
}

func (accountRecoveryActivationModel) TableName() string { return "account_recovery_activations" }

var _ repository.RecoveryActivationRepository = (*AccountRepository)(nil)

func eligibleRecoveryActivations(db *gorm.DB) *gorm.DB {
	return db.Table("account_recovery_activations AS activation").
		Joins("JOIN provider_accounts AS a ON a.id = activation.account_id").
		Where("a.provider = ? AND a.enabled = ? AND a.auth_status = ?", account.ProviderBuild, false, account.AuthStatusActive).
		Where("a.source_key = activation.source_key AND a.user_id = activation.user_id AND a.email = activation.email")
}

func (r *AccountRepository) ListPendingRecoveryActivationIDs(ctx context.Context, limit int) ([]uint64, error) {
	if limit <= 0 {
		return nil, nil
	}
	var ids []uint64
	err := eligibleRecoveryActivations(r.db.db.WithContext(ctx)).Select("activation.account_id").
		Where("NOT EXISTS (SELECT 1 FROM account_quota_recovery AS recovery WHERE recovery.account_id = activation.account_id)").
		Order("activation.account_id ASC").Limit(min(limit, 1000)).Scan(&ids).Error
	return ids, err
}

// PrepareRecoveredActivationQuota is restart-safe and never changes a previous
// quota result or retry deadline. The grant must still match the same account.
func (r *AccountRepository) PrepareRecoveredActivationQuota(ctx context.Context, id uint64, now time.Time) (bool, error) {
	now = now.UTC()
	inserted := false
	err := r.db.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		var ids []uint64
		if err := eligibleRecoveryActivations(tx).Select("activation.account_id").Where("a.id = ?", id).Limit(1).Scan(&ids).Error; err != nil {
			return err
		}
		if len(ids) == 0 {
			return nil
		}
		kind := account.QuotaRecoveryKindFree
		var billing billingModel
		billingErr := tx.Where("account_id = ?", id).Take(&billing).Error
		if billingErr == nil && toRoutingBillingDomain(billing).IsPaid() {
			kind = account.QuotaRecoveryKindPaid
		} else if billingErr != nil && !errors.Is(billingErr, gorm.ErrRecordNotFound) {
			return billingErr
		}
		row := quotaRecoveryModel{AccountID: id, Kind: string(kind), Status: string(account.QuotaRecoveryStatusExhausted), NextProbeAt: &now, UpdatedAt: now}
		result := tx.Clauses(clause.OnConflict{Columns: []clause.Column{{Name: "account_id"}}, DoNothing: true}).Create(&row)
		inserted = result.RowsAffected == 1
		return result.Error
	})
	if err != nil {
		return false, err
	}
	if inserted {
		r.notifyInvalidation(ctx, repository.InvalidationEvent{Kind: repository.InvalidationAccountRecoveryChanged, Provider: account.ProviderBuild, AccountID: id})
	}
	return inserted, err
}

func (r *AccountRepository) RevokeRecoveryActivations(ctx context.Context, providerValue account.Provider, ids []uint64) error {
	if providerValue != account.ProviderBuild || len(ids) == 0 {
		return nil
	}
	return r.db.db.WithContext(ctx).Where("account_id IN ?", ids).Delete(&accountRecoveryActivationModel{}).Error
}

// activateRecoveredAccount runs only inside the successful quota-lease CAS
// transaction. A manual enablement update revokes the grant before its write.
func activateRecoveredAccount(tx *gorm.DB, id uint64, now time.Time) (bool, error) {
	eligible := eligibleRecoveryActivations(tx).Select("activation.account_id").Where("a.id = ?", id).
		Where("a.cooldown_until IS NULL OR a.cooldown_until <= ?", now.UTC())
	result := tx.Model(&accountModel{}).Where("id IN (?)", eligible).Updates(map[string]any{"enabled": true, "updated_at": now.UTC()})
	if result.Error != nil {
		return false, result.Error
	}
	if result.RowsAffected == 0 {
		return false, nil
	}
	if err := tx.Where("account_id = ?", id).Delete(&accountRecoveryActivationModel{}).Error; err != nil {
		return false, err
	}
	return true, nil
}
