package relational

import (
	"context"
	"time"

	"github.com/chenyme/grok2api/backend/internal/domain/account"
	"github.com/chenyme/grok2api/backend/internal/repository"
	"gorm.io/gorm"
)

var _ repository.AccountManagementWriter = (*AccountRepository)(nil)
var _ repository.AccountAuthenticationStateWriter = (*AccountRepository)(nil)

// UpdateAccountManagement never serializes a previously read Credential. Only
// explicit management fields are written, so concurrent identity enrichment,
// health feedback and credential rotation survive a delayed administrator edit.
func (r *AccountRepository) UpdateAccountManagement(ctx context.Context, id uint64, patch repository.ManagementAccountUpdates) (account.Credential, error) {
	updates := map[string]any{}
	if patch.Name != nil {
		updates["name"] = *patch.Name
	}
	if patch.Enabled != nil {
		updates["enabled"] = *patch.Enabled
	}
	if patch.Priority != nil {
		updates["priority"] = *patch.Priority
	}
	if patch.MaxConcurrent != nil {
		updates["max_concurrent"] = *patch.MaxConcurrent
	}
	if patch.MinimumRemaining != nil {
		updates["minimum_remaining"] = *patch.MinimumRemaining
	}
	if patch.BuildSuperEntitled != nil {
		updates["build_super_entitled"] = *patch.BuildSuperEntitled
	}
	if patch.BuildRouteMode != nil {
		updates["build_route_mode"] = string(*patch.BuildRouteMode)
	}
	now := time.Now().UTC()
	var storedProvider string
	err := r.db.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		var stored accountModel
		if err := tx.Select("id", "provider").First(&stored, id).Error; err != nil {
			return err
		}
		storedProvider = stored.Provider
		if len(updates) > 0 {
			updates["updated_at"] = now
			if err := tx.Model(&accountModel{}).Where("id = ?", id).Updates(updates).Error; err != nil {
				return err
			}
		}
		if patch.EncryptedCloudflareCookie != nil {
			if err := tx.Model(&accountCredentialModel{}).Where("account_id = ?", id).Updates(map[string]any{"encrypted_cloudflare_cookie": *patch.EncryptedCloudflareCookie, "updated_at": now}).Error; err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		return account.Credential{}, mapError(err)
	}
	kind := repository.InvalidationAccountStateChanged
	if patch.EncryptedCloudflareCookie != nil {
		kind = repository.InvalidationAccountCredentialChanged
	}
	r.notifyInvalidation(ctx, repository.InvalidationEvent{Kind: kind, Provider: account.Provider(storedProvider), AccountID: id})
	return r.Get(ctx, id)
}

// MarkAccountReauthRequired compares the failed credential snapshot and changes
// only authentication metadata. In-flight failures cannot invalidate a newer
// token pair, rewrite account settings, or reset an existing reauth age anchor.
func (r *AccountRepository) MarkAccountReauthRequired(ctx context.Context, value account.Credential, reason string, now time.Time) (bool, error) {
	credential := r.db.db.WithContext(ctx).Model(&accountCredentialModel{}).Select("account_id").
		Where("account_id = ? AND auth_type = ? AND encrypted_primary = ? AND encrypted_refresh = ? AND client_id = ?", value.ID, value.AuthType, value.EncryptedAccessToken, value.EncryptedRefreshToken, value.OIDCClientID)
	result := r.db.db.WithContext(ctx).Model(&accountModel{}).
		Where("id IN (?) AND provider = ? AND auth_status = ?", credential, value.Provider, value.AuthStatus).
		Updates(map[string]any{"auth_status": account.AuthStatusReauthRequired, "last_error": truncate(reason, 512), "reauth_marked_at": gorm.Expr("CASE WHEN auth_status = ? AND reauth_marked_at IS NOT NULL THEN reauth_marked_at ELSE ? END", account.AuthStatusReauthRequired, now.UTC()), "updated_at": now.UTC()})
	if result.Error == nil && result.RowsAffected > 0 {
		r.notifyInvalidation(ctx, repository.InvalidationEvent{Kind: repository.InvalidationAccountStateChanged, Provider: value.Provider, AccountID: value.ID})
	}
	return result.RowsAffected > 0, mapError(result.Error)
}
