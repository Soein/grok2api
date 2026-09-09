package relational

import (
	"context"
	"errors"
	"time"

	"github.com/chenyme/grok2api/backend/internal/domain/account"
	"github.com/chenyme/grok2api/backend/internal/repository"
	"gorm.io/gorm"
)

var _ repository.ReauthRepository = (*AccountRepository)(nil)

func (r *AccountRepository) buildReauthCandidates(ctx context.Context, now time.Time, includeDisabled bool) *gorm.DB {
	q := r.db.db.WithContext(ctx).Table("provider_accounts AS b").
		Joins("JOIN account_credentials AS bc ON bc.account_id = b.id").
		Joins("JOIN account_provider_links AS l ON l.build_account_id = b.id").
		Joins("JOIN provider_accounts AS w ON w.id = l.web_account_id").
		Joins("JOIN account_credentials AS wc ON wc.account_id = w.id").
		Where("b.provider = ? AND b.auth_status = ? AND bc.auth_type = ?", account.ProviderBuild, account.AuthStatusReauthRequired, account.AuthTypeOAuth).
		Where("w.provider = ? AND w.auth_status = ? AND wc.auth_type = ? AND wc.encrypted_primary <> ''", account.ProviderWeb, account.AuthStatusActive, account.AuthTypeSSO).
		Where("bc.refresh_due_at IS NULL OR bc.refresh_due_at <= ?", now.UTC()).
		Where("b.cooldown_until IS NULL OR b.cooldown_until <= ?", now.UTC()).
		Where("w.cooldown_until IS NULL OR w.cooldown_until <= ?", now.UTC())
	if !includeDisabled {
		q = q.Where("b.enabled = ?", true)
	}
	return q
}

func (r *AccountRepository) ListBuildReauthCandidates(ctx context.Context, now time.Time, includeDisabled bool, limit int) ([]uint64, error) {
	if limit <= 0 {
		return nil, nil
	}
	var ids []uint64
	err := r.buildReauthCandidates(ctx, now, includeDisabled).Select("b.id").Order("bc.refresh_due_at ASC, b.id ASC").Limit(limit).Scan(&ids).Error
	return ids, err
}

func (r *AccountRepository) ClaimBuildReauth(ctx context.Context, build, web account.Credential, now, leaseUntil time.Time, includeDisabled bool) (bool, error) {
	candidates := r.buildReauthCandidates(ctx, now, includeDisabled).Select("b.id").Where("b.id = ? AND w.id = ? AND bc.encrypted_primary = ? AND bc.encrypted_refresh = ? AND bc.client_id = ? AND wc.encrypted_primary = ?", build.ID, web.ID, build.EncryptedAccessToken, build.EncryptedRefreshToken, build.OIDCClientID, web.EncryptedAccessToken)
	result := r.db.db.WithContext(ctx).Model(&accountCredentialModel{}).Where("account_id IN (?)", candidates).Updates(map[string]any{"refresh_due_at": leaseUntil.UTC(), "updated_at": now.UTC()})
	return result.RowsAffected == 1, result.Error
}

// SaveBuildReauth commits the token pair and client ID together. Compare-and-swap
// protects concurrent imports/refreshes; the link and both identities must still
// match the snapshots checked before the upstream authorization.
func (r *AccountRepository) SaveBuildReauth(ctx context.Context, build, web account.Credential, access, refresh, clientID string, expires time.Time, botFlag int) (bool, error) {
	conflict := errors.New("reauth_compare_and_swap_conflict")
	now := time.Now().UTC()
	err := r.db.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		peers := tx.Table("account_provider_links AS l").Select("l.build_account_id").
			Joins("JOIN provider_accounts AS b ON b.id = l.build_account_id").
			Joins("JOIN provider_accounts AS w ON w.id = l.web_account_id").
			Joins("JOIN account_credentials AS wc ON wc.account_id = w.id").
			Where("l.build_account_id = ? AND l.web_account_id = ?", build.ID, web.ID).
			Where("b.provider = ? AND b.auth_status = ? AND b.user_id = ? AND b.email = ?", account.ProviderBuild, build.AuthStatus, build.UserID, build.Email).
			Where("w.provider = ? AND w.auth_status = ? AND w.user_id = ? AND w.email = ? AND wc.encrypted_primary = ?", account.ProviderWeb, web.AuthStatus, web.UserID, web.Email, web.EncryptedAccessToken)
		result := tx.Model(&accountCredentialModel{}).
			Where("account_id IN (?) AND auth_type = ? AND encrypted_primary = ? AND encrypted_refresh = ? AND client_id = ?", peers, account.AuthTypeOAuth, build.EncryptedAccessToken, build.EncryptedRefreshToken, build.OIDCClientID).
			Updates(map[string]any{"encrypted_primary": access, "encrypted_refresh": refresh, "client_id": clientID, "expires_at": expires, "refresh_due_at": account.CredentialRefreshDueAt(build.ID, expires), "last_refresh_at": now, "refresh_failures": 0, "refresh_unclassified_auth_failures": 0, "last_refresh_error_status": 0, "last_refresh_error": "", "last_refresh_error_message": "", "last_refresh_error_response": "", "refresh_permanent": false, "build_bot_flag_source": normalizeBuildBotFlagSource(account.ProviderBuild, botFlag), "updated_at": now})
		if result.Error != nil {
			return result.Error
		}
		if result.RowsAffected != 1 {
			return conflict
		}
		return tx.Model(&accountModel{}).Where("id = ?", build.ID).Updates(map[string]any{"auth_status": account.AuthStatusActive, "reauth_marked_at": nil, "last_error": "", "updated_at": now}).Error
	})
	if errors.Is(err, conflict) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	r.notifyInvalidation(ctx, repository.InvalidationEvent{Kind: repository.InvalidationAccountCredentialChanged, Provider: account.ProviderBuild, AccountID: build.ID})
	return true, nil
}

func (r *AccountRepository) FailBuildReauth(ctx context.Context, build account.Credential, leaseUntil, retryAt time.Time, code string) error {
	accounts := r.db.db.WithContext(ctx).Model(&accountModel{}).Select("id").Where("id = ? AND provider = ? AND auth_status = ?", build.ID, account.ProviderBuild, account.AuthStatusReauthRequired)
	return r.db.db.WithContext(ctx).Model(&accountCredentialModel{}).
		Where("account_id IN (?) AND encrypted_primary = ? AND encrypted_refresh = ? AND refresh_due_at = ?", accounts, build.EncryptedAccessToken, build.EncryptedRefreshToken, leaseUntil.UTC()).
		Updates(map[string]any{"refresh_due_at": retryAt.UTC(), "refresh_failures": gorm.Expr("refresh_failures + 1"), "last_refresh_error": code, "last_refresh_error_message": "", "last_refresh_error_response": "", "updated_at": time.Now().UTC()}).Error
}

// RejectReauthSSO marks only the rejected SSO snapshot, so an administrator's
// concurrent replacement cannot be invalidated by the old session's response.
func (r *AccountRepository) RejectReauthSSO(ctx context.Context, web account.Credential) error {
	if web.Provider != account.ProviderWeb || web.AuthType != account.AuthTypeSSO || web.AuthStatus != account.AuthStatusActive {
		return nil
	}
	_, err := r.MarkAccountReauthRequired(ctx, web, "Grok Web SSO credential rejected", time.Now().UTC())
	return err
}
