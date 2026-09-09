package relational

import (
	"context"
	"time"

	account "github.com/chenyme/grok2api/backend/internal/domain/account"
)

// ListDueQuotaWindowsForRecovery filters account policy before applying the
// bounded reconciliation limit, so disabled or invalid pools cannot starve it.
func (r *AccountRepository) ListDueQuotaWindowsForRecovery(ctx context.Context, now time.Time, limit int, allowed []account.Provider, includeDisabled bool) ([]account.QuotaWindow, error) {
	return r.ListDueQuotaWindowsForRecoveryAfter(ctx, now, limit, nil, allowed, includeDisabled)
}

// ListDueQuotaWindowsForRecoveryAfter advances reconciliation past pending
// windows without overriding their existing queue leases or failure backoff.
func (r *AccountRepository) ListDueQuotaWindowsForRecoveryAfter(ctx context.Context, now time.Time, limit int, after *account.QuotaWindow, allowed []account.Provider, includeDisabled bool) ([]account.QuotaWindow, error) {
	providers := make([]account.Provider, 0, len(allowed))
	for _, value := range allowed {
		if value == account.ProviderWeb || value == account.ProviderConsole {
			providers = append(providers, value)
		}
	}
	if len(providers) == 0 || limit <= 0 {
		return nil, nil
	}
	limit = min(limit, 1000)
	webModes := append([]string{"auto", "fast", "expert", "heavy", "weekly"}, account.WebImagineQuotaModes()...)
	// Match the existing application prediction: Console 24h, remote window
	// duration when provided, otherwise 5m. Do not synthesize upstream ResetAt.
	elapsedSeconds := "ROUND((julianday(?) - julianday(COALESCE(quota.synced_at, quota.updated_at))) * 86400, 3)"
	if r.db.db.Dialector.Name() == "postgres" {
		elapsedSeconds = "EXTRACT(EPOCH FROM (CAST(? AS TIMESTAMPTZ) - COALESCE(quota.synced_at, quota.updated_at)))"
	}
	const cursorTime = "COALESCE(quota.reset_at, quota.synced_at, quota.updated_at)"
	query := r.db.db.WithContext(ctx).Table("account_quota_windows AS quota").Select("quota.*").
		Joins("JOIN provider_accounts AS account ON account.id = quota.account_id").
		Where("account.provider IN ? AND account.auth_status = ?", providers, account.AuthStatusActive).
		Where("account.cooldown_until IS NULL OR account.cooldown_until <= ?", now).
		Where("quota.remaining = 0").
		Where("(quota.reset_at IS NOT NULL AND quota.reset_at <= ?) OR (quota.reset_at IS NULL AND "+elapsedSeconds+" >= CASE WHEN account.provider = ? THEN 86400 WHEN quota.window_seconds > 0 THEN quota.window_seconds ELSE 300 END)", now, now, account.ProviderConsole).
		Where("(account.provider = ? AND quota.mode IN ?) OR (account.provider = ? AND quota.mode IN ?)", account.ProviderWeb, webModes, account.ProviderConsole, []string{"console", "console_image", "console_video"})
	if !includeDisabled {
		query = query.Where("account.enabled = ?", true)
	}
	if after != nil {
		cursorAt := after.UpdatedAt
		if after.SyncedAt != nil {
			cursorAt = *after.SyncedAt
		}
		if after.ResetAt != nil {
			cursorAt = *after.ResetAt
		}
		query = query.Where(cursorTime+" > ? OR ("+cursorTime+" = ? AND quota.account_id > ?) OR ("+cursorTime+" = ? AND quota.account_id = ? AND quota.mode > ?)", cursorAt, cursorAt, after.AccountID, cursorAt, after.AccountID, after.Mode)
	}
	var rows []quotaWindowModel
	if err := query.Order(cursorTime + " ASC, quota.account_id ASC, quota.mode ASC").Limit(limit).Find(&rows).Error; err != nil {
		return nil, err
	}
	values := make([]account.QuotaWindow, 0, len(rows))
	for _, row := range rows {
		values = append(values, toQuotaWindowDomain(row))
	}
	return values, nil
}
