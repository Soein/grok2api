package account

import (
	"context"
	"errors"
	"slices"
	"strconv"
	"strings"
	"time"

	accountdomain "github.com/chenyme/grok2api/backend/internal/domain/account"
	"github.com/chenyme/grok2api/backend/internal/repository"
)

// QuotaRecoverySkipError distinguishes ineligible or temporarily busy candidates
// from provider failures without exposing credentials or upstream error text.
type QuotaRecoverySkipError struct {
	Reason    string
	Permanent bool
	RetryAt   time.Time
}

func (e *QuotaRecoverySkipError) Error() string                        { return "quota recovery skipped: " + e.Reason }
func (e *QuotaRecoverySkipError) QuotaRecoverySkip() (bool, time.Time) { return e.Permanent, e.RetryAt }

func quotaRecoveryCredentialPolicy(value accountdomain.Credential, mode string, now time.Time, allowed []accountdomain.Provider, includeDisabled bool) error {
	validMode := (value.Provider == accountdomain.ProviderConsole && isConsoleUsageQuotaMode(mode)) ||
		(value.Provider == accountdomain.ProviderWeb && (isWebChatQuotaMode(mode) || mode == "weekly" || isWebImagineQuotaMode(mode)))
	if !validMode || !slices.Contains(allowed, value.Provider) {
		return &QuotaRecoverySkipError{Reason: "provider_or_mode", Permanent: true}
	}
	if value.AuthStatus != accountdomain.AuthStatusActive {
		return &QuotaRecoverySkipError{Reason: "authentication", Permanent: true}
	}
	if !value.Enabled && !includeDisabled {
		return &QuotaRecoverySkipError{Reason: "disabled", Permanent: true}
	}
	if value.CooldownUntil != nil && value.CooldownUntil.After(now) {
		return &QuotaRecoverySkipError{Reason: "cooldown", RetryAt: *value.CooldownUntil}
	}
	return nil
}

func quotaRecoveryRefreshLockKey(id uint64, mode string) string {
	if isConsoleUsageQuotaMode(mode) {
		return consoleQuotaRefreshLockKey(id)
	}
	if isWebImagineQuotaMode(mode) {
		mode = accountdomain.QuotaGroupWebImagine
	}
	return "quota-refresh:" + strconv.FormatUint(id, 10) + ":" + mode
}

// ProbeQuotaModeForRecovery checks the latest account policy and quota while
// holding the same refresh lease as request-triggered refreshes. It never
// changes Enabled and does not create a second recovery queue event.
func (s *Service) ProbeQuotaModeForRecovery(ctx context.Context, id uint64, mode string, now time.Time, allowed []accountdomain.Provider, includeDisabled bool) (accountdomain.QuotaWindow, error) {
	return s.probeQuotaModeWithPolicy(ctx, id, mode, now, allowed, includeDisabled, true)
}

func (s *Service) quotaRecoveryCandidate(ctx context.Context, id uint64, mode string, now time.Time, allowed []accountdomain.Provider, includeDisabled, requireDue bool) (accountdomain.Credential, []accountdomain.QuotaWindow, error) {
	value, err := s.accounts.Get(ctx, id)
	if errors.Is(err, repository.ErrNotFound) {
		return value, nil, &QuotaRecoverySkipError{Reason: "missing", Permanent: true}
	}
	if err != nil {
		return value, nil, err
	}
	if err := quotaRecoveryCredentialPolicy(value, mode, now, allowed, includeDisabled); err != nil {
		return value, nil, err
	}
	byID, err := s.accounts.GetQuotaWindows(ctx, []uint64{id})
	if err != nil {
		return value, nil, err
	}
	windows := byID[id]
	if requireDue {
		window, exists := quotaWindowByMode(windows, mode)
		if !exists || window.Remaining != 0 {
			return value, windows, &QuotaRecoverySkipError{Reason: "not_exhausted", Permanent: true}
		}
		var due time.Time
		if window.ResetAt != nil {
			due = *window.ResetAt
		} else {
			due = window.UpdatedAt
			if window.SyncedAt != nil {
				due = *window.SyncedAt
			}
			delay := unknownRemoteQuotaProbeDelay
			if isConsoleUsageQuotaMode(mode) {
				delay = consolePredictedQuotaProbeDelay
			} else if window.WindowSeconds > 0 {
				delay = time.Duration(window.WindowSeconds) * time.Second
			}
			due = due.Add(delay)
		}
		if due.After(now) {
			return value, windows, &QuotaRecoverySkipError{Reason: "not_due", RetryAt: due}
		}
	}
	return value, windows, nil
}

func (s *Service) probeQuotaModeWithPolicy(ctx context.Context, id uint64, mode string, now time.Time, allowed []accountdomain.Provider, includeDisabled, requireDue bool) (accountdomain.QuotaWindow, error) {
	if err := ctx.Err(); err != nil {
		return accountdomain.QuotaWindow{}, err
	}
	mode = strings.TrimSpace(mode)
	value, windows, err := s.quotaRecoveryCandidate(ctx, id, mode, now, allowed, includeDisabled, requireDue)
	if err != nil {
		return accountdomain.QuotaWindow{}, err
	}
	lockMode := mode
	if value.Provider == accountdomain.ProviderWeb && !isWebImagineQuotaMode(mode) {
		if _, exists := quotaWindowByMode(windows, "weekly"); exists {
			lockMode = "weekly"
		}
	}
	if s.refreshLock == nil && requireDue {
		return accountdomain.QuotaWindow{}, &QuotaRecoverySkipError{Reason: "lock_unavailable", RetryAt: now.Add(time.Minute)}
	}
	if s.refreshLock != nil {
		release, acquired, err := s.refreshLock.Acquire(ctx, quotaRecoveryRefreshLockKey(id, lockMode), 2*quotaRefreshTimeout)
		if err != nil {
			return accountdomain.QuotaWindow{}, err
		}
		if !acquired {
			return accountdomain.QuotaWindow{}, &QuotaRecoverySkipError{Reason: "busy", RetryAt: now.Add(time.Minute)}
		}
		defer release()
		if value.Provider == accountdomain.ProviderWeb && isWebImagineQuotaMode(mode) && (value.WebTier == accountdomain.WebTierSuper || value.WebTier == accountdomain.WebTierHeavy) {
			// Paid Imagine may resolve to the shared weekly pool after the group snapshot.
			releaseWeekly, acquired, err := s.refreshLock.Acquire(ctx, quotaRecoveryRefreshLockKey(id, "weekly"), 2*quotaRefreshTimeout)
			if err != nil {
				return accountdomain.QuotaWindow{}, err
			}
			if !acquired {
				return accountdomain.QuotaWindow{}, &QuotaRecoverySkipError{Reason: "busy", RetryAt: now.Add(time.Minute)}
			}
			defer releaseWeekly()
		}
		// Policy may change between queueing and lease acquisition.
		if _, _, err := s.quotaRecoveryCandidate(ctx, id, mode, now, allowed, includeDisabled, requireDue); err != nil {
			return accountdomain.QuotaWindow{}, err
		}
	}
	return s.probeQuotaMode(ctx, id, mode)
}

type dueQuotaRecoveryRepository interface {
	ListDueQuotaWindowsForRecovery(context.Context, time.Time, int, []accountdomain.Provider, bool) ([]accountdomain.QuotaWindow, error)
}

type dueQuotaRecoveryCursorRepository interface {
	ListDueQuotaWindowsForRecoveryAfter(context.Context, time.Time, int, *accountdomain.QuotaWindow, []accountdomain.Provider, bool) ([]accountdomain.QuotaWindow, error)
}

// ListDueQuotaWindowsForRecoveryAfter provides stable pagination for background
// reconciliation even when earlier windows retain future queue backoff.
func (s *Service) ListDueQuotaWindowsForRecoveryAfter(ctx context.Context, now time.Time, limit int, after *accountdomain.QuotaWindow, allowed []accountdomain.Provider, includeDisabled bool) ([]accountdomain.QuotaWindow, error) {
	if repo, ok := s.accounts.(dueQuotaRecoveryCursorRepository); ok {
		return repo.ListDueQuotaWindowsForRecoveryAfter(ctx, now, limit, after, allowed, includeDisabled)
	}
	return s.ListDueQuotaWindowsForRecovery(ctx, now, limit, allowed, includeDisabled)
}

// ListDueQuotaWindowsForRecovery bounds queue reconciliation to eligible providers
// and account states; claims still recheck policy immediately before probing.
func (s *Service) ListDueQuotaWindowsForRecovery(ctx context.Context, now time.Time, limit int, allowed []accountdomain.Provider, includeDisabled bool) ([]accountdomain.QuotaWindow, error) {
	if repo, ok := s.accounts.(dueQuotaRecoveryRepository); ok {
		return repo.ListDueQuotaWindowsForRecovery(ctx, now, limit, allowed, includeDisabled)
	}
	windows, err := s.accounts.ListDueQuotaWindows(ctx, now, limit)
	if err != nil {
		return nil, err
	}
	eligible := make([]accountdomain.QuotaWindow, 0, len(windows))
	for _, window := range windows {
		_, _, err := s.quotaRecoveryCandidate(ctx, window.AccountID, window.Mode, now, allowed, includeDisabled, true)
		var skip *QuotaRecoverySkipError
		if errors.As(err, &skip) {
			continue
		}
		if err != nil {
			return nil, err
		}
		eligible = append(eligible, window)
	}
	return eligible, nil
}
