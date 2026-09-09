package account

import (
	"context"
	"errors"
	"strconv"
	"strings"
	"time"

	accountdomain "github.com/chenyme/grok2api/backend/internal/domain/account"
	"github.com/chenyme/grok2api/backend/internal/infra/provider"
	"github.com/chenyme/grok2api/backend/internal/repository"
)

// ListBuildReauthCandidates selects only existing Build/Web links whose SSO has
// not been rejected. Disabled Web accounts can supply existing credentials.
func (s *Service) ListBuildReauthCandidates(ctx context.Context, now time.Time, includeDisabled bool, limit int) ([]uint64, error) {
	repo, ok := s.accounts.(repository.ReauthRepository)
	if !ok {
		return nil, ErrUnsupported
	}
	return repo.ListBuildReauthCandidates(ctx, now, includeDisabled, limit)
}

// RecoverBuildAuthentication uses an existing linked SSO to reauthorize the same
// Build identity. It preserves business enablement and quota state, shares locks
// with manual conversion/OAuth refresh, and persists retry timing across restarts.
func (s *Service) RecoverBuildAuthentication(ctx context.Context, id uint64, includeDisabled bool, backoffBase, backoffMax time.Duration) (accountdomain.RecoveryResult, error) {
	skipped := func(reason string) (accountdomain.RecoveryResult, error) {
		return accountdomain.RecoveryResult{Skipped: true, Reason: reason}, nil
	}
	if err := ctx.Err(); err != nil {
		return accountdomain.RecoveryResult{Skipped: true, Reason: "canceled"}, err
	}
	value, err := s.accounts.Get(ctx, id)
	if err != nil {
		return accountdomain.RecoveryResult{}, err
	}
	if !reauthBuildEligible(value, s.now(), includeDisabled) {
		return skipped("ineligible")
	}
	if value.LinkedAccountID == 0 || value.LinkedProvider != accountdomain.ProviderWeb {
		return skipped("no_linked_sso")
	}
	if s.refreshLock == nil {
		return skipped("lock_unavailable")
	}
	release, acquired, err := s.refreshLock.Acquire(ctx, "web-build-conversion:"+strconv.FormatUint(value.LinkedAccountID, 10), 2*time.Minute)
	if err != nil {
		return accountdomain.RecoveryResult{}, err
	}
	if !acquired {
		return skipped("busy")
	}
	defer release()
	return s.recoverLinkedBuild(ctx, value.LinkedAccountID, id, true, includeDisabled, backoffBase, backoffMax)
}

// Caller owns the Web conversion lock; acquire the Build credential lock second
// everywhere to avoid lock-order inversion with manual conversion.
func (s *Service) recoverLinkedBuild(ctx context.Context, webID, buildID uint64, scheduled, includeDisabled bool, backoffBase, backoffMax time.Duration) (result accountdomain.RecoveryResult, err error) {
	skip := func(reason string) (accountdomain.RecoveryResult, error) {
		return accountdomain.RecoveryResult{Skipped: true, Reason: reason}, nil
	}
	repo, ok := s.accounts.(repository.ReauthRepository)
	if !ok {
		return result, ErrUnsupported
	}
	if s.refreshLock == nil {
		return skip("lock_unavailable")
	}
	release, acquired, err := s.refreshLock.Acquire(ctx, "credential-refresh:"+strconv.FormatUint(buildID, 10), 2*time.Minute)
	if err != nil {
		return result, err
	}
	if !acquired {
		return skip("busy")
	}
	defer release()
	build, err := s.accounts.Get(ctx, buildID)
	if err != nil {
		return result, err
	}
	web, err := s.accounts.Get(ctx, webID)
	if err != nil {
		return result, err
	}
	if build.Provider != accountdomain.ProviderBuild || build.AuthType != accountdomain.AuthTypeOAuth || build.LinkedAccountID != webID || web.LinkedAccountID != buildID {
		return skip("link_changed")
	}
	if web.Provider != accountdomain.ProviderWeb || web.AuthType != accountdomain.AuthTypeSSO || web.AuthStatus != accountdomain.AuthStatusActive || strings.TrimSpace(web.EncryptedAccessToken) == "" {
		return skip("sso_unavailable")
	}
	now := s.now()
	if scheduled && (!reauthBuildEligible(build, now, includeDisabled) || (web.CooldownUntil != nil && web.CooldownUntil.After(now))) {
		return skip("ineligible")
	}
	converter, ok := s.providers.BuildConverter(accountdomain.ProviderWeb)
	if !ok {
		return result, ErrUnsupported
	}
	leaseUntil := now.Add(2 * time.Minute)
	if scheduled {
		claimed, claimErr := repo.ClaimBuildReauth(ctx, build, web, now, leaseUntil, includeDisabled)
		if claimErr != nil {
			return result, claimErr
		}
		if !claimed {
			return skip("not_claimed")
		}
		defer func() {
			if err == nil {
				return
			}
			writeCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), credentialStateWriteTimeout)
			defer cancel()
			retryAt := s.now().Add(reauthBackoff(build.RefreshFailureCount, backoffBase, backoffMax))
			if saveErr := repo.FailBuildReauth(writeCtx, build, leaseUntil, retryAt, result.Reason); saveErr != nil {
				err = errors.Join(err, errors.New("reauth_backoff_write_failed"))
			}
		}()
	}
	fail := func(reason string) (accountdomain.RecoveryResult, error) {
		result.Reason = reason
		if reason == "canceled" && ctx.Err() != nil {
			return result, ctx.Err()
		}
		return result, errors.New(reason)
	}
	if !linkedReauthIdentityCompatible(build, web) {
		return fail("identity_conflict")
	}
	if ctx.Err() != nil {
		return fail("canceled")
	}
	result.Claimed = true
	attemptCtx, cancel := context.WithTimeout(ctx, 90*time.Second)
	seed, convertErr := converter.ConvertToBuild(attemptCtx, web)
	cancel()
	if convertErr != nil {
		if errors.Is(convertErr, provider.ErrUnauthorized) {
			writeCtx, cancelWrite := context.WithTimeout(context.WithoutCancel(ctx), credentialStateWriteTimeout)
			markErr := repo.RejectReauthSSO(writeCtx, web)
			cancelWrite()
			if markErr != nil {
				return fail("sso_rejection_write_failed")
			}
			return fail("sso_unauthorized")
		}
		if ctx.Err() != nil {
			return fail("canceled")
		}
		return fail("sso_transient")
	}
	if !reauthSeedMatches(build, web, seed) {
		return fail("identity_conflict")
	}
	if strings.TrimSpace(seed.AccessToken) == "" || strings.TrimSpace(seed.RefreshToken) == "" || strings.TrimSpace(seed.OIDCClientID) == "" {
		return fail("invalid_credentials")
	}
	access, encryptErr := s.cipher.Encrypt(seed.AccessToken)
	if encryptErr != nil {
		return fail("credential_encryption_failed")
	}
	refresh, encryptErr := s.cipher.Encrypt(seed.RefreshToken)
	if encryptErr != nil {
		return fail("credential_encryption_failed")
	}
	metadataCredential := build
	metadataCredential.EncryptedAccessToken = access
	botFlag := build.BuildBotFlagSource
	if metadata := s.credentialMetadata(metadataCredential); metadata.BuildBotFlagInspected {
		botFlag = metadata.BuildBotFlagSource
	}
	writeCtx, cancelWrite := context.WithTimeout(context.WithoutCancel(ctx), credentialStateWriteTimeout)
	saved, saveErr := repo.SaveBuildReauth(writeCtx, build, web, access, refresh, seed.OIDCClientID, seed.ExpiresAt, botFlag)
	cancelWrite()
	if saveErr != nil {
		return fail("credential_write_failed")
	}
	if !saved {
		return fail("credential_changed")
	}
	s.invalidateBuildBotFlagCache()
	s.markRefreshSuccess(buildID, s.now())
	s.WakeCredentialRefresh()
	result.Recovered = true
	result.Reason = "recovered"
	return result, nil
}

func reauthBuildEligible(value accountdomain.Credential, now time.Time, includeDisabled bool) bool {
	return value.Provider == accountdomain.ProviderBuild && value.AuthType == accountdomain.AuthTypeOAuth && value.AuthStatus == accountdomain.AuthStatusReauthRequired && (includeDisabled || value.Enabled) && (value.RefreshDueAt == nil || !value.RefreshDueAt.After(now)) && (value.CooldownUntil == nil || !value.CooldownUntil.After(now))
}

func linkedReauthIdentityCompatible(build, web accountdomain.Credential) bool {
	if strings.TrimSpace(build.UserID) != "" && strings.TrimSpace(web.UserID) != "" && strings.TrimSpace(build.UserID) != strings.TrimSpace(web.UserID) {
		return false
	}
	b, w := strings.ToLower(strings.TrimSpace(build.Email)), strings.ToLower(strings.TrimSpace(web.Email))
	return b == "" || w == "" || b == w
}

func reauthSeedMatches(build, web accountdomain.Credential, seed provider.CredentialSeed) bool {
	userID := strings.TrimSpace(seed.UserID)
	email := strings.ToLower(strings.TrimSpace(seed.Email))
	if buildUserID := strings.TrimSpace(build.UserID); buildUserID != "" {
		if userID != buildUserID {
			return false
		}
	} else if buildEmail := strings.ToLower(strings.TrimSpace(build.Email)); buildEmail == "" || email != buildEmail {
		return false
	}
	for _, value := range []accountdomain.Credential{build, web} {
		if id := strings.TrimSpace(value.UserID); id != "" && id != userID {
			return false
		}
		if previous := strings.ToLower(strings.TrimSpace(value.Email)); previous != "" && email != "" && previous != email {
			return false
		}
	}
	return true
}

func reauthBackoff(failures int, base, maxDelay time.Duration) time.Duration {
	if base <= 0 {
		base = time.Hour
	}
	if maxDelay < base {
		maxDelay = base
	}
	delay := base
	for i := 0; i < failures && delay < maxDelay; i++ {
		if delay > maxDelay/2 {
			return maxDelay
		}
		delay *= 2
	}
	return min(delay, maxDelay)
}
