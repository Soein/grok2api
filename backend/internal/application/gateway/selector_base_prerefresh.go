package gateway

import (
	"context"
	"errors"
	"time"

	"github.com/chenyme/grok2api/backend/internal/domain/account"
	"github.com/chenyme/grok2api/backend/internal/pkg/perfmetrics"
	"github.com/chenyme/grok2api/backend/internal/repository"
)

const (
	defaultBuildBasePreRefreshAhead   = 5 * time.Second
	defaultBuildBasePreRefreshTimeout = 5 * time.Second
	buildBasePreRefreshTick           = time.Second
)

var errUnexpectedQuotaWindow = errors.New("unexpected quota window in build base prerefresh")

// UpdateBuildBasePreRefreshPolicy updates the Build base pre-refresh configuration.
func (s *Selector) UpdateBuildBasePreRefreshPolicy(enabled bool, ahead, timeout time.Duration) {
	s.configMu.Lock()
	defer s.configMu.Unlock()
	changed := false
	if s.buildBasePreRefreshEnabled != enabled {
		s.buildBasePreRefreshEnabled = enabled
		changed = true
	}
	if ahead > 0 && s.buildBasePreRefreshAhead != ahead {
		s.buildBasePreRefreshAhead = ahead
		changed = true
	}
	if timeout > 0 && s.buildBasePreRefreshTimeout != timeout {
		s.buildBasePreRefreshTimeout = timeout
		changed = true
	}
	if changed {
		s.buildBasePreRefreshGeneration++
	}
}

// BuildBasePreRefreshPolicy returns the current Build base pre-refresh policy.
func (s *Selector) BuildBasePreRefreshPolicy() (bool, time.Duration, time.Duration) {
	s.configMu.RLock()
	defer s.configMu.RUnlock()
	return s.buildBasePreRefreshEnabled, s.buildBasePreRefreshAhead, s.buildBasePreRefreshTimeout
}

// RunBasePreRefresh executes the serial background worker until context cancellation.
func (s *Selector) RunBasePreRefresh(ctx context.Context) error {
	ticker := time.NewTicker(buildBasePreRefreshTick)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			s.preRefreshBuildBaseAt(ctx, time.Now().UTC())
		}
	}
}

// preRefreshBuildBaseAt checks and conditionally refreshes the Build base snapshot.
// Returns true if a refresh was attempted and successfully published.
func (s *Selector) preRefreshBuildBaseAt(ctx context.Context, now time.Time) bool {
	if !s.buildBasePreRefreshRunning.CompareAndSwap(false, true) {
		return false
	}
	defer s.buildBasePreRefreshRunning.Store(false)

	s.configMu.RLock()
	enabled := s.buildBasePreRefreshEnabled
	ahead := s.buildBasePreRefreshAhead
	timeout := s.buildBasePreRefreshTimeout
	policyGen := s.buildBasePreRefreshGeneration
	s.configMu.RUnlock()

	if !enabled || ahead <= 0 || timeout <= 0 {
		return false
	}

	layered, ok := s.accounts.(repository.RoutingLayerRepository)
	if !ok {
		return false
	}

	key := routingBaseCacheKey{provider: account.ProviderBuild, quotaMode: ""}
	var targetGeneration uint64
	var targetVersion routingLayerVersion

	s.candidateMu.Lock()
	if !s.buildBasePreRefreshRetryUntil.IsZero() && now.Before(s.buildBasePreRefreshRetryUntil) {
		s.candidateMu.Unlock()
		return false
	}

	baseSnap, ok := s.routingBases[key]
	if !ok {
		s.candidateMu.Unlock()
		return false
	}

	currentVersion := s.routingBaseVersionLocked(account.ProviderBuild)
	if baseSnap.version != currentVersion {
		s.candidateMu.Unlock()
		return false
	}

	if !now.Before(baseSnap.expiresAt) {
		s.candidateMu.Unlock()
		return false
	}

	if baseSnap.expiresAt.Sub(now) > ahead {
		s.candidateMu.Unlock()
		return false
	}

	hasRecentCandidate := false
	for cKey, cSnap := range s.candidates {
		if cKey.provider == account.ProviderBuild && cKey.quotaMode == "" {
			if !cSnap.lastAccess.IsZero() && now.Sub(cSnap.lastAccess) <= candidateCacheTTL && (cSnap.lastAccess.Before(now) || cSnap.lastAccess.Equal(now)) {
				hasRecentCandidate = true
				break
			}
		}
	}
	if !hasRecentCandidate {
		s.candidateMu.Unlock()
		return false
	}

	targetGeneration = baseSnap.generation
	targetVersion = currentVersion
	s.candidateMu.Unlock()

	attemptLabels := perfmetrics.Labels{
		Subsystem: "gateway",
		Operation: "base_prerefresh",
		Provider:  string(account.ProviderBuild),
		Outcome:   "attempt",
	}
	perfmetrics.Default.Inc("selector_base_prerefresh_total", attemptLabels)

	queryCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	queryStart := time.Now().UTC()
	values, err := layered.ListRoutingAccountBases(queryCtx, account.ProviderBuild, "")
	duration := time.Since(queryStart)

	if err == nil && queryCtx.Err() != nil {
		err = queryCtx.Err()
	}

	if err == nil {
		for _, val := range values {
			if val.QuotaWindow != nil {
				err = errUnexpectedQuotaWindow
				break
			}
		}
	}

	if err != nil {
		s.candidateMu.Lock()
		s.buildBasePreRefreshRetryUntil = time.Now().UTC().Add(candidateCacheRetryTTL)
		s.candidateMu.Unlock()

		outcome := "failure"
		if errors.Is(err, context.DeadlineExceeded) || queryCtx.Err() == context.DeadlineExceeded {
			outcome = "timeout"
		} else if errors.Is(err, context.Canceled) || queryCtx.Err() == context.Canceled {
			outcome = "canceled"
		}
		labels := perfmetrics.Labels{
			Subsystem: "gateway",
			Operation: "base_prerefresh",
			Provider:  string(account.ProviderBuild),
			Outcome:   outcome,
		}
		perfmetrics.Default.Inc("selector_base_prerefresh_total", labels)
		perfmetrics.Default.ObserveDuration("selector_base_prerefresh_duration_us", labels, duration)
		return false
	}

	s.configMu.RLock()
	defer s.configMu.RUnlock()

	s.candidateMu.Lock()
	defer s.candidateMu.Unlock()

	if queryCtx.Err() != nil || ctx.Err() != nil {
		s.recordDiscard(duration)
		return false
	}

	if !s.buildBasePreRefreshEnabled || s.buildBasePreRefreshGeneration != policyGen {
		s.recordDiscard(duration)
		return false
	}

	currentSnap, exists := s.routingBases[key]
	if !exists || currentSnap.generation != targetGeneration || currentSnap.version != targetVersion || currentSnap.version != s.routingBaseVersionLocked(account.ProviderBuild) {
		s.recordDiscard(duration)
		return false
	}

	newExpiresAt := queryStart.Add(candidateCacheTTL)
	newSnapshot := routingBaseSnapshot{
		values:     values,
		version:    targetVersion,
		expiresAt:  newExpiresAt,
		staleUntil: newExpiresAt.Add(candidateCacheStaleTTL),
	}
	s.storeRoutingBaseSnapshotLockedWithAccess(key, newSnapshot, currentSnap.lastAccess, queryStart)

	for accountID, cachedProvider := range s.routingAccountProvider {
		if cachedProvider == account.ProviderBuild {
			delete(s.routingAccountProvider, accountID)
		}
	}
	for _, value := range values {
		s.routingAccountProvider[value.Credential.ID] = account.ProviderBuild
	}

	s.buildBasePreRefreshRetryUntil = time.Time{}

	successLabels := perfmetrics.Labels{
		Subsystem: "gateway",
		Operation: "base_prerefresh",
		Provider:  string(account.ProviderBuild),
		Outcome:   "success",
	}
	perfmetrics.Default.Inc("selector_base_prerefresh_total", successLabels)
	perfmetrics.Default.ObserveDuration("selector_base_prerefresh_duration_us", successLabels, duration)
	perfmetrics.Default.Add("selector_base_prerefresh_rows", successLabels, int64(len(values)))
	return true
}

func (s *Selector) recordDiscard(duration time.Duration) {
	labels := perfmetrics.Labels{
		Subsystem: "gateway",
		Operation: "base_prerefresh",
		Provider:  string(account.ProviderBuild),
		Outcome:   "discard",
	}
	perfmetrics.Default.Inc("selector_base_prerefresh_total", labels)
	perfmetrics.Default.ObserveDuration("selector_base_prerefresh_duration_us", labels, duration)
}
