package gateway

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/chenyme/grok2api/backend/internal/domain/account"
	"github.com/chenyme/grok2api/backend/internal/domain/clientkey"
	modeldomain "github.com/chenyme/grok2api/backend/internal/domain/model"
	"github.com/chenyme/grok2api/backend/internal/repository"
)

type quotaRecoveryClaimContextKey struct{}

type quotaRecoveryClaimWriter interface {
	SaveClaimedQuotaRecovery(context.Context, time.Time, account.QuotaRecovery) (bool, error)
}

// saveQuotaRecovery preserves the lease identity for failures discovered by a
// probe while retaining the ordinary request path for fresh exhaustion reports.
func (s *Selector) saveQuotaRecovery(ctx context.Context, value account.QuotaRecovery) error {
	if until, ok := ctx.Value(quotaRecoveryClaimContextKey{}).(time.Time); ok {
		writer, supported := s.accounts.(quotaRecoveryClaimWriter)
		if !supported {
			return errors.New("quota recovery claim writer unavailable")
		}
		_, err := writer.SaveClaimedQuotaRecovery(ctx, until, value)
		return err
	}
	return s.accounts.SaveQuotaRecovery(ctx, value)
}

type quotaRecoveryRepository interface {
	GetQuotaRecoveryCandidate(context.Context, uint64, uint64, string, string) (account.RoutingCandidate, error)
	CompleteQuotaProbe(context.Context, uint64, time.Time, bool, time.Time) (bool, error)
}

// SetQuotaRecoveryIdentity installs the non-exportable server identity before workers start.
func (s *Service) SetQuotaRecoveryIdentity(key clientkey.Key) { s.quotaRecoveryIdentity = key }

func quotaRecoveryCandidateReason(c account.RoutingCandidate, now time.Time, includeDisabled, checkModel bool) string {
	if c.Credential.Provider != account.ProviderBuild {
		return "provider"
	}
	if !includeDisabled && !c.Credential.Enabled {
		return "disabled"
	}
	if c.Credential.AuthStatus != account.AuthStatusActive {
		return "authentication"
	}
	if c.Credential.CooldownUntil != nil && now.Before(*c.Credential.CooldownUntil) {
		return "cooldown"
	}
	if candidateEgressLeaseCooling(c, c.Credential, now) {
		return "egress_cooldown"
	}
	r := c.QuotaRecovery
	if r == nil || (r.Status != account.QuotaRecoveryStatusExhausted && r.Status != account.QuotaRecoveryStatusProbing) || r.NextProbeAt == nil || now.Before(*r.NextProbeAt) {
		return "not_due"
	}
	if checkModel {
		if c.ModelCapabilityKnown && !c.SupportsModel {
			return "model_unavailable"
		}
		if c.ModelQuotaBlock != nil && now.Before(c.ModelQuotaBlock.CooldownUntil) {
			return "model_cooldown"
		}
	}
	return ""
}

func (s *Service) quotaRecoveryEligibility(ctx context.Context, candidate account.RoutingCandidate, includeDisabled, checkModel bool) string {
	if reason := quotaRecoveryCandidateReason(candidate, time.Now().UTC(), includeDisabled, checkModel); reason != "" {
		return reason
	}
	filtered, err := s.selector.applyBuildBotFlaggedFilter(ctx, account.ProviderBuild, []account.RoutingCandidate{candidate})
	if err != nil || len(filtered) == 0 {
		return "scheduling_excluded"
	}
	return ""
}

// ProbeBuildQuotaRecovery probes one due Build account using its ordinary quota
// claim, concurrency limit, credential refresh lock and request failure policy.
// includeDisabled permits maintenance only; it never enables business routing.
func (s *Service) ProbeBuildQuotaRecovery(ctx context.Context, accountID uint64, models []string, includeDisabled bool) (out account.RecoveryResult, err error) {
	// All upstream work ends before the shared five-minute claim can expire.
	ctx, cancel := context.WithTimeout(ctx, 2*time.Minute)
	defer cancel()
	repo, ok := s.selector.accounts.(quotaRecoveryRepository)
	if !ok {
		return out, errors.New("quota recovery repository unavailable")
	}
	candidate, err := repo.GetQuotaRecoveryCandidate(ctx, accountID, 0, "", "")
	if err != nil {
		return out, err
	}
	if reason := s.quotaRecoveryEligibility(ctx, candidate, includeDisabled, false); reason != "" {
		return account.RecoveryResult{Skipped: true, Reason: reason}, nil
	}
	paid := candidate.QuotaRecovery.Kind == account.QuotaRecoveryKindPaid
	var route modeldomain.Route
	if !paid {
		if !s.quotaRecoveryIdentity.IsAvailable(time.Now().UTC()) || s.quotaRecoveryIdentity.InternalKind != clientkey.InternalKindAccountRecovery {
			return out, errors.New("quota recovery identity unavailable")
		}
		found := false
		for _, name := range models {
			public, valid := qualityProbeBuildPublicModel(name)
			if !valid {
				continue
			}
			routes, _, resolveErr := s.resolvePublicModelRoutes(ctx, public, false)
			if resolveErr != nil {
				routes, resolveErr = s.quotaRecoveryConfiguredRoutes(ctx, public)
				if resolveErr != nil {
					continue
				}
			}
			for _, r := range routes {
				if !r.Enabled || r.Provider != account.ProviderBuild || (r.Capability != modeldomain.CapabilityChat && r.Capability != modeldomain.CapabilityResponses) {
					continue
				}
				c, readErr := repo.GetQuotaRecoveryCandidate(ctx, accountID, r.ID, r.UpstreamModel, "")
				if errors.Is(readErr, repository.ErrNotFound) {
					continue
				}
				if readErr != nil {
					return out, readErr
				}
				c.Credential = s.selector.applyRoutingHealth(c.Credential, time.Now().UTC())
				if s.quotaRecoveryEligibility(ctx, c, includeDisabled, true) != "" {
					continue
				}
				if _, limited := s.activeTeamModelRateLimit(c.Credential, r.UpstreamModel, time.Now().UTC()); limited {
					continue
				}
				candidate, route, found = c, r, true
				break
			}
			if found {
				break
			}
		}
		if !found {
			return account.RecoveryResult{Skipped: true, Reason: "no_eligible_model"}, nil
		}
	}
	limit := candidate.Credential.MaxConcurrent
	if limit <= 0 {
		limit = account.DefaultMaxConcurrent
	}
	release, acquired, err := s.selector.concurrency.Acquire(ctx, accountConcurrencyKey(accountID), limit)
	if err != nil {
		return out, err
	}
	if !acquired {
		return account.RecoveryResult{Skipped: true, Reason: "busy"}, nil
	}
	lease := &accountLease{release: func() { release(); s.selector.announceLeaseReturn() }}
	defer lease.Release()
	// Recheck authoritative eligibility after waiting for shared account capacity.
	candidate, err = repo.GetQuotaRecoveryCandidate(ctx, accountID, route.ID, route.UpstreamModel, "")
	if err != nil {
		return out, err
	}
	candidate.Credential = s.selector.applyRoutingHealth(candidate.Credential, time.Now().UTC())
	if reason := s.quotaRecoveryEligibility(ctx, candidate, includeDisabled, !paid); reason != "" {
		return account.RecoveryResult{Skipped: true, Reason: reason}, nil
	}
	if !paid {
		latestRoute, routeErr := s.models.Get(ctx, route.ID)
		if routeErr != nil {
			return out, routeErr
		}
		if !latestRoute.Enabled || latestRoute.Provider != route.Provider || latestRoute.UpstreamModel != route.UpstreamModel {
			return account.RecoveryResult{Skipped: true, Reason: "route_changed"}, nil
		}
	}
	now := time.Now().UTC()
	until := now.Add(quotaProbeLease)
	claimed, err := s.selector.accounts.ClaimQuotaProbe(ctx, accountID, now, until)
	if err != nil {
		return out, err
	}
	if !claimed {
		return account.RecoveryResult{Skipped: true, Reason: "claimed"}, nil
	}
	out.Claimed = true
	lease.Credential = candidate.Credential
	lease.Billing = candidate.Billing
	lease.QuotaProbe = true
	lease.QuotaProbeKind = candidate.QuotaRecovery.Kind
	lease.quotaProbeUntil = until
	if paid {
		out.Recovered, err = s.accounts.ProbePaidQuotaClaimed(ctx, lease.Credential, until)
		s.selector.MarkQuotaStateChanged(account.ProviderBuild, accountID)
		if out.Recovered {
			out.Reason = "recovered"
		} else {
			out.Reason = "quota_pending"
		}
		return out, err
	}
	body, _ := json.Marshal(map[string]any{"model": route.UpstreamModel, "messages": []map[string]string{{"role": "user", "content": "Reply OK."}}, "stream": true, "stream_options": map[string]bool{"include_usage": true}, "max_tokens": 16})
	result, err := s.CreateChatCompletion(ctx, Input{ClientKey: s.quotaRecoveryIdentity, PublicModel: route.PublicID, RequestID: newAuditEventID(), Body: body, Streaming: true, skipQualityHold: true, quotaRecoveryLease: lease, quotaRecoveryRoute: &route})
	if err != nil {
		out.Reason = "probe_failed"
		return out, err
	}
	if err = consumeQuotaRecoveryStream(ctx, result); err != nil {
		out.Reason = "probe_failed"
		return out, err
	}
	recovery, readErr := s.selector.accounts.GetQuotaRecovery(ctx, accountID)
	out.Recovered = errors.Is(readErr, repository.ErrNotFound) || (readErr == nil && recovery.Status == account.QuotaRecoveryStatusActive)
	if readErr != nil && !errors.Is(readErr, repository.ErrNotFound) {
		return out, readErr
	}
	if out.Recovered {
		out.Reason = "recovered"
	} else {
		out.Reason = "claim_changed"
	}
	return out, nil
}

// Configured routes remain eligible for maintenance even when every account
// behind them is disabled for business traffic. Only the local catalog is read.
func (s *Service) quotaRecoveryConfiguredRoutes(ctx context.Context, public string) ([]modeldomain.Route, error) {
	catalog, ok := s.models.(interface {
		ListConfiguredEnabled(context.Context) ([]modeldomain.Route, error)
	})
	if !ok {
		return nil, repository.ErrNotFound
	}
	routes, err := catalog.ListConfiguredEnabled(ctx)
	if err != nil {
		return nil, err
	}
	groups := modeldomain.PublicIDCandidateGroups(public)
	alias, hasAlias := s.providers.ResolveModelAlias(public)
	var matched []modeldomain.Route
	for _, route := range routes {
		if route.Provider != account.ProviderBuild || !route.Enabled {
			continue
		}
		routePublic, _ := modeldomain.NormalizePublicID(route.Provider, route.PublicID)
		requested, _ := modeldomain.NormalizePublicID(route.Provider, public)
		match := routePublic == requested
		for _, group := range groups {
			for _, name := range group {
				if route.PublicID == name {
					match = true
				}
			}
		}
		if hasAlias && alias.Provider == route.Provider && alias.UpstreamModel == route.UpstreamModel {
			match = true
		}
		if match {
			matched = append(matched, route)
		}
	}
	return matched, nil
}

func consumeQuotaRecoveryStream(ctx context.Context, result *Result) (err error) {
	usage := Usage{}
	responseID := ""
	code := "upstream_stream_incomplete"
	defer func() { result.Finalize(usage, responseID, code); _ = result.Body.Close() }()
	if result.StatusCode < http.StatusOK || result.StatusCode >= http.StatusMultipleChoices {
		code = "upstream_stream_error"
		return errors.New("quota probe upstream rejected")
	}
	scanner := bufio.NewScanner(result.Body)
	scanner.Buffer(make([]byte, 64<<10), 1<<20)
	total := 0
	terminal := false
	for scanner.Scan() {
		if ctx.Err() != nil {
			code = "request_canceled"
			return ctx.Err()
		}
		line := strings.TrimSpace(scanner.Text())
		total += len(line) + 1
		if total > qualityProbeMaxStreamBytes {
			return errors.New("quota probe stream too large")
		}
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "[DONE]" {
			terminal = true
			break
		}
		var raw map[string]json.RawMessage
		if json.Unmarshal([]byte(payload), &raw) != nil {
			return errors.New("quota probe malformed stream")
		}
		if v, ok := raw["error"]; ok && string(v) != "null" {
			code = "upstream_stream_error"
			return errors.New("quota probe stream error")
		}
		var event qualityProbeChatEvent
		if json.Unmarshal([]byte(payload), &event) != nil {
			return errors.New("quota probe malformed stream")
		}
		if responseID == "" {
			responseID = event.ID
		}
		if event.Usage != nil {
			usage.Reported = true
			usage.InputTokens = event.Usage.PromptTokens
			usage.OutputTokens = event.Usage.CompletionTokens
			usage.TotalTokens = event.Usage.TotalTokens
			usage.ReasoningTokens = event.Usage.CompletionTokensDetails.ReasoningTokens
			usage.ResponseModel = event.Model
		}
		for _, c := range event.Choices {
			if strings.TrimSpace(c.Delta.Content) != "" {
				usage.OutputObserved = true
				if result.MarkFirstToken != nil {
					result.MarkFirstToken()
				}
			}
		}
	}
	if ctx.Err() != nil {
		code = "request_canceled"
		return ctx.Err()
	}
	if scanner.Err() != nil {
		code = "upstream_stream_interrupted"
		return fmt.Errorf("quota probe stream interrupted: %w", scanner.Err())
	}
	if !terminal {
		return errors.New("quota probe stream incomplete")
	}
	if !usage.OutputObserved {
		code = "upstream_response_empty"
		return errors.New("quota probe empty output")
	}
	code = ""
	return nil
}
