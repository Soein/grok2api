package gateway

import (
	"context"
	"errors"
	"fmt"
	accountapp "github.com/chenyme/grok2api/backend/internal/application/account"
	clientkeyapp "github.com/chenyme/grok2api/backend/internal/application/clientkey"
	"github.com/chenyme/grok2api/backend/internal/domain/clientkey"
	"github.com/chenyme/grok2api/backend/internal/infra/persistence/relational"
	"github.com/chenyme/grok2api/backend/internal/infra/provider"
	"github.com/chenyme/grok2api/backend/internal/infra/runtime/memory"
	"github.com/chenyme/grok2api/backend/internal/repository"
	"io"
	"log/slog"
	"net/http"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/chenyme/grok2api/backend/internal/domain/account"
)

func TestQuotaRecoveryStreamRequiresTerminalAndOutput(t *testing.T) {
	for _, tc := range []struct {
		name, body string
		valid      bool
	}{
		{"complete", "data: {\"choices\":[{\"delta\":{\"content\":\"OK\"}}]}\n\ndata: [DONE]\n\n", true},
		{"empty", "data: [DONE]\n\n", false},
		{"truncated", "data: {\"choices\":[{\"delta\":{\"content\":\"OK\"}}]}\n\n", false},
		{"error", "data: {\"error\":{\"message\":\"failed\"}}\n\ndata: [DONE]\n\n", false},
		{"malformed", "data: {bad\n\ndata: [DONE]\n\n", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			finalized, closed := false, false
			result := &Result{StatusCode: http.StatusOK, Body: &quotaTestBody{Reader: strings.NewReader(tc.body), close: func() {
				if !finalized {
					t.Error("closed before finalize")
				}
				closed = true
			}}, Finalize: func(u Usage, id, code string) {
				finalized = true
				if tc.valid && code != "" {
					t.Errorf("code = %s", code)
				}
				if !tc.valid && code == "" {
					t.Error("failure finalized successfully")
				}
			}}
			err := consumeQuotaRecoveryStream(context.Background(), result)
			if (err == nil) != tc.valid {
				t.Fatalf("err = %v", err)
			}
			if !closed || !finalized {
				t.Fatal("stream not finalized and closed")
			}
		})
	}
}

type quotaTestBody struct {
	io.Reader
	close func()
}

func (b *quotaTestBody) Close() error { b.close(); return nil }

func TestQuotaRecoveryCandidateEligibility(t *testing.T) {
	now := time.Now().UTC()
	due := now.Add(-time.Minute)
	future := now.Add(time.Hour)
	base := account.RoutingCandidate{Credential: account.Credential{Provider: account.ProviderBuild, Enabled: true, AuthStatus: account.AuthStatusActive}, QuotaRecovery: &account.QuotaRecovery{Kind: account.QuotaRecoveryKindFree, Status: account.QuotaRecoveryStatusExhausted, NextProbeAt: &due}}
	for _, tc := range []struct {
		name            string
		change          func(*account.RoutingCandidate)
		includeDisabled bool
		eligible        bool
	}{
		{"due", func(c *account.RoutingCandidate) {}, false, true},
		{"future", func(c *account.RoutingCandidate) {
			r := *c.QuotaRecovery
			r.NextProbeAt = &future
			c.QuotaRecovery = &r
		}, false, false},
		{"disabled", func(c *account.RoutingCandidate) { c.Credential.Enabled = false }, false, false},
		{"disabled maintenance", func(c *account.RoutingCandidate) { c.Credential.Enabled = false }, true, true},
		{"reauth", func(c *account.RoutingCandidate) { c.Credential.AuthStatus = account.AuthStatusReauthRequired }, true, false},
		{"cooldown", func(c *account.RoutingCandidate) { c.Credential.CooldownUntil = &future }, false, false},
		{"model cooldown", func(c *account.RoutingCandidate) { c.ModelQuotaBlock = &account.ModelQuotaBlock{CooldownUntil: future} }, false, false},
		{"model denied", func(c *account.RoutingCandidate) { c.ModelCapabilityKnown = true; c.SupportsModel = false }, false, false},
		{"egress cooldown", func(c *account.RoutingCandidate) {
			c.Credential.EgressNodeID = 7
			c.EgressLeaseBlock = &account.EgressLeaseBlock{NodeID: 7, CooldownUntil: future}
		}, false, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := base
			tc.change(&c)
			if got := quotaRecoveryCandidateReason(c, now, tc.includeDisabled, true); (got == "") != tc.eligible {
				t.Fatalf("reason=%q", got)
			}
		})
	}
}

func newQuotaRecoveryFixture(t *testing.T, adapter provider.Adapter) (*Service, *relational.AccountRepository, account.Credential, clientkey.Key) {
	t.Helper()
	ctx := context.Background()
	db, err := relational.OpenSQLite(ctx, filepath.Join(t.TempDir(), "quota.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = db.Close() })
	if err = db.InitializeSchema(ctx); err != nil {
		t.Fatal(err)
	}
	repo := relational.NewAccountRepository(db)
	models := relational.NewModelRepository(db)
	audits := relational.NewAuditRepository(db)
	value, _, err := repo.UpsertByIdentity(ctx, account.Credential{Provider: account.ProviderBuild, Name: "quota", SourceKey: "quota", EncryptedAccessToken: "access", EncryptedRefreshToken: "refresh", ExpiresAt: time.Now().Add(time.Hour), Enabled: true, AuthStatus: account.AuthStatusActive, MaxConcurrent: 2})
	if err != nil {
		t.Fatal(err)
	}
	if err = models.UpsertDiscovered(ctx, account.ProviderBuild, []string{"grok-test"}); err != nil {
		t.Fatal(err)
	}
	if err = models.ReplaceAccountCapabilities(ctx, value.ID, []string{"grok-test"}, time.Now().UTC()); err != nil {
		t.Fatal(err)
	}
	due := time.Now().UTC().Add(-time.Minute)
	if err = repo.SaveQuotaRecovery(ctx, account.QuotaRecovery{AccountID: value.ID, Kind: account.QuotaRecoveryKindFree, Status: account.QuotaRecoveryStatusExhausted, NextProbeAt: &due, UpdatedAt: due}); err != nil {
		t.Fatal(err)
	}
	key, err := relational.NewClientKeyRepository(db).Create(ctx, clientkey.Key{Name: "maintenance", Prefix: "maintenance", SecretHash: strings.Repeat("a", 64), EncryptedSecret: "encrypted", InternalKind: "account_recovery", Enabled: true})
	if err != nil {
		t.Fatal(err)
	}
	registry := provider.NewRegistry(adapter)
	sticky := memory.NewStickyStore()
	accounts := accountapp.NewService(repo, audits, memory.NewDeviceSessionStore(), sticky, registry, testCipher(t), nil)
	selector := NewSelector(repo, memory.NewConcurrencyLimiter(), sticky, registry, time.Hour, time.Second, time.Minute)
	service := NewService(models, audits, accounts, clientkeyapp.NewService(nil, nil, nil, 60, 4, nil), registry, selector, relational.NewResponseRepository(db), 1)
	service.SetQuotaRecoveryIdentity(key)
	return service, repo, value, key
}

type quotaRecoveryAdapter struct {
	started        chan struct{}
	release        chan struct{}
	calls          atomic.Int64
	billingCalls   atomic.Int64
	body           string
	failure        error
	catalogChanged bool
}

func (a *quotaRecoveryAdapter) Provider() account.Provider { return account.ProviderBuild }
func (a *quotaRecoveryAdapter) Definition() provider.Definition {
	d := testConversationDefinition(account.ProviderBuild)
	d.Quota = provider.QuotaBilling
	return d
}
func (a *quotaRecoveryAdapter) RefreshCredential(context.Context, account.Credential) (provider.RefreshedCredential, error) {
	return provider.RefreshedCredential{}, errors.New("unexpected refresh")
}
func (a *quotaRecoveryAdapter) GetBilling(context.Context, account.Credential) (account.Billing, error) {
	a.billingCalls.Add(1)
	return account.Billing{}, nil
}
func (a *quotaRecoveryAdapter) ForwardResponse(ctx context.Context, req provider.ResponseResourceRequest) (*provider.Response, error) {
	if a.calls.Add(1) == 1 && a.started != nil {
		close(a.started)
	}
	if a.release != nil {
		select {
		case <-a.release:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	if a.failure != nil {
		return nil, a.failure
	}
	return &provider.Response{StatusCode: 200, ModelCatalogChanged: a.catalogChanged, Status: "200 OK", Header: make(http.Header), Body: io.NopCloser(strings.NewReader(a.body))}, nil
}

const recoveryGoodStream = "data: {\"choices\":[{\"delta\":{\"content\":\"OK\"}}]}\n\ndata: [DONE]\n\n"

func TestBuildQuotaRecoveryConcurrentBusinessClaimAndCancellation(t *testing.T) {
	adapter := &quotaRecoveryAdapter{started: make(chan struct{}), release: make(chan struct{}), body: recoveryGoodStream}
	service, repo, value, _ := newQuotaRecoveryFixture(t, adapter)
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		_, err := service.ProbeBuildQuotaRecovery(ctx, value.ID, []string{"grok-test"}, false)
		done <- err
	}()
	select {
	case <-adapter.started:
	case <-time.After(3 * time.Second):
		t.Fatal("probe did not reach upstream")
	}
	recovery, err := repo.GetQuotaRecovery(context.Background(), value.ID)
	if err != nil || recovery.Status != account.QuotaRecoveryStatusProbing {
		t.Fatalf("recovery=%+v err=%v", recovery, err)
	}
	result, err := service.ProbeBuildQuotaRecovery(context.Background(), value.ID, []string{"grok-test"}, false)
	if err != nil || !result.Skipped || result.Claimed {
		t.Fatalf("second=%+v err=%v", result, err)
	}
	if lease, err := service.selector.Acquire(context.Background(), account.ProviderBuild, 0, "grok-test", "", "", nil, true); err == nil {
		lease.Release()
		t.Fatal("business acquired probing account")
	}
	cancel()
	select {
	case err := <-done:
		if err == nil {
			t.Fatal("canceled probe succeeded")
		}
	case <-time.After(3 * time.Second):
		t.Fatal("cancellation did not interrupt")
	}
	if adapter.calls.Load() != 1 {
		t.Fatalf("calls=%d", adapter.calls.Load())
	}
	if _, err = repo.GetQuotaRecovery(context.Background(), value.ID); err != nil {
		t.Fatal("cancel cleared recovery", err)
	}
	latest, _ := repo.Get(context.Background(), value.ID)
	if latest.AuthStatus != account.AuthStatusActive {
		t.Fatal("cancel invalidated authentication")
	}
}

func TestBuildQuotaRecoveryOutcomes(t *testing.T) {
	for _, tc := range []struct {
		name, body                           string
		disabled, includeDisabled, recovered bool
		failure                              error
	}{
		{name: "success", body: recoveryGoodStream, recovered: true},
		{name: "disabled skipped", body: recoveryGoodStream, disabled: true},
		{name: "disabled maintained", body: recoveryGoodStream, disabled: true, includeDisabled: true, recovered: true},
		{name: "empty", body: "data: [DONE]\n\n"},
		{name: "truncated", body: "data: {\"choices\":[{\"delta\":{\"content\":\"OK\"}}]}\n\n"},
		{name: "network", failure: context.DeadlineExceeded},
	} {
		t.Run(tc.name, func(t *testing.T) {
			adapter := &quotaRecoveryAdapter{body: tc.body, failure: tc.failure}
			service, repo, value, _ := newQuotaRecoveryFixture(t, adapter)
			if tc.disabled {
				value.Enabled = false
				if _, err := repo.Update(context.Background(), value); err != nil {
					t.Fatal(err)
				}
			}
			result, err := service.ProbeBuildQuotaRecovery(context.Background(), value.ID, []string{"missing", "grok-test"}, tc.includeDisabled)
			if result.Recovered != tc.recovered {
				t.Fatalf("result=%+v err=%v", result, err)
			}
			latest, _ := repo.Get(context.Background(), value.ID)
			if latest.Enabled == tc.disabled {
				t.Fatal("maintenance changed enabled state")
			}
			if latest.AuthStatus != account.AuthStatusActive {
				t.Fatal("transient outcome invalidated authentication")
			}
			if !tc.recovered {
				r, err := repo.GetQuotaRecovery(context.Background(), value.ID)
				if err != nil {
					t.Fatal(err)
				}
				if !result.Skipped && (r.NextProbeAt == nil || r.NextProbeAt.Before(time.Now())) {
					t.Fatal("failed probe has no backoff")
				}
			}
		})
	}
}

func TestBuildQuotaRecoveryPaidDoesNotInfer(t *testing.T) {
	adapter := &quotaRecoveryAdapter{}
	service, repo, value, _ := newQuotaRecoveryFixture(t, adapter)
	due := time.Now().UTC().Add(-time.Minute)
	if err := repo.SaveQuotaRecovery(context.Background(), account.QuotaRecovery{AccountID: value.ID, Kind: account.QuotaRecoveryKindPaid, Status: account.QuotaRecoveryStatusExhausted, NextProbeAt: &due, UpdatedAt: due}); err != nil {
		t.Fatal(err)
	}
	result, err := service.ProbeBuildQuotaRecovery(context.Background(), value.ID, nil, false)
	if err != nil || !result.Claimed || !result.Recovered {
		t.Fatalf("result=%+v err=%v", result, err)
	}
	if adapter.calls.Load() != 0 || adapter.billingCalls.Load() != 1 {
		t.Fatalf("inference=%d billing=%d", adapter.calls.Load(), adapter.billingCalls.Load())
	}
}

func TestBusinessQuotaRecoveryWaitsForFinalization(t *testing.T) {
	adapter := &quotaRecoveryAdapter{body: recoveryGoodStream}
	service, repo, value, key := newQuotaRecoveryFixture(t, adapter)
	result, err := service.CreateChatCompletion(context.Background(), Input{RequestID: "quota-business", ClientKey: key, PublicModel: "grok-test", Streaming: true, skipQualityHold: true, Body: []byte(`{"model":"grok-test","stream":true,"messages":[{"role":"user","content":"OK"}]}`)})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := repo.GetQuotaRecovery(context.Background(), value.ID); err != nil {
		t.Fatal("2xx headers cleared recovery", err)
	}
	if err := consumeQuotaRecoveryStream(context.Background(), result); err != nil {
		t.Fatal(err)
	}
	if _, err := repo.GetQuotaRecovery(context.Background(), value.ID); !errors.Is(err, repository.ErrNotFound) {
		t.Fatalf("successful finalization did not recover: %v", err)
	}
}

func TestBusinessQuotaRecoveryLateFinalizerKeepsNewClaim(t *testing.T) {
	adapter := &quotaRecoveryAdapter{body: recoveryGoodStream}
	service, repo, value, key := newQuotaRecoveryFixture(t, adapter)
	result, err := service.CreateChatCompletion(context.Background(), Input{RequestID: "quota-business", ClientKey: key, PublicModel: "grok-test", Streaming: true, skipQualityHold: true, Body: []byte(`{"model":"grok-test","stream":true}`)})
	if err != nil {
		t.Fatal(err)
	}
	newer := time.Now().UTC().Add(10 * time.Minute)
	if err := repo.SaveQuotaRecovery(context.Background(), account.QuotaRecovery{AccountID: value.ID, Kind: account.QuotaRecoveryKindFree, Status: account.QuotaRecoveryStatusProbing, NextProbeAt: &newer, UpdatedAt: time.Now().UTC()}); err != nil {
		t.Fatal(err)
	}
	if err := consumeQuotaRecoveryStream(context.Background(), result); err != nil {
		t.Fatal(err)
	}
	state, err := repo.GetQuotaRecovery(context.Background(), value.ID)
	if err != nil || state.NextProbeAt == nil || !state.NextProbeAt.Equal(newer) {
		t.Fatalf("late finalizer changed new claim: %+v %v", state, err)
	}
	old := time.Now().UTC().Add(time.Minute)
	service.selector.MarkFreeQuotaExhausted(context.WithValue(context.Background(), quotaRecoveryClaimContextKey{}, old), value, 10, 10)
	state, err = repo.GetQuotaRecovery(context.Background(), value.ID)
	if err != nil || state.NextProbeAt == nil || !state.NextProbeAt.Equal(newer) {
		t.Fatalf("late failure changed new claim: %+v %v", state, err)
	}
}

type quotaRecoveryLateAudit struct {
	auditRecorder
	onReady func()
}

func (a quotaRecoveryLateAudit) CheckLedgerReady() error { a.onReady(); return nil }

func TestBuildQuotaRecoveryStopsForLateTeamRateLimit(t *testing.T) {
	for _, cancelRequest := range []bool{false, true} {
		t.Run(fmt.Sprintf("cancel_%t", cancelRequest), func(t *testing.T) {
			adapter := &quotaRecoveryAdapter{body: recoveryGoodStream}
			service, repo, value, _ := newQuotaRecoveryFixture(t, adapter)
			service.logger = slog.New(slog.NewTextHandler(io.Discard, nil))
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			service.audits = quotaRecoveryLateAudit{auditRecorder: service.audits, onReady: func() {
				service.markTeamModelRateLimit(value, "grok-test", provider.RateLimitMetadata{TeamID: "late-team", RetryAfter: 250 * time.Millisecond}, time.Now().UTC())
				if cancelRequest {
					cancel()
				}
			}}
			started := time.Now()
			result, err := service.ProbeBuildQuotaRecovery(ctx, value.ID, []string{"grok-test"}, false)
			if elapsed := time.Since(started); elapsed >= 200*time.Millisecond {
				t.Errorf("late limit blocked the worker for %s", elapsed)
			}
			if err == nil || !result.Claimed || result.Recovered {
				t.Fatalf("result=%+v error=%v", result, err)
			}
			if adapter.calls.Load() != 0 {
				t.Fatalf("sent %d requests after late limit", adapter.calls.Load())
			}
			if _, err := repo.GetQuotaRecovery(context.Background(), value.ID); err != nil {
				t.Fatal("late limit cleared recovery", err)
			}
		})
	}
}

func TestBuildQuotaRecoveryRespectsBotFlagSchedulingExclusion(t *testing.T) {
	adapter := &quotaRecoveryAdapter{body: recoveryGoodStream}
	service, repo, value, _ := newQuotaRecoveryFixture(t, adapter)
	if err := repo.UpdateBuildBotFlagSources(context.Background(), []repository.BuildBotFlagSourceUpdate{{AccountID: value.ID, ExpectedEncryptedAccessToken: value.EncryptedAccessToken, Source: 1}}); err != nil {
		t.Fatal(err)
	}
	service.selector.UpdateExcludeBuildBotFlaggedFromScheduling(true)
	result, err := service.ProbeBuildQuotaRecovery(context.Background(), value.ID, []string{"grok-test"}, false)
	if err != nil || !result.Skipped || result.Claimed || adapter.calls.Load() != 0 {
		t.Fatalf("excluded account probed: %+v error=%v calls=%d", result, err, adapter.calls.Load())
	}
}

type quotaRecoveryCatalogObserver struct {
	routeResolver
	called chan struct{}
}

func (r *quotaRecoveryCatalogObserver) SyncAccount(context.Context, uint64) (int, error) {
	r.called <- struct{}{}
	return 0, nil
}

func TestBuildQuotaRecoveryDoesNotQueueCatalogSync(t *testing.T) {
	adapter := &quotaRecoveryAdapter{body: recoveryGoodStream, catalogChanged: true}
	service, _, value, _ := newQuotaRecoveryFixture(t, adapter)
	observer := &quotaRecoveryCatalogObserver{routeResolver: service.models, called: make(chan struct{}, 1)}
	service.models = observer
	result, err := service.ProbeBuildQuotaRecovery(context.Background(), value.ID, []string{"grok-test"}, false)
	if err != nil || !result.Recovered {
		t.Fatalf("result=%+v err=%v", result, err)
	}
	select {
	case <-observer.called:
		t.Fatal("maintenance scheduled an extra catalog request outside its budget")
	case <-time.After(20 * time.Millisecond):
	}
}
