package gateway

import (
	"context"
	"errors"
	"net/http"
	"path/filepath"
	"strings"
	"testing"
	"time"

	accountapp "github.com/chenyme/grok2api/backend/internal/application/account"
	clientkeyapp "github.com/chenyme/grok2api/backend/internal/application/clientkey"
	"github.com/chenyme/grok2api/backend/internal/domain/account"
	"github.com/chenyme/grok2api/backend/internal/domain/clientkey"
	"github.com/chenyme/grok2api/backend/internal/domain/media"
	"github.com/chenyme/grok2api/backend/internal/domain/model"
	"github.com/chenyme/grok2api/backend/internal/infra/persistence/relational"
	"github.com/chenyme/grok2api/backend/internal/infra/provider"
	"github.com/chenyme/grok2api/backend/internal/infra/runtime/memory"
	"github.com/chenyme/grok2api/backend/internal/repository"
)

type videoJobSafetyFixture struct {
	service *Service
	keys    *relational.ClientKeyRepository
	models  *relational.ModelRepository
	jobs    *relational.MediaJobRepository
	key     clientkey.Key
	route   model.Route
	first   account.Credential
	second  account.Credential
	limiter *memory.ConcurrencyLimiter
	adapter *videoCreateFailoverAdapter
	job     media.Job
}

func newVideoJobSafetyFixture(t *testing.T) *videoJobSafetyFixture {
	t.Helper()
	ctx := context.Background()
	database, err := relational.OpenSQLite(ctx, filepath.Join(t.TempDir(), "video-safety.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = database.Close() })
	if err := database.InitializeSchema(ctx); err != nil {
		t.Fatal(err)
	}
	accounts := relational.NewAccountRepository(database)
	models := relational.NewModelRepository(database)
	audits := relational.NewAuditRepository(database)
	f := &videoJobSafetyFixture{
		keys:    relational.NewClientKeyRepository(database),
		models:  models,
		jobs:    relational.NewMediaJobRepository(database),
		limiter: memory.NewConcurrencyLimiter(),
		adapter: &videoCreateFailoverAdapter{failures: make(map[uint64]int)},
	}
	f.key, err = f.keys.Create(ctx, clientkey.Key{
		Name: "video-safety", Prefix: "video-safety", SecretHash: strings.Repeat("a", 64),
		EncryptedSecret: "encrypted", Enabled: true, RPMLimit: 60, MaxConcurrent: 4,
	})
	if err != nil {
		t.Fatal(err)
	}
	createAccount := func(name string, tier account.WebTier, priority int) account.Credential {
		t.Helper()
		credential, _, err := accounts.UpsertByIdentity(ctx, account.Credential{
			Provider: account.ProviderWeb, AuthType: account.AuthTypeSSO, WebTier: tier,
			Name: name, SourceKey: name, EncryptedAccessToken: name + "-token", ExpiresAt: time.Now().Add(time.Hour),
			Enabled: true, AuthStatus: account.AuthStatusActive, Priority: priority, MaxConcurrent: 1,
		})
		if err != nil {
			t.Fatal(err)
		}
		return credential
	}
	f.first = createAccount("free", account.WebTierBasic, 200)
	f.second = createAccount("super", account.WebTierSuper, 100)
	if err := models.UpsertDiscovered(ctx, account.ProviderWeb, []string{"grok-imagine-video"}); err != nil {
		t.Fatal(err)
	}
	for _, id := range []uint64{f.first.ID, f.second.ID} {
		if err := models.ReplaceAccountCapabilities(ctx, id, []string{"grok-imagine-video"}, time.Now().UTC()); err != nil {
			t.Fatal(err)
		}
	}
	f.route, err = models.GetByProviderUpstream(ctx, account.ProviderWeb, "grok-imagine-video")
	if err != nil {
		t.Fatal(err)
	}
	registry := provider.NewRegistry(f.adapter)
	sticky := memory.NewStickyStore()
	accountService := accountapp.NewService(accounts, audits, memory.NewDeviceSessionStore(), sticky, registry, testCipher(t), nil)
	selector := NewSelector(accounts, f.limiter, sticky, registry, time.Hour, time.Second, time.Minute)
	f.service = NewService(models, audits, accountService, clientkeyapp.NewService(f.keys, nil, nil, 60, 4, nil), registry, selector, nil, 3)
	f.service.ConfigureMedia(f.jobs, 1)
	f.service.UpdateVideoMaxAttempts(3)
	now := time.Now().UTC()
	f.job = media.Job{
		ID: "video_safety", RequestID: "request-video-safety", ClientKeyID: f.key.ID, ClientKeyName: f.key.Name,
		AccountID: f.first.ID, AccountName: f.first.Name, Provider: string(account.ProviderWeb),
		Model: f.route.PublicID, ModelRouteID: f.route.ID, UpstreamModel: f.route.UpstreamModel,
		Operation: provider.VideoOperationGenerate, Prompt: "test", Seconds: 5, Quality: "720p",
		Status: media.StatusInProgress, InputJSON: `{}`, CreatedAt: now, UpdatedAt: now,
	}
	if err := f.jobs.CreateMediaJob(ctx, f.job); err != nil {
		t.Fatal(err)
	}
	return f
}

func (f *videoJobSafetyFixture) storedJob(t *testing.T) media.Job {
	t.Helper()
	job, err := f.jobs.GetMediaJob(context.Background(), f.job.ID, f.job.ClientKeyID)
	if err != nil {
		t.Fatal(err)
	}
	return job
}

func TestVideoJobReleasesAccountLease(t *testing.T) {
	for _, test := range []struct {
		name     string
		status   int
		failures int
	}{
		{name: "success"},
		{name: "unclassified error", failures: 1},
		{name: "server error", status: http.StatusInternalServerError, failures: 1},
		{name: "forbidden attempts exhausted", status: http.StatusForbidden, failures: 1},
		{name: "rate limit attempts exhausted", status: http.StatusTooManyRequests, failures: 1},
	} {
		t.Run(test.name, func(t *testing.T) {
			f := newVideoJobSafetyFixture(t)
			f.adapter.failures[f.first.ID] = test.failures
			f.adapter.status = test.status
			f.service.UpdateVideoMaxAttempts(1)
			f.service.runVideoJob(context.Background(), f.job, f.route)
			wantStatus := media.StatusCompleted
			if test.failures > 0 {
				wantStatus = media.StatusFailed
			}
			if job := f.storedJob(t); job.Status != wantStatus {
				t.Fatalf("job status = %s, want %s", job.Status, wantStatus)
			}
			count, err := f.limiter.Current(context.Background(), accountConcurrencyKey(f.first.ID))
			if err != nil || count != 0 {
				t.Fatalf("account slots after video completion = %d, err=%v, want 0", count, err)
			}
		})
	}
}

func TestVideoJobPreservesCurrentKeyAccountScope(t *testing.T) {
	for _, scenario := range []string{"forbidden retry", "rate limit retry", "pinned account saturated", "pinned scope changed"} {
		t.Run(scenario, func(t *testing.T) {
			ctx := context.Background()
			f := newVideoJobSafetyFixture(t)
			f.key.ProviderScope, f.key.TierScope = clientkey.ProviderScopeWeb, clientkey.TierScopeFree
			if _, err := f.keys.Update(ctx, f.key); err != nil {
				t.Fatal(err)
			}
			switch scenario {
			case "forbidden retry", "rate limit retry":
				f.adapter.failures[f.first.ID] = 2
				f.adapter.status = http.StatusForbidden
				if scenario == "rate limit retry" {
					f.adapter.status = http.StatusTooManyRequests
				}
			case "pinned account saturated":
				release, acquired, err := f.limiter.Acquire(ctx, accountConcurrencyKey(f.first.ID), 1)
				if err != nil || !acquired {
					t.Fatalf("saturate pinned account: acquired=%v err=%v", acquired, err)
				}
				t.Cleanup(release)
			case "pinned scope changed":
				// A recovered job may still reference an account allowed by an older key policy.
				f.job.AccountID, f.job.AccountName = f.second.ID, f.second.Name
			}
			f.service.runVideoJob(ctx, f.job, f.route)
			attempts := f.adapter.Attempts()
			for _, id := range attempts {
				if id != f.first.ID {
					t.Fatalf("Free-only key used forbidden account %d: attempts=%v", id, attempts)
				}
			}
			wantStatus, wantAttempts := media.StatusFailed, 1
			switch scenario {
			case "forbidden retry":
				wantAttempts = 2
			case "pinned account saturated":
				wantAttempts = 0
			case "pinned scope changed":
				wantStatus = media.StatusCompleted
			}
			if len(attempts) != wantAttempts || f.storedJob(t).Status != wantStatus {
				t.Fatalf("attempts=%v status=%s, want %d attempts and %s", attempts, f.storedJob(t).Status, wantAttempts, wantStatus)
			}
		})
	}
}

func TestVideoJobRejectsUnavailableClientKey(t *testing.T) {
	for _, scenario := range []string{"disabled", "expired", "missing", "model permission removed", "provider permission removed"} {
		t.Run(scenario, func(t *testing.T) {
			f := newVideoJobSafetyFixture(t)
			switch scenario {
			case "disabled":
				f.key.Enabled = false
			case "expired":
				past := time.Now().Add(-time.Minute)
				f.key.ExpiresAt = &past
			case "missing":
				f.service.clientKeys = clientkeyapp.NewService(videoKeyReadErrorRepository{ClientKeyRepository: f.keys, err: repository.ErrNotFound}, nil, nil, 60, 4, nil)
			case "model permission removed":
				ctx := context.Background()
				other, err := f.models.Create(ctx, model.Route{PublicID: "other-video", Provider: account.ProviderWeb, UpstreamModel: f.route.UpstreamModel, Capability: model.CapabilityVideo, Enabled: true}, []uint64{f.first.ID})
				if err != nil {
					t.Fatal(err)
				}
				f.key.AllowedModels = []uint64{other.ID}
			case "provider permission removed":
				f.key.ProviderScope = clientkey.ProviderScopeBuild
			}
			if _, err := f.keys.Update(context.Background(), f.key); err != nil {
				t.Fatal(err)
			}
			f.service.runVideoJob(context.Background(), f.job, f.route)
			if attempts := f.adapter.Attempts(); len(attempts) != 0 {
				t.Fatalf("unavailable key reached upstream: %v", attempts)
			}
			if job := f.storedJob(t); job.Status != media.StatusFailed {
				t.Fatalf("job status = %s, want failed", job.Status)
			}
		})
	}
}

type videoKeyReadErrorRepository struct {
	repository.ClientKeyRepository
	err error
}

func (r videoKeyReadErrorRepository) Get(context.Context, uint64) (clientkey.Key, error) {
	return clientkey.Key{}, r.err
}

func TestVideoJobDefersWhenKeyPolicyCannotBeLoaded(t *testing.T) {
	f := newVideoJobSafetyFixture(t)
	f.service.clientKeys = clientkeyapp.NewService(videoKeyReadErrorRepository{ClientKeyRepository: f.keys, err: errors.New("database unavailable")}, nil, nil, 60, 4, nil)
	f.service.runVideoJob(context.Background(), f.job, f.route)
	if attempts := f.adapter.Attempts(); len(attempts) != 0 {
		t.Fatalf("key policy lookup failed but upstream was called: %v", attempts)
	}
	if job := f.storedJob(t); job.Status != media.StatusInProgress || job.CompletedAt != nil || job.LeaseUntil == nil || !job.LeaseUntil.After(time.Now()) {
		t.Fatalf("temporary key lookup failure terminated job: %#v", job)
	}
}
