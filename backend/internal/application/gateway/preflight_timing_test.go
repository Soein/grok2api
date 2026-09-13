package gateway

import (
	"context"
	"testing"
	"time"

	modeldomain "github.com/chenyme/grok2api/backend/internal/domain/model"
	infraegress "github.com/chenyme/grok2api/backend/internal/infra/egress"
	"github.com/chenyme/grok2api/backend/internal/infra/provider"
)

type preflightRouteResolver struct{ routeResolver }

func (r preflightRouteResolver) GetByPublicIDCandidates(ctx context.Context, publicID string) ([]modeldomain.Route, error) {
	time.Sleep(20 * time.Millisecond)
	return r.routeResolver.GetByPublicIDCandidates(ctx, publicID)
}

type preflightCaptureAdapter struct {
	quotaRecoveryAdapter
	snapshot *infraegress.PreflightTimingSnapshot
}

func (a *preflightCaptureAdapter) ForwardResponse(ctx context.Context, req provider.ResponseResourceRequest) (*provider.Response, error) {
	a.snapshot = infraegress.PreflightTimingFromContext(ctx).Snapshot()
	return a.quotaRecoveryAdapter.ForwardResponse(ctx, req)
}

func TestGatewayPreflightIncludesRouteWorkAndHonorsSwitch(t *testing.T) {
	for _, enabled := range []bool{false, true} {
		t.Run(map[bool]string{false: "disabled", true: "enabled"}[enabled], func(t *testing.T) {
			adapter := &preflightCaptureAdapter{quotaRecoveryAdapter: quotaRecoveryAdapter{body: recoveryGoodStream}}
			service, repo, credential, key := newQuotaRecoveryFixture(t, adapter)
			if err := repo.ClearQuotaRecovery(context.Background(), credential.ID); err != nil {
				t.Fatal(err)
			}
			service.models = preflightRouteResolver{service.models}
			service.UpdateRequestTimingEnabled(enabled)
			result, err := service.CreateResponse(context.Background(), Input{RequestID: "preflight-test", ClientKey: key, PublicModel: "grok-test", Streaming: true, Body: []byte(`{"model":"grok-test","input":"hello","stream":true}`)})
			if err != nil {
				t.Fatal(err)
			}
			result.Finalize(Usage{}, "", "")
			_ = result.Body.Close()
			if !enabled {
				if adapter.snapshot != nil {
					t.Fatalf("disabled collector: %#v", adapter.snapshot)
				}
				return
			}
			if adapter.snapshot == nil {
				t.Fatal("missing Gateway preflight snapshot at adapter entry")
			}
			route := adapter.snapshot.Stages["route_resolve"]
			if route.Count != 1 || route.DurationMS < 20 {
				t.Fatalf("route timing = %#v", route)
			}
			if adapter.snapshot.TotalMS < route.DurationMS {
				t.Fatalf("preflight total does not include route: %#v", adapter.snapshot)
			}
			for _, stage := range []string{"route_ownership", "preselection", "media_summary", "session_identity", "billing_estimate", "selection_acquire", "credential"} {
				if adapter.snapshot.Stages[stage].Count == 0 {
					t.Errorf("stage %s missing: %#v", stage, adapter.snapshot)
				}
			}
			if adapter.snapshot.Counters["adapter_calls"] != 1 {
				t.Fatalf("adapter ordinal = %d", adapter.snapshot.Counters["adapter_calls"])
			}
		})
	}
}
