package gateway

import (
	"testing"
	"time"

	"github.com/chenyme/grok2api/backend/internal/domain/account"
	"github.com/chenyme/grok2api/backend/internal/pkg/perfmetrics"
	"github.com/chenyme/grok2api/backend/internal/repository"
)

func TestSelectorInvalidationMetricsHonorDiagnosticSwitch(t *testing.T) {
	previous := perfmetrics.Default
	defer func() { perfmetrics.Default = previous }()
	perfmetrics.Default = perfmetrics.NewRegistry()
	selector := NewSelector(nil, nil, nil, nil, time.Hour, time.Second, time.Minute)
	service := &Service{selector: selector}
	event := repository.InvalidationEvent{Kind: repository.InvalidationAccountBillingChanged, Provider: account.ProviderWeb, AccountID: 123}
	selector.ApplyInvalidation(event)
	if len(perfmetrics.Default.CollectAndReset()) != 0 {
		t.Fatal("disabled diagnostics emitted metrics")
	}
	service.UpdateRequestTimingEnabled(true)
	selector.ApplyInvalidation(event)
	samples := perfmetrics.Default.CollectAndReset()
	if len(samples) != 1 || samples[0].Name != "selector_cache_invalidation_total" || samples[0].Labels.Provider != string(account.ProviderWeb) || samples[0].Labels.Stage != string(event.Kind) || samples[0].Labels.Outcome != "provider" || samples[0].Total != 1 {
		t.Fatalf("scoped diagnostic = %+v", samples)
	}
	selector.ApplyInvalidation(repository.InvalidationEvent{Kind: repository.InvalidationAccountBillingChanged, AccountID: 999})
	samples = perfmetrics.Default.CollectAndReset()
	if len(samples) != 1 || samples[0].Labels.Outcome != "global" || samples[0].Labels.Provider != "" {
		t.Fatalf("unknown-account diagnostic = %+v", samples)
	}
	service.UpdateRequestTimingEnabled(false)
	selector.ApplyInvalidation(event)
	if len(perfmetrics.Default.CollectAndReset()) != 0 {
		t.Fatal("diagnostics did not turn off")
	}
}
