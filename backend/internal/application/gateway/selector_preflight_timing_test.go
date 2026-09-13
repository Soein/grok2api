package gateway

import (
	"context"
	"testing"
	"time"

	"github.com/chenyme/grok2api/backend/internal/domain/account"
	"github.com/chenyme/grok2api/backend/internal/infra/egress"
	"github.com/chenyme/grok2api/backend/internal/repository"
)

func TestSelectorPreflightSeparatesLoadAndFilter(t *testing.T) {
	repo := newLayeredRepositoryFixture()
	repo.baseHook = func() { time.Sleep(25 * time.Millisecond) }
	selector := NewSelector(repo, nil, nil, nil, time.Hour, time.Second, time.Minute)
	ctx, p := egress.WithPreflightTiming(context.Background(), time.Now())
	session, err := selector.beginSelectionSession(ctx, account.ProviderBuild, 0, "model-a", "", "", nil, false)
	if err != nil || len(session.normalCandidates) != 1 {
		t.Fatalf("session=%v error=%v", session, err)
	}
	snapshot := p.Snapshot()
	if snapshot.Stages["base_query"].DurationMS < 20 || snapshot.Stages["candidate_load"].DurationMS < 20 {
		t.Fatalf("missing real query latency: %+v", snapshot)
	}
	if snapshot.Stages["candidate_filter"].Count != 1 || snapshot.Stages["candidate_filter"].DurationMS >= snapshot.Stages["base_query"].DurationMS {
		t.Fatalf("filter includes load: %+v", snapshot)
	}
	if snapshot.Counters["candidate_cache_miss"] != 1 || snapshot.Counters["candidate_loader_executed"] != 1 || snapshot.Counters["candidates_loaded"] != 1 || snapshot.Counters["candidates_normal"] != 1 {
		t.Fatalf("counters = %+v", snapshot.Counters)
	}
	ctx, p = egress.WithPreflightTiming(context.Background(), time.Now())
	if _, err := selector.beginSelectionSession(ctx, account.ProviderBuild, 0, "model-a", "", "", nil, false); err != nil {
		t.Fatal(err)
	}
	cached := p.Snapshot()
	if cached.Counters["candidate_cache_hit"] != 1 || cached.Stages["base_query"].Count != 0 || cached.Stages["candidate_shared_load"].Count != 0 {
		t.Fatalf("cache hit performed query: %+v", cached)
	}
}

func TestSelectorPreflightRecordsVersionChurnFallback(t *testing.T) {
	repo := newLayeredRepositoryFixture()
	repo.combined = []account.RoutingCandidate{{Credential: account.Credential{ID: 9, Provider: account.ProviderBuild}}}
	selector := NewSelector(repo, nil, nil, nil, time.Hour, time.Second, time.Minute)
	repo.baseHook = func() {
		selector.ApplyInvalidation(repository.InvalidationEvent{Kind: repository.InvalidationAccountStateChanged, Provider: account.ProviderBuild})
	}
	ctx, timing := egress.WithPreflightTiming(context.Background(), time.Now())
	values, err := selector.loadCandidates(ctx, account.ProviderBuild, 0, "model-a", "", time.Now())
	if err != nil || len(values) != 1 || values[0].Credential.ID != 9 {
		t.Fatalf("values=%v error=%v", values, err)
	}
	snapshot := timing.Snapshot()
	if snapshot.Counters["candidate_version_retry_after_load"] != 1 || snapshot.Counters["candidate_combined_fallback"] != 1 || snapshot.Stages["base_query"].Count != 1 || snapshot.Stages["combined_query"].Count != 1 {
		t.Fatalf("missing churn diagnostics: %+v", snapshot)
	}
}

func TestSelectorPreflightSharedLoadAttributesOnlyExecutingCaller(t *testing.T) {
	repo := newLayeredRepositoryFixture()
	repo.firstBaseStart = make(chan struct{})
	repo.firstBaseReady = make(chan struct{})
	selector := NewSelector(repo, nil, nil, nil, time.Hour, time.Second, time.Minute)
	leaderCtx, leaderTiming := egress.WithPreflightTiming(context.Background(), time.Now())
	waiterCtx, waiterTiming := egress.WithPreflightTiming(context.Background(), time.Now())
	type outcome struct {
		values []account.RoutingCandidate
		err    error
	}
	load := func(ctx context.Context, done chan<- outcome) {
		values, err := selector.loadCandidates(ctx, account.ProviderBuild, 0, "model-a", "", time.Now())
		done <- outcome{values: values, err: err}
	}
	leaderDone := make(chan outcome, 1)
	waiterDone := make(chan outcome, 1)
	go load(leaderCtx, leaderDone)
	released := false
	defer func() {
		if !released {
			close(repo.firstBaseReady)
		}
	}()
	select {
	case <-repo.firstBaseStart:
	case <-time.After(5 * time.Second):
		t.Fatal("leader did not reach the repository")
	}
	go load(waiterCtx, waiterDone)
	deadline := time.NewTimer(5 * time.Second)
	defer deadline.Stop()
	ticker := time.NewTicker(time.Millisecond)
	defer ticker.Stop()
	for waiterTiming.Snapshot().Counters["candidate_cache_miss"] == 0 {
		select {
		case <-ticker.C:
		case <-deadline.C:
			t.Fatal("waiter did not reach the shared-load path")
		}
	}
	// The counter is immediately before Do. Keep the leader blocked briefly to
	// let the observed waiter enter Do; repository entry itself is channel-gated.
	time.Sleep(20 * time.Millisecond)
	select {
	case got := <-waiterDone:
		t.Fatalf("waiter completed while the repository was blocked: %+v", got)
	default:
	}
	close(repo.firstBaseReady)
	released = true
	for _, done := range []<-chan outcome{leaderDone, waiterDone} {
		select {
		case got := <-done:
			if got.err != nil || len(got.values) != 1 || got.values[0].Credential.ID != 1 {
				t.Fatalf("unexpected shared result: %+v", got)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("shared load did not complete")
		}
	}
	leader := leaderTiming.Snapshot()
	waiter := waiterTiming.Snapshot()
	if leader.Counters["candidate_loader_executed"] != 1 || leader.Counters["base_loader_executed"] != 1 || leader.Counters["overlay_loader_executed"] != 1 || leader.Stages["base_query"].Count != 1 || leader.Stages["overlay_query"].Count != 1 {
		t.Fatalf("executing caller lost query attribution: %+v", leader)
	}
	if waiter.Stages["candidate_shared_load"].Count != 1 || waiter.Stages["candidate_shared_load"].DurationMS < 15 || waiter.Counters["candidate_loader_executed"] != 0 || waiter.Stages["base_query"].Count != 0 || waiter.Stages["overlay_query"].Count != 0 || waiter.Stages["combined_query"].Count != 0 {
		t.Fatalf("waiter incorrectly owns loader/query timing: %+v", waiter)
	}
	baseCalls, overlayCalls := repo.callCounts("model-a")
	if baseCalls != 1 || overlayCalls != 1 {
		t.Fatalf("shared load queried repository again: base=%d overlay=%d", baseCalls, overlayCalls)
	}
}
