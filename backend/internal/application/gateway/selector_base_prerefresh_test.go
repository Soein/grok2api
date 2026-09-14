package gateway

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/chenyme/grok2api/backend/internal/domain/account"
	"github.com/chenyme/grok2api/backend/internal/pkg/perfmetrics"
	"github.com/chenyme/grok2api/backend/internal/repository"
)

// blockingRoutingLayerRepo wraps layeredAccountRepository and provides fine-grained control
// over base query and overlay query execution and blocking for deterministic concurrency testing.
type blockingRoutingLayerRepo struct {
	*layeredAccountRepository
	mu            sync.Mutex
	blockBase     chan struct{}
	baseStarted   chan struct{}
	baseCalled    int
	customBases   []account.RoutingAccountBase
	returnErr     error
	ignoreContext bool
	baseSleep     time.Duration

	blockOverlay   chan struct{}
	overlayStarted chan struct{}
}

func newBlockingRoutingLayerRepo(bases []account.RoutingAccountBase) *blockingRoutingLayerRepo {
	layered := newLayeredRepositoryFixture()
	if bases != nil {
		layered.bases = bases
	}
	return &blockingRoutingLayerRepo{
		layeredAccountRepository: layered,
		blockBase:                make(chan struct{}),
		baseStarted:              make(chan struct{}, 10),
	}
}

func (r *blockingRoutingLayerRepo) ListRoutingAccountBases(ctx context.Context, provider account.Provider, quotaMode string) ([]account.RoutingAccountBase, error) {
	r.mu.Lock()
	r.baseCalled++
	started := r.baseStarted
	block := r.blockBase
	customBases := r.customBases
	retErr := r.returnErr
	ignoreCtx := r.ignoreContext
	sleep := r.baseSleep
	r.mu.Unlock()

	if sleep > 0 {
		time.Sleep(sleep)
	}

	if started != nil {
		select {
		case started <- struct{}{}:
		default:
		}
	}

	if block != nil {
		if ignoreCtx {
			<-block
		} else {
			select {
			case <-block:
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}
	}

	if retErr != nil {
		return nil, retErr
	}
	if customBases != nil {
		return customBases, nil
	}
	return r.layeredAccountRepository.ListRoutingAccountBases(ctx, provider, quotaMode)
}

func (r *blockingRoutingLayerRepo) ListRoutingAccountOverlays(ctx context.Context, provider account.Provider, modelRouteID uint64, upstreamModel string) (account.RoutingOverlaySnapshot, error) {
	r.mu.Lock()
	started := r.overlayStarted
	block := r.blockOverlay
	r.mu.Unlock()

	if started != nil {
		select {
		case started <- struct{}{}:
		default:
		}
	}

	if block != nil {
		select {
		case <-block:
		case <-ctx.Done():
			return account.RoutingOverlaySnapshot{}, ctx.Err()
		}
	}

	return r.layeredAccountRepository.ListRoutingAccountOverlays(ctx, provider, modelRouteID, upstreamModel)
}

func (r *blockingRoutingLayerRepo) unblock() {
	r.unblockBase()
	r.unblockOverlay()
}

func (r *blockingRoutingLayerRepo) unblockBase() {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.blockBase != nil {
		close(r.blockBase)
		r.blockBase = nil
	}
}

func (r *blockingRoutingLayerRepo) unblockOverlay() {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.blockOverlay != nil {
		close(r.blockOverlay)
		r.blockOverlay = nil
	}
}

func (r *blockingRoutingLayerRepo) getBaseCalled() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.baseCalled
}

// TestSelectorBasePreRefreshWarmHitAndSeparateModelBaseHitWhileBlocked verifies warm cache hits
// during blocked background refresh and independent foreground coalescing upon expiry.
func TestSelectorBasePreRefreshWarmHitAndSeparateModelBaseHitWhileBlocked(t *testing.T) {
	repo := newBlockingRoutingLayerRepo(nil)
	repo.unblockBase()

	selector := NewSelector(repo, nil, nil, nil, time.Hour, time.Second, time.Minute)
	selector.UpdateBuildBasePreRefreshPolicy(true, 5*time.Second, 5*time.Second)

	t0 := time.Now().UTC()
	// Foreground loads candidates for model-a at t0 -> populates base cache expiring at t0 + 30s
	candsA, err := selector.loadCandidates(context.Background(), account.ProviderBuild, 0, "model-a", "", t0)
	if err != nil || len(candsA) != 1 {
		t.Fatalf("initial loadCandidates model-a failed: %v", err)
	}
	if repo.getBaseCalled() != 1 {
		t.Fatalf("expected 1 initial base call, got %d", repo.getBaseCalled())
	}

	// Prepare repo to block on next base query
	repo.mu.Lock()
	repo.blockBase = make(chan struct{})
	repo.baseStarted = make(chan struct{}, 10)
	repo.mu.Unlock()

	// At t0 + 26s (within 5s ahead of expiry t0 + 30s), trigger multiple concurrent BG prerefresh attempts
	tPrerefresh := t0.Add(26 * time.Second)
	const concurrentAttempts = 5
	var bgWg sync.WaitGroup
	bgResults := make(chan bool, concurrentAttempts)
	for i := 0; i < concurrentAttempts; i++ {
		bgWg.Add(1)
		go func() {
			defer bgWg.Done()
			bgResults <- selector.preRefreshBuildBaseAt(context.Background(), tPrerefresh)
		}()
	}

	// Wait until background prerefresh actually starts ListRoutingAccountBases
	select {
	case <-repo.baseStarted:
	case <-time.After(2 * time.Second):
		t.Fatal("background prerefresh did not enter ListRoutingAccountBases")
	}

	// Give concurrent attempts time to attempt acquisition and be rejected by the serial atomic guard
	time.Sleep(20 * time.Millisecond)

	// Foreground candidate hit for model-a must return immediately with warm data
	fgDoneA := make(chan struct{})
	go func() {
		defer close(fgDoneA)
		fgCandidates, fgErr := selector.loadCandidates(context.Background(), account.ProviderBuild, 0, "model-a", "", tPrerefresh)
		if fgErr != nil || len(fgCandidates) != 1 {
			t.Errorf("foreground model-a candidate hit failed: %v", fgErr)
		}
	}()
	select {
	case <-fgDoneA:
	case <-time.After(500 * time.Millisecond):
		t.Fatal("foreground candidate hit blocked on background repo query")
	}

	// Foreground candidate miss for model-b but base hit: must assemble candidate without blocking on repo
	fgDoneB := make(chan struct{})
	go func() {
		defer close(fgDoneB)
		fgCandidates, fgErr := selector.loadCandidates(context.Background(), account.ProviderBuild, 0, "model-b", "", tPrerefresh)
		if fgErr != nil || len(fgCandidates) != 1 {
			t.Errorf("foreground model-b base hit failed: %v", fgErr)
		}
	}()
	select {
	case <-fgDoneB:
	case <-time.After(500 * time.Millisecond):
		t.Fatal("foreground separate model base hit blocked on background repo query")
	}

	// Exact base query count: exactly 1 initial + 1 background query
	if calls := repo.getBaseCalled(); calls != 2 {
		t.Fatalf("expected exactly 2 base calls while BG blocked, got %d", calls)
	}

	// Clean up background
	repo.unblockBase()
	bgWg.Wait()

	// Exactly 1 background attempt published, remaining were rejected by atomic guard
	successCount := 0
	for i := 0; i < concurrentAttempts; i++ {
		if <-bgResults {
			successCount++
		}
	}
	if successCount != 1 {
		t.Fatalf("expected exactly 1 BG attempt to publish, got %d", successCount)
	}

	// Foreground coalescing remains independent at expiry without propagating background errors
	baseKey := routingBaseCacheKey{provider: account.ProviderBuild, quotaMode: ""}
	candKey := candidateCacheKey{provider: account.ProviderBuild, upstreamModel: "model-a"}

	// Set base and candidate expiry to past relative to real clock so inner candidate loader runs
	tExpiredPast := time.Now().UTC().Add(-time.Millisecond)
	selector.candidateMu.Lock()
	bSnap := selector.routingBases[baseKey]
	bSnap.expiresAt = tExpiredPast
	selector.routingBases[baseKey] = bSnap
	delete(selector.candidates, candKey)
	selector.candidateMu.Unlock()

	// Next BG query fails
	repo.mu.Lock()
	repo.returnErr = errors.New("simulated background failure")
	repo.mu.Unlock()

	bgFailed := selector.preRefreshBuildBaseAt(context.Background(), time.Now().UTC())
	if bgFailed {
		t.Fatal("expected background query to fail")
	}

	// Restore repo for foreground
	repo.mu.Lock()
	repo.returnErr = nil
	repo.mu.Unlock()

	// Multiple concurrent foreground requests at expiry coalesce independently via foreground singleflight
	callsBeforeFG := repo.getBaseCalled()
	const fgCoalesceCount = 3
	var fgWg sync.WaitGroup
	fgErrs := make(chan error, fgCoalesceCount)
	for i := 0; i < fgCoalesceCount; i++ {
		fgWg.Add(1)
		go func() {
			defer fgWg.Done()
			_, fErr := selector.loadCandidates(context.Background(), account.ProviderBuild, 0, "model-a", "", time.Now().UTC())
			fgErrs <- fErr
		}()
	}
	fgWg.Wait()
	close(fgErrs)

	for fErr := range fgErrs {
		if fErr != nil {
			t.Fatalf("foreground request at expiry failed: %v (BG error must not become FG error)", fErr)
		}
	}
	if callsAfter := repo.getBaseCalled(); callsAfter != callsBeforeFG+1 {
		t.Fatalf("expected foreground requests to coalesce into exactly 1 base query, got delta %d", callsAfter-callsBeforeFG)
	}
}

// TestSelectorBasePreRefreshCrossingOldExpiryReturnsVisiblyNewContent verifies that crossing
// the original base expiry returns refreshed content without triggering duplicate repository queries.
func TestSelectorBasePreRefreshCrossingOldExpiryReturnsVisiblyNewContent(t *testing.T) {
	repo := newLayeredRepositoryFixture()
	repo.bases = []account.RoutingAccountBase{{
		Credential: account.Credential{ID: 101, Provider: account.ProviderBuild, Enabled: true, AuthStatus: account.AuthStatusActive},
	}}
	repo.overlays = map[string]account.RoutingOverlaySnapshot{
		"model-a": {Values: []account.RoutingAccountOverlay{
			{AccountID: 101, ModelCapabilityKnown: true, SupportsModel: true},
			{AccountID: 202, ModelCapabilityKnown: true, SupportsModel: true},
		}},
	}

	selector := NewSelector(repo, nil, nil, nil, time.Hour, time.Second, time.Minute)
	selector.UpdateBuildBasePreRefreshPolicy(true, 5*time.Second, 5*time.Second)

	now := time.Now().UTC()
	// Foreground load at now
	cands, err := selector.loadCandidates(context.Background(), account.ProviderBuild, 0, "model-a", "", now)
	if err != nil || len(cands) != 1 || cands[0].Credential.ID != 101 {
		t.Fatalf("initial load failed: err=%v, cands=%+v", err, cands)
	}
	if repo.baseCalls != 1 {
		t.Fatalf("expected 1 base call, got %d", repo.baseCalls)
	}

	// Prepare repo to return visibly distinct base on next query
	repo.nextBases = []account.RoutingAccountBase{{
		Credential: account.Credential{ID: 202, Provider: account.ProviderBuild, Enabled: true, AuthStatus: account.AuthStatusActive},
	}}

	// Force actual expired snapshot state using real clock:
	// Set the existing base and candidate snapshots to expire in 40ms from now.
	tExpiry := time.Now().UTC().Add(40 * time.Millisecond)
	baseKey := routingBaseCacheKey{provider: account.ProviderBuild, quotaMode: ""}
	candKey := candidateCacheKey{provider: account.ProviderBuild, upstreamModel: "model-a"}

	selector.candidateMu.Lock()
	baseSnap := selector.routingBases[baseKey]
	baseSnap.expiresAt = tExpiry
	selector.routingBases[baseKey] = baseSnap

	candSnap := selector.candidates[candKey]
	candSnap.expiresAt = tExpiry
	candSnap.lastAccess = time.Now().UTC()
	selector.candidates[candKey] = candSnap
	selector.candidateMu.Unlock()

	// Prerefresh at current time (which is within 5s ahead of tExpiry)
	tPrerefresh := time.Now().UTC()
	if !selector.preRefreshBuildBaseAt(context.Background(), tPrerefresh) {
		t.Fatal("expected prerefresh to return true on success")
	}
	if repo.baseCalls != 2 {
		t.Fatalf("expected 2 base calls after prerefresh, got %d", repo.baseCalls)
	}

	// Wait for real clock to actually cross tExpiry so inner check checkTime := time.Now().UTC()
	// also sees the old candidate as expired
	for time.Now().UTC().Before(tExpiry.Add(5 * time.Millisecond)) {
		time.Sleep(5 * time.Millisecond)
	}

	realTimeAfterExpiry := time.Now().UTC()
	newCands, err := selector.loadCandidates(context.Background(), account.ProviderBuild, 0, "model-a", "", realTimeAfterExpiry)
	if err != nil {
		t.Fatalf("loadCandidates after crossing old expiry failed: %v", err)
	}
	if len(newCands) != 1 {
		t.Fatalf("expected 1 candidate, got %d", len(newCands))
	}
	// Visibly NEW content returned!
	if newCands[0].Credential.ID != 202 {
		t.Fatalf("expected visibly NEW candidate ID 202, got %d", newCands[0].Credential.ID)
	}
	// No extra foreground base query!
	if repo.baseCalls != 2 {
		t.Fatalf("foreground query crossing old expiry triggered duplicate base call, got %d (expected 2)", repo.baseCalls)
	}
}

// TestSelectorBasePreRefreshMultipleModelCandidatesShareOneBaseRefresh verifies that active candidates
// for multiple models share a single underlying base pre-refresh query.
func TestSelectorBasePreRefreshMultipleModelCandidatesShareOneBaseRefresh(t *testing.T) {
	repo := newLayeredRepositoryFixture()
	selector := NewSelector(repo, nil, nil, nil, time.Hour, time.Second, time.Minute)
	selector.UpdateBuildBasePreRefreshPolicy(true, 5*time.Second, 5*time.Second)

	t0 := time.Now().UTC()
	// Load candidate for model-a and model-b
	_, _ = selector.loadCandidates(context.Background(), account.ProviderBuild, 0, "model-a", "", t0)
	_, _ = selector.loadCandidates(context.Background(), account.ProviderBuild, 0, "model-b", "", t0)

	if repo.baseCalls != 1 {
		t.Fatalf("expected 1 shared base call for multiple models, got %d", repo.baseCalls)
	}

	// Prerefresh at t0 + 26s
	refreshed := selector.preRefreshBuildBaseAt(context.Background(), t0.Add(26*time.Second))
	if !refreshed {
		t.Fatal("expected prerefresh to succeed")
	}
	// Prerefresh queries base only ONCE for both model candidates
	if repo.baseCalls != 2 {
		t.Fatalf("expected exactly 2 base calls, got %d", repo.baseCalls)
	}
}

// TestSelectorBasePreRefreshEarlyColdInactiveDisabledOtherDoNotQuery verifies that inactive, early,
// disabled, or mismatched selector states do not trigger pre-refresh queries.
func TestSelectorBasePreRefreshEarlyColdInactiveDisabledOtherDoNotQuery(t *testing.T) {
	t.Run("cold startup does not query", func(t *testing.T) {
		repo := newLayeredRepositoryFixture()
		selector := NewSelector(repo, nil, nil, nil, time.Hour, time.Second, time.Minute)
		selector.UpdateBuildBasePreRefreshPolicy(true, 5*time.Second, 5*time.Second)

		if selector.preRefreshBuildBaseAt(context.Background(), time.Now().UTC()) {
			t.Fatal("cold selector should not prerefresh")
		}
		if repo.baseCalls != 0 {
			t.Fatalf("cold scan made %d calls", repo.baseCalls)
		}
	})

	t.Run("early does not query", func(t *testing.T) {
		repo := newLayeredRepositoryFixture()
		selector := NewSelector(repo, nil, nil, nil, time.Hour, time.Second, time.Minute)
		selector.UpdateBuildBasePreRefreshPolicy(true, 5*time.Second, 5*time.Second)

		t0 := time.Now().UTC()
		_, _ = selector.loadCandidates(context.Background(), account.ProviderBuild, 0, "model-a", "", t0)
		initialCalls := repo.baseCalls

		// At t0 + 10s, remaining TTL is 20s > 5s ahead
		if selector.preRefreshBuildBaseAt(context.Background(), t0.Add(10*time.Second)) {
			t.Fatal("early tick should not prerefresh")
		}
		if repo.baseCalls != initialCalls {
			t.Fatal("early tick triggered base query")
		}
	})

	t.Run("inactive candidates do not query", func(t *testing.T) {
		repo := newLayeredRepositoryFixture()
		selector := NewSelector(repo, nil, nil, nil, time.Hour, time.Second, time.Minute)
		selector.UpdateBuildBasePreRefreshPolicy(true, 5*time.Second, 5*time.Second)

		t0 := time.Now().UTC()
		_, _ = selector.loadCandidates(context.Background(), account.ProviderBuild, 0, "model-a", "", t0)
		initialCalls := repo.baseCalls

		// At t0 + 35s, candidates have not been accessed within candidateCacheTTL (30s)
		if selector.preRefreshBuildBaseAt(context.Background(), t0.Add(35*time.Second)) {
			t.Fatal("inactive tick should not prerefresh")
		}
		if repo.baseCalls != initialCalls {
			t.Fatal("inactive tick triggered base query")
		}
	})

	t.Run("disabled policy does not query", func(t *testing.T) {
		repo := newLayeredRepositoryFixture()
		selector := NewSelector(repo, nil, nil, nil, time.Hour, time.Second, time.Minute)
		selector.UpdateBuildBasePreRefreshPolicy(false, 5*time.Second, 5*time.Second)

		t0 := time.Now().UTC()
		_, _ = selector.loadCandidates(context.Background(), account.ProviderBuild, 0, "model-a", "", t0)
		initialCalls := repo.baseCalls

		if selector.preRefreshBuildBaseAt(context.Background(), t0.Add(26*time.Second)) {
			t.Fatal("disabled policy should not prerefresh")
		}
		if repo.baseCalls != initialCalls {
			t.Fatal("disabled policy triggered base query")
		}
	})

	t.Run("nonmatching provider does not query", func(t *testing.T) {
		repo := newLayeredRepositoryFixture()
		selector := NewSelector(repo, nil, nil, nil, time.Hour, time.Second, time.Minute)
		selector.UpdateBuildBasePreRefreshPolicy(true, 5*time.Second, 5*time.Second)

		t0 := time.Now().UTC()
		_, _ = selector.loadCandidates(context.Background(), account.ProviderWeb, 0, "model-a", "", t0)
		initialCalls := repo.baseCalls

		if selector.preRefreshBuildBaseAt(context.Background(), t0.Add(26*time.Second)) {
			t.Fatal("nonmatching provider should not prerefresh")
		}
		if repo.baseCalls != initialCalls {
			t.Fatal("nonmatching provider triggered base query")
		}
	})

	t.Run("nonmatching quota mode does not query", func(t *testing.T) {
		repo := newLayeredRepositoryFixture()
		selector := NewSelector(repo, nil, nil, nil, time.Hour, time.Second, time.Minute)
		selector.UpdateBuildBasePreRefreshPolicy(true, 5*time.Second, 5*time.Second)

		t0 := time.Now().UTC()
		_, _ = selector.loadCandidates(context.Background(), account.ProviderBuild, 0, "model-a", "custom-quota", t0)
		initialCalls := repo.baseCalls

		if selector.preRefreshBuildBaseAt(context.Background(), t0.Add(26*time.Second)) {
			t.Fatal("nonmatching quota mode should not prerefresh")
		}
		if repo.baseCalls != initialCalls {
			t.Fatal("nonmatching quota mode triggered base query")
		}
	})

	t.Run("background does not maintain heat", func(t *testing.T) {
		repo := newLayeredRepositoryFixture()
		selector := NewSelector(repo, nil, nil, nil, time.Hour, time.Second, time.Minute)
		selector.UpdateBuildBasePreRefreshPolicy(true, 5*time.Second, 5*time.Second)

		t0 := time.Now().UTC()
		_, _ = selector.loadCandidates(context.Background(), account.ProviderBuild, 0, "model-a", "", t0)

		// Prerefresh at t0 + 26s
		refreshed := selector.preRefreshBuildBaseAt(context.Background(), t0.Add(26*time.Second))
		if !refreshed {
			t.Fatal("prerefresh should succeed")
		}

		// In the next cycle, if no foreground requests arrive:
		// Candidate lastAccess is still t0 (background prerefresh did NOT update candidate heat).
		// When t > t0 + 30s, heat is gone.
		tCold := t0.Add(31 * time.Second)
		key := routingBaseCacheKey{provider: account.ProviderBuild, quotaMode: ""}
		selector.candidateMu.Lock()
		snap := selector.routingBases[key]
		snap.expiresAt = tCold.Add(4 * time.Second)
		selector.routingBases[key] = snap
		selector.candidateMu.Unlock()

		callsBefore := repo.baseCalls
		if selector.preRefreshBuildBaseAt(context.Background(), tCold) {
			t.Fatal("cold candidates should not trigger prerefresh")
		}
		if repo.baseCalls != callsBefore {
			t.Fatal("background should not keep heat alive on its own")
		}
	})
}

// TestSelectorBasePreRefreshFailureLeavesExpiryAndBoundedRetry verifies that failed refreshes
// retain existing expiry and calculate retry bounds from query completion rather than start.
func TestSelectorBasePreRefreshFailureLeavesExpiryAndBoundedRetry(t *testing.T) {
	repo := newBlockingRoutingLayerRepo(nil)
	repo.unblockBase()
	defer repo.unblock()
	selector := NewSelector(repo, nil, nil, nil, time.Hour, time.Second, time.Minute)
	selector.UpdateBuildBasePreRefreshPolicy(true, 5*time.Second, 5*time.Second)

	t0 := time.Now().UTC()
	_, _ = selector.loadCandidates(context.Background(), account.ProviderBuild, 0, "model-a", "", t0)

	key := routingBaseCacheKey{provider: account.ProviderBuild, quotaMode: ""}
	candKey := candidateCacheKey{provider: account.ProviderBuild, upstreamModel: "model-a"}

	now := time.Now().UTC()
	actualExpiry := now.Add(4 * time.Second)

	selector.candidateMu.Lock()
	baseSnap := selector.routingBases[key]
	baseSnap.expiresAt = actualExpiry
	selector.routingBases[key] = baseSnap

	candSnap := selector.candidates[candKey]
	candSnap.lastAccess = now
	selector.candidates[candKey] = candSnap
	selector.candidateMu.Unlock()

	repo.mu.Lock()
	repo.returnErr = errors.New("transient database error")
	repo.baseSleep = 150 * time.Millisecond
	repo.mu.Unlock()

	refreshed := selector.preRefreshBuildBaseAt(context.Background(), now)
	failureCompletion := time.Now().UTC()
	if refreshed {
		t.Fatal("expected prerefresh to return false on error")
	}

	selector.candidateMu.Lock()
	currentExpiry := selector.routingBases[key].expiresAt
	retryUntil := selector.buildBasePreRefreshRetryUntil
	selector.candidateMu.Unlock()

	if !currentExpiry.Equal(actualExpiry) {
		t.Fatalf("expiry was changed on failure: orig %v, current %v", actualExpiry, currentExpiry)
	}

	minRetry := failureCompletion.Add(candidateCacheRetryTTL - 50*time.Millisecond)
	maxRetry := failureCompletion.Add(candidateCacheRetryTTL + 150*time.Millisecond)
	if retryUntil.Before(minRetry) || retryUntil.After(maxRetry) {
		t.Fatalf("retryUntil %v is outside expected bounds [%v, %v] near failureCompletion + 5s", retryUntil, minRetry, maxRetry)
	}

	retryNow := time.Now().UTC()
	callsBefore := repo.getBaseCalled()
	if selector.preRefreshBuildBaseAt(context.Background(), retryNow) {
		t.Fatal("immediate retry should be suppressed by retryUntil")
	}
	if callsAfter := repo.getBaseCalled(); callsAfter != callsBefore {
		t.Fatalf("immediate retry made an unexpected query to repo: calls before %d, after %d", callsBefore, callsAfter)
	}
}

// TestSelectorBasePreRefreshPublicationPreemption verifies preemption and discard semantics
// across invalidations, evictions, generation bumps, and config updates.
func TestSelectorBasePreRefreshPublicationPreemption(t *testing.T) {
	t.Run("invalidation during query discards publication", func(t *testing.T) {
		repo := newBlockingRoutingLayerRepo(nil)
		repo.unblockBase()
		defer repo.unblock()
		selector := NewSelector(repo, nil, nil, nil, time.Hour, time.Second, time.Minute)
		selector.UpdateBuildBasePreRefreshPolicy(true, 5*time.Second, 5*time.Second)

		t0 := time.Now().UTC()
		_, _ = selector.loadCandidates(context.Background(), account.ProviderBuild, 0, "model-a", "", t0)

		repo.mu.Lock()
		repo.blockBase = make(chan struct{})
		repo.baseStarted = make(chan struct{}, 1)
		repo.mu.Unlock()
		defer repo.unblockBase()

		done := make(chan bool, 1)
		go func() {
			done <- selector.preRefreshBuildBaseAt(context.Background(), t0.Add(26*time.Second))
		}()

		select {
		case <-repo.baseStarted:
		case <-time.After(2 * time.Second):
			t.Fatal("repo did not start")
		}

		// Invalidation arrives for Build base while query is in-flight
		selector.ApplyInvalidation(repository.InvalidationEvent{
			Kind:     repository.InvalidationAccountStateChanged,
			Provider: account.ProviderBuild,
		})

		repo.unblockBase()
		select {
		case published := <-done:
			if published {
				t.Fatal("publication should be discarded after invalidation")
			}
		case <-time.After(2 * time.Second):
			t.Fatal("BG did not return")
		}

		baseKey := routingBaseCacheKey{provider: account.ProviderBuild, quotaMode: ""}
		selector.candidateMu.Lock()
		_, exists := selector.routingBases[baseKey]
		selector.candidateMu.Unlock()
		if exists {
			t.Fatal("invalidation must leave target absent from base cache")
		}
	})

	t.Run("eviction during query discards publication and does not resurrect stale", func(t *testing.T) {
		repo := newBlockingRoutingLayerRepo(nil)
		repo.unblockBase()
		selector := NewSelector(repo, nil, nil, nil, time.Hour, time.Second, time.Minute)
		selector.UpdateBuildBasePreRefreshPolicy(true, 5*time.Second, 5*time.Second)

		t0 := time.Now().UTC()
		_, _ = selector.loadCandidates(context.Background(), account.ProviderBuild, 0, "model-a", "", t0)

		repo.mu.Lock()
		repo.blockBase = make(chan struct{})
		repo.baseStarted = make(chan struct{}, 1)
		repo.mu.Unlock()

		done := make(chan bool, 1)
		go func() {
			done <- selector.preRefreshBuildBaseAt(context.Background(), t0.Add(26*time.Second))
		}()

		select {
		case <-repo.baseStarted:
		case <-time.After(2 * time.Second):
			t.Fatal("repo did not start")
		}

		// Evict entry from routingBases
		baseKey := routingBaseCacheKey{provider: account.ProviderBuild, quotaMode: ""}
		selector.candidateMu.Lock()
		delete(selector.routingBases, baseKey)
		selector.candidateMu.Unlock()

		repo.unblockBase()
		select {
		case published := <-done:
			if published {
				t.Fatal("publication should be discarded after eviction")
			}
		case <-time.After(2 * time.Second):
			t.Fatal("BG did not return")
		}

		selector.candidateMu.Lock()
		_, exists := selector.routingBases[baseKey]
		selector.candidateMu.Unlock()
		if exists {
			t.Fatal("evicted snapshot must not be resurrected by completed query")
		}
	})

	t.Run("newer store generation with same expiry discards publication", func(t *testing.T) {
		repo := newBlockingRoutingLayerRepo(nil)
		repo.unblockBase()
		selector := NewSelector(repo, nil, nil, nil, time.Hour, time.Second, time.Minute)
		selector.UpdateBuildBasePreRefreshPolicy(true, 5*time.Second, 5*time.Second)

		t0 := time.Now().UTC()
		_, _ = selector.loadCandidates(context.Background(), account.ProviderBuild, 0, "model-a", "", t0)

		repo.mu.Lock()
		repo.blockBase = make(chan struct{})
		repo.baseStarted = make(chan struct{}, 1)
		repo.mu.Unlock()

		done := make(chan bool, 1)
		go func() {
			done <- selector.preRefreshBuildBaseAt(context.Background(), t0.Add(26*time.Second))
		}()

		select {
		case <-repo.baseStarted:
		case <-time.After(2 * time.Second):
			t.Fatal("repo did not start")
		}

		// Update snapshot with same expiry but new store generation
		key := routingBaseCacheKey{provider: account.ProviderBuild, quotaMode: ""}
		selector.candidateMu.Lock()
		snap := selector.routingBases[key]
		selector.storeRoutingBaseSnapshotLockedWithAccess(key, snap, snap.lastAccess, time.Now().UTC())
		newGen := selector.routingBases[key].generation
		selector.candidateMu.Unlock()

		repo.unblockBase()
		select {
		case published := <-done:
			if published {
				t.Fatal("publication should be discarded when snapshot generation was bumped")
			}
		case <-time.After(2 * time.Second):
			t.Fatal("BG did not return")
		}

		selector.candidateMu.Lock()
		currentGen := selector.routingBases[key].generation
		selector.candidateMu.Unlock()
		if currentGen != newGen {
			t.Fatalf("expected generation %d, got %d", newGen, currentGen)
		}
	})

	t.Run("disable then reenable policy during query discards publication", func(t *testing.T) {
		repo := newBlockingRoutingLayerRepo(nil)
		repo.unblockBase()
		defer repo.unblock()
		selector := NewSelector(repo, nil, nil, nil, time.Hour, time.Second, time.Minute)
		selector.UpdateBuildBasePreRefreshPolicy(true, 5*time.Second, 5*time.Second)

		t0 := time.Now().UTC()
		_, _ = selector.loadCandidates(context.Background(), account.ProviderBuild, 0, "model-a", "", t0)

		baseKey := routingBaseCacheKey{provider: account.ProviderBuild, quotaMode: ""}
		selector.candidateMu.Lock()
		initialSnap := selector.routingBases[baseKey]
		selector.candidateMu.Unlock()

		repo.mu.Lock()
		repo.blockBase = make(chan struct{})
		repo.baseStarted = make(chan struct{}, 1)
		repo.mu.Unlock()
		defer repo.unblockBase()

		done := make(chan bool, 1)
		go func() {
			done <- selector.preRefreshBuildBaseAt(context.Background(), t0.Add(26*time.Second))
		}()

		select {
		case <-repo.baseStarted:
		case <-time.After(2 * time.Second):
			t.Fatal("repo did not start")
		}

		// Disable then re-enable policy -> increments policy generation
		selector.UpdateBuildBasePreRefreshPolicy(false, 5*time.Second, 5*time.Second)
		selector.UpdateBuildBasePreRefreshPolicy(true, 5*time.Second, 5*time.Second)

		repo.unblockBase()
		select {
		case published := <-done:
			if published {
				t.Fatal("publication should be discarded after policy epoch change")
			}
		case <-time.After(2 * time.Second):
			t.Fatal("BG did not return")
		}

		selector.candidateMu.Lock()
		currentSnap := selector.routingBases[baseKey]
		selector.candidateMu.Unlock()
		if currentSnap.generation != initialSnap.generation {
			t.Fatalf("expected generation %d to remain unchanged, got %d", initialSnap.generation, currentSnap.generation)
		}
		if !currentSnap.expiresAt.Equal(initialSnap.expiresAt) {
			t.Fatalf("expected expiry %v to remain unchanged, got %v", initialSnap.expiresAt, currentSnap.expiresAt)
		}
	})

	t.Run("same-value config update does not discard valid publication", func(t *testing.T) {
		repo := newBlockingRoutingLayerRepo(nil)
		repo.unblockBase()
		selector := NewSelector(repo, nil, nil, nil, time.Hour, time.Second, time.Minute)
		selector.UpdateBuildBasePreRefreshPolicy(true, 5*time.Second, 5*time.Second)

		t0 := time.Now().UTC()
		_, _ = selector.loadCandidates(context.Background(), account.ProviderBuild, 0, "model-a", "", t0)

		repo.mu.Lock()
		repo.blockBase = make(chan struct{})
		repo.baseStarted = make(chan struct{}, 1)
		repo.mu.Unlock()

		done := make(chan bool, 1)
		go func() {
			done <- selector.preRefreshBuildBaseAt(context.Background(), t0.Add(26*time.Second))
		}()

		select {
		case <-repo.baseStarted:
		case <-time.After(2 * time.Second):
			t.Fatal("repo did not start")
		}

		// Same-value config update -> does NOT bump policy generation
		selector.UpdateBuildBasePreRefreshPolicy(true, 5*time.Second, 5*time.Second)

		repo.unblockBase()
		select {
		case published := <-done:
			if !published {
				t.Fatal("same-value config update should NOT discard valid publication")
			}
		case <-time.After(2 * time.Second):
			t.Fatal("BG did not return")
		}
	})

	t.Run("foreground lastAccess hits do not discard valid publication", func(t *testing.T) {
		repo := newBlockingRoutingLayerRepo(nil)
		repo.unblockBase()
		selector := NewSelector(repo, nil, nil, nil, time.Hour, time.Second, time.Minute)
		selector.UpdateBuildBasePreRefreshPolicy(true, 5*time.Second, 5*time.Second)

		t0 := time.Now().UTC()
		_, _ = selector.loadCandidates(context.Background(), account.ProviderBuild, 0, "model-a", "", t0)

		repo.mu.Lock()
		repo.blockBase = make(chan struct{})
		repo.baseStarted = make(chan struct{}, 1)
		repo.mu.Unlock()

		done := make(chan bool, 1)
		go func() {
			done <- selector.preRefreshBuildBaseAt(context.Background(), t0.Add(26*time.Second))
		}()

		select {
		case <-repo.baseStarted:
		case <-time.After(2 * time.Second):
			t.Fatal("repo did not start")
		}

		// Foreground hit updates lastAccess in place without bumping generation
		tHit := t0.Add(27 * time.Second)
		key := routingBaseCacheKey{provider: account.ProviderBuild, quotaMode: ""}
		selector.candidateMu.Lock()
		snap := selector.routingBases[key]
		snap.lastAccess = tHit
		selector.routingBases[key] = snap
		selector.candidateMu.Unlock()

		repo.unblockBase()
		select {
		case published := <-done:
			if !published {
				t.Fatal("foreground lastAccess hit should NOT discard valid publication")
			}
		case <-time.After(2 * time.Second):
			t.Fatal("BG did not return")
		}

		// Preserves FG lastAccess
		selector.candidateMu.Lock()
		finalAccess := selector.routingBases[key].lastAccess
		selector.candidateMu.Unlock()
		if !finalAccess.Equal(tHit) {
			t.Fatalf("expected lastAccess %v to be preserved, got %v", tHit, finalAccess)
		}
	})
}

// TestSelectorBasePreRefreshCancellationAndContextRejection verifies cancellation semantics,
// timeout handling, and foreground query independence.
func TestSelectorBasePreRefreshCancellationAndContextRejection(t *testing.T) {
	t.Run("fake repo ignores context and returns success after timeout", func(t *testing.T) {
		repo := newBlockingRoutingLayerRepo(nil)
		repo.unblockBase()
		defer repo.unblock()
		selector := NewSelector(repo, nil, nil, nil, time.Hour, time.Second, time.Minute)
		selector.UpdateBuildBasePreRefreshPolicy(true, 5*time.Second, 50*time.Millisecond) // 50ms timeout

		t0 := time.Now().UTC()
		_, _ = selector.loadCandidates(context.Background(), account.ProviderBuild, 0, "model-a", "", t0)

		baseKey := routingBaseCacheKey{provider: account.ProviderBuild, quotaMode: ""}
		selector.candidateMu.Lock()
		initialSnap := selector.routingBases[baseKey]
		selector.candidateMu.Unlock()

		repo.mu.Lock()
		repo.blockBase = make(chan struct{})
		repo.baseStarted = make(chan struct{}, 1)
		repo.ignoreContext = true // repo ignores ctx.Done()
		repo.mu.Unlock()
		defer repo.unblockBase()

		done := make(chan bool, 1)
		go func() {
			done <- selector.preRefreshBuildBaseAt(context.Background(), t0.Add(26*time.Second))
		}()

		select {
		case <-repo.baseStarted:
		case <-time.After(2 * time.Second):
			t.Fatal("repo did not start")
		}

		// Wait past 50ms timeout
		time.Sleep(80 * time.Millisecond)
		// Repo unblocks and returns success after timeout
		repo.unblockBase()

		select {
		case published := <-done:
			if published {
				t.Fatal("timed out query must not publish even if fake repo returned success")
			}
		case <-time.After(2 * time.Second):
			t.Fatal("query did not return")
		}

		selector.candidateMu.Lock()
		currentSnap := selector.routingBases[baseKey]
		selector.candidateMu.Unlock()
		if currentSnap.generation != initialSnap.generation {
			t.Fatalf("expected generation %d to remain unchanged, got %d", initialSnap.generation, currentSnap.generation)
		}
		if !currentSnap.expiresAt.Equal(initialSnap.expiresAt) {
			t.Fatalf("expected expiry %v to remain unchanged, got %v", initialSnap.expiresAt, currentSnap.expiresAt)
		}
	})

	t.Run("unexpected quota window rejected", func(t *testing.T) {
		repo := newBlockingRoutingLayerRepo(nil)
		repo.unblockBase()
		defer repo.unblock()
		selector := NewSelector(repo, nil, nil, nil, time.Hour, time.Second, time.Minute)
		selector.UpdateBuildBasePreRefreshPolicy(true, 5*time.Second, 5*time.Second)

		t0 := time.Now().UTC()
		_, _ = selector.loadCandidates(context.Background(), account.ProviderBuild, 0, "model-a", "", t0)

		baseKey := routingBaseCacheKey{provider: account.ProviderBuild, quotaMode: ""}
		selector.candidateMu.Lock()
		initialSnap := selector.routingBases[baseKey]
		selector.candidateMu.Unlock()

		repo.mu.Lock()
		repo.customBases = []account.RoutingAccountBase{
			{
				Credential:  account.Credential{ID: 1, Provider: account.ProviderBuild, Enabled: true, AuthStatus: account.AuthStatusActive},
				QuotaWindow: &account.QuotaWindow{Mode: "daily", Remaining: 100},
			},
		}
		repo.mu.Unlock()

		if selector.preRefreshBuildBaseAt(context.Background(), t0.Add(26*time.Second)) {
			t.Fatal("prerefresh should reject unexpected QuotaWindow")
		}

		selector.candidateMu.Lock()
		currentSnap := selector.routingBases[baseKey]
		selector.candidateMu.Unlock()
		if currentSnap.generation != initialSnap.generation {
			t.Fatalf("expected generation %d to remain unchanged, got %d", initialSnap.generation, currentSnap.generation)
		}
		if !currentSnap.expiresAt.Equal(initialSnap.expiresAt) {
			t.Fatalf("expected expiry %v to remain unchanged, got %v", initialSnap.expiresAt, currentSnap.expiresAt)
		}
	})

	t.Run("active worker cancellation with repo still blocked terminates via repo context", func(t *testing.T) {
		repo := newBlockingRoutingLayerRepo(nil)
		repo.unblockBase()
		selector := NewSelector(repo, nil, nil, nil, time.Hour, time.Second, time.Minute)
		selector.UpdateBuildBasePreRefreshPolicy(true, 5*time.Second, 5*time.Second)

		t0 := time.Now().UTC()
		_, _ = selector.loadCandidates(context.Background(), account.ProviderBuild, 0, "model-a", "", t0)

		// Set base expiry within ahead
		key := routingBaseCacheKey{provider: account.ProviderBuild, quotaMode: ""}
		selector.candidateMu.Lock()
		snap := selector.routingBases[key]
		snap.expiresAt = time.Now().UTC().Add(3 * time.Second)
		selector.routingBases[key] = snap
		selector.candidateMu.Unlock()

		// Prepare repo to block on base query
		repo.mu.Lock()
		repo.blockBase = make(chan struct{})
		repo.baseStarted = make(chan struct{}, 1)
		repo.ignoreContext = false
		repo.mu.Unlock()

		workerCtx, workerCancel := context.WithCancel(context.Background())
		workerErr := make(chan error, 1)
		go func() {
			workerErr <- selector.RunBasePreRefresh(workerCtx)
		}()

		// Wait for the background worker to tick (1s ticker) and enter ListRoutingAccountBases
		select {
		case <-repo.baseStarted:
		case <-time.After(3 * time.Second):
			workerCancel()
			t.Fatal("worker did not enter ListRoutingAccountBases within timeout")
		}

		// Cancel worker context while repo is STILL blocked on blockBase
		// (Do NOT call repo.unblockBase() first!)
		workerCancel()

		select {
		case err := <-workerErr:
			if !errors.Is(err, context.Canceled) {
				t.Fatalf("expected context.Canceled, got %v", err)
			}
		case <-time.After(2 * time.Second):
			t.Fatal("worker did not terminate via repo context cancellation")
		}
	})

	t.Run("background cancellation does not cancel independent foreground query", func(t *testing.T) {
		repo := newBlockingRoutingLayerRepo(nil)
		repo.unblockBase()
		defer repo.unblock()

		selector := NewSelector(repo, nil, nil, nil, time.Hour, time.Second, time.Minute)
		selector.UpdateBuildBasePreRefreshPolicy(true, 5*time.Second, 5*time.Second)

		t0 := time.Now().UTC()
		initialCands, err := selector.loadCandidates(context.Background(), account.ProviderBuild, 0, "model-a", "", t0)
		if err != nil || len(initialCands) != 1 {
			t.Fatalf("initial loadCandidates failed: err=%v, cands=%+v", err, initialCands)
		}
		if repo.getBaseCalled() != 1 {
			t.Fatalf("expected 1 initial base call, got %d", repo.getBaseCalled())
		}

		repo.mu.Lock()
		repo.blockBase = make(chan struct{})
		repo.baseStarted = make(chan struct{}, 10)
		repo.mu.Unlock()
		defer repo.unblockBase()

		bgCtx, bgCancel := context.WithCancel(context.Background())
		defer bgCancel()
		bgDone := make(chan bool, 1)
		go func() {
			bgDone <- selector.preRefreshBuildBaseAt(bgCtx, t0.Add(26*time.Second))
		}()

		select {
		case <-repo.baseStarted:
		case <-time.After(2 * time.Second):
			t.Fatal("bg query did not enter repo")
		}

		baseKey := routingBaseCacheKey{provider: account.ProviderBuild, quotaMode: ""}
		candKey := candidateCacheKey{provider: account.ProviderBuild, upstreamModel: "model-a"}
		selector.candidateMu.Lock()
		baseSnap := selector.routingBases[baseKey]
		baseSnap.expiresAt = time.Now().UTC().Add(-time.Second)
		selector.routingBases[baseKey] = baseSnap
		delete(selector.candidates, candKey)
		selector.candidateMu.Unlock()

		type fgResult struct {
			cands []account.RoutingCandidate
			err   error
		}
		fgCtx, fgCancel := context.WithCancel(context.Background())
		defer fgCancel()
		fgDone := make(chan fgResult, 1)
		go func() {
			cands, fgErr := selector.loadCandidates(fgCtx, account.ProviderBuild, 0, "model-a", "", time.Now().UTC())
			fgDone <- fgResult{cands: cands, err: fgErr}
		}()

		select {
		case <-repo.baseStarted:
		case <-time.After(2 * time.Second):
			t.Fatal("fg query did not enter repo")
		}

		if calls := repo.getBaseCalled(); calls != 3 {
			t.Fatalf("expected exact base calls initial+BG+FG=3, got %d", calls)
		}

		bgCancel()

		select {
		case published := <-bgDone:
			if published {
				t.Fatal("canceled BG should not publish")
			}
		case <-time.After(2 * time.Second):
			t.Fatal("BG did not terminate after cancel")
		}

		select {
		case res := <-fgDone:
			t.Fatalf("foreground query completed prematurely while repo blocked: %+v", res)
		case <-time.After(50 * time.Millisecond):
		}

		repo.unblockBase()

		select {
		case res := <-fgDone:
			if res.err != nil {
				t.Fatalf("foreground query failed: %v", res.err)
			}
			if len(res.cands) != 1 {
				t.Fatalf("expected exactly 1 candidate, got %d", len(res.cands))
			}
		case <-time.After(2 * time.Second):
			t.Fatal("foreground query timed out after unblock")
		}
	})
}

// TestSelectorBasePreRefreshPreservesQuotaConsumed verifies that quota consumption deltas
// across providers and modes are preserved across pre-refresh cycles.
func TestSelectorBasePreRefreshPreservesQuotaConsumed(t *testing.T) {
	repo := newLayeredRepositoryFixture()
	selector := NewSelector(repo, nil, nil, nil, time.Hour, time.Second, time.Minute)
	selector.UpdateBuildBasePreRefreshPolicy(true, 5*time.Second, 5*time.Second)

	t0 := time.Now().UTC()
	_, _ = selector.loadCandidates(context.Background(), account.ProviderBuild, 0, "model-a", "", t0)

	// Accumulate quota consumption deltas across multiple providers and modes
	selector.ConsumeQuota(account.ProviderBuild, 1, "standard", 5)
	selector.ConsumeQuota(account.ProviderBuild, 1, "reasoning", 2)
	selector.ConsumeQuota(account.ProviderWeb, 2, "standard", 3)
	selector.ConsumeQuota(account.ProviderConsole, 3, "console_image", 4)

	// Prerefresh Build base
	refreshed := selector.preRefreshBuildBaseAt(context.Background(), t0.Add(26*time.Second))
	if !refreshed {
		t.Fatal("prerefresh should succeed")
	}

	// Verify consumption deltas for all providers and modes are NOT cleared
	selector.quotaMu.RLock()
	buildStd := selector.quotaConsumed[quotaConsumptionKey{provider: account.ProviderBuild, accountID: 1, mode: "standard"}]
	buildReason := selector.quotaConsumed[quotaConsumptionKey{provider: account.ProviderBuild, accountID: 1, mode: "reasoning"}]
	webStd := selector.quotaConsumed[quotaConsumptionKey{provider: account.ProviderWeb, accountID: 2, mode: "standard"}]
	consoleImg := selector.quotaConsumed[quotaConsumptionKey{provider: account.ProviderConsole, accountID: 3, mode: "console_image"}]
	selector.quotaMu.RUnlock()

	if buildStd != 5 {
		t.Errorf("expected build standard quota delta 5, got %d", buildStd)
	}
	if buildReason != 2 {
		t.Errorf("expected build reasoning quota delta 2, got %d", buildReason)
	}
	if webStd != 3 {
		t.Errorf("expected web standard quota delta 3, got %d", webStd)
	}
	if consoleImg != 4 {
		t.Errorf("expected console image quota delta 4, got %d", consoleImg)
	}
}

// TestSelectorBasePreRefreshForegroundCapturesOldBaseHoldsOverlayUntilBGPublishes verifies
// that foreground assembly retains the expiry of the base snapshot it captured.
func TestSelectorBasePreRefreshForegroundCapturesOldBaseHoldsOverlayUntilBGPublishes(t *testing.T) {
	repo := newBlockingRoutingLayerRepo(nil)
	repo.unblockBase()

	selector := NewSelector(repo, nil, nil, nil, time.Hour, time.Second, time.Minute)
	selector.UpdateBuildBasePreRefreshPolicy(true, 5*time.Second, 5*time.Second)

	t0 := time.Now().UTC()
	oldBaseExpiry := t0.Add(10 * time.Second)

	// Store initial base snapshot expiring at t0 + 10s with credential ID 10
	baseKey := routingBaseCacheKey{provider: account.ProviderBuild, quotaMode: ""}
	baseSnap := routingBaseSnapshot{
		values: []account.RoutingAccountBase{
			{Credential: account.Credential{ID: 10, Provider: account.ProviderBuild, Enabled: true, AuthStatus: account.AuthStatusActive}},
		},
		version:    selector.routingBaseVersion(account.ProviderBuild),
		expiresAt:  oldBaseExpiry,
		staleUntil: oldBaseExpiry.Add(time.Minute),
		lastAccess: t0,
	}
	selector.candidateMu.Lock()
	selector.storeRoutingBaseSnapshotLockedWithAccess(baseKey, baseSnap, t0, t0)

	// Dummy candidate so hasRecentCandidate check passes for prerefresh
	dummyCandKey := candidateCacheKey{provider: account.ProviderBuild, upstreamModel: "dummy-active"}
	selector.candidates[dummyCandKey] = candidateSnapshot{
		values:     []account.RoutingCandidate{{Credential: account.Credential{ID: 10, Provider: account.ProviderBuild}}},
		expiresAt:  t0.Add(30 * time.Second),
		lastAccess: t0,
	}
	// model-a candidate is NOT cached, so foreground must assemble
	candKey := candidateCacheKey{provider: account.ProviderBuild, upstreamModel: "model-a"}
	delete(selector.candidates, candKey)
	selector.candidateMu.Unlock()

	// Configure repo: hold overlay when foreground loads model-a overlay, and next base returns ID 99
	repo.mu.Lock()
	repo.blockOverlay = make(chan struct{})
	repo.overlayStarted = make(chan struct{}, 1)
	repo.customBases = []account.RoutingAccountBase{{
		Credential: account.Credential{ID: 99, Provider: account.ProviderBuild, Enabled: true, AuthStatus: account.AuthStatusActive},
	}}
	repo.mu.Unlock()
	defer repo.unblockOverlay()

	// Foreground starts candidate load for model-a
	fgDone := make(chan []account.RoutingCandidate, 1)
	go func() {
		cands, err := selector.loadCandidates(context.Background(), account.ProviderBuild, 0, "model-a", "", t0)
		if err != nil {
			t.Errorf("foreground loadCandidates failed: %v", err)
		}
		fgDone <- cands
	}()

	// Wait until foreground has loaded OLD base and entered ListRoutingAccountOverlays
	select {
	case <-repo.overlayStarted:
	case <-time.After(2 * time.Second):
		t.Fatal("foreground did not enter ListRoutingAccountOverlays")
	}

	// While foreground is holding overlay, background prerefresh runs at t0 + 6s (remaining 4s <= ahead 5s)
	tPrerefresh := t0.Add(6 * time.Second)
	refreshed := selector.preRefreshBuildBaseAt(context.Background(), tPrerefresh)
	if !refreshed {
		t.Fatal("prerefresh should succeed")
	}

	// Base snapshot in cache is now the NEW base with ID 99 and newer expiry
	selector.candidateMu.Lock()
	newSnap := selector.routingBases[baseKey]
	selector.candidateMu.Unlock()
	if len(newSnap.values) == 0 || newSnap.values[0].Credential.ID != 99 {
		t.Fatalf("expected new base snapshot to have ID 99, got %+v", newSnap.values)
	}
	if !newSnap.expiresAt.After(oldBaseExpiry) {
		t.Fatalf("expected new base expiry %v to be after old base expiry %v", newSnap.expiresAt, oldBaseExpiry)
	}

	// Release overlay for foreground
	repo.unblockOverlay()

	select {
	case cands := <-fgDone:
		if len(cands) != 1 || cands[0].Credential.ID != 10 {
			t.Fatalf("foreground should have assembled with OLD base ID 10, got %+v", cands)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("foreground candidate assembly timed out")
	}

	// Check candidate snapshot: old content must keep OLD expiry!
	selector.candidateMu.Lock()
	candSnap := selector.candidates[candKey]
	selector.candidateMu.Unlock()

	if !candSnap.expiresAt.Equal(oldBaseExpiry) {
		t.Fatalf("candidate assembled from old base must keep OLD expiry %v, got %v", oldBaseExpiry, candSnap.expiresAt)
	}
}

// TestSelectorBasePreRefreshObservabilityMetrics verifies perfmetrics telemetry recording
// for attempts, success, discards, and failures.
func TestSelectorBasePreRefreshObservabilityMetrics(t *testing.T) {
	prev := perfmetrics.Default
	defer func() { perfmetrics.Default = prev }()
	perfmetrics.Default = perfmetrics.NewRegistry()

	repo := newBlockingRoutingLayerRepo(nil)
	repo.unblockBase()
	repo.baseSleep = 200 * time.Microsecond
	selector := NewSelector(repo, nil, nil, nil, time.Hour, time.Second, time.Minute)
	selector.UpdateBuildBasePreRefreshPolicy(true, 5*time.Second, 5*time.Second)

	t0 := time.Now().UTC()
	_, _ = selector.loadCandidates(context.Background(), account.ProviderBuild, 0, "model-a", "", t0)
	perfmetrics.Default.CollectAndReset()

	// Successful prerefresh metrics
	selector.preRefreshBuildBaseAt(context.Background(), t0.Add(26*time.Second))

	samples := perfmetrics.Default.CollectAndReset()
	var hasAttempt, hasSuccess, hasRows, hasDuration bool
	for _, sample := range samples {
		if sample.Labels.Subsystem == "gateway" && sample.Labels.Operation == "base_prerefresh" && sample.Labels.Provider == string(account.ProviderBuild) {
			if sample.Labels.Outcome == "attempt" && sample.Name == "selector_base_prerefresh_total" {
				hasAttempt = true
			}
			if sample.Labels.Outcome == "success" {
				if sample.Name == "selector_base_prerefresh_total" {
					hasSuccess = true
				}
				if sample.Name == "selector_base_prerefresh_rows" {
					hasRows = true
				}
				if sample.Name == "selector_base_prerefresh_duration_us" {
					hasDuration = true
				}
			}
		}
	}

	if !hasAttempt || !hasSuccess || !hasRows || !hasDuration {
		t.Fatalf("missing success metrics: attempt=%v, success=%v, rows=%v, duration=%v; samples=%+v",
			hasAttempt, hasSuccess, hasRows, hasDuration, samples)
	}

	// Discarded prerefresh metrics
	blockingRepo := newBlockingRoutingLayerRepo(nil)
	blockingRepo.unblockBase()
	blockingRepo.baseSleep = 200 * time.Microsecond
	selectorDiscard := NewSelector(blockingRepo, nil, nil, nil, time.Hour, time.Second, time.Minute)
	selectorDiscard.UpdateBuildBasePreRefreshPolicy(true, 5*time.Second, 5*time.Second)
	_, _ = selectorDiscard.loadCandidates(context.Background(), account.ProviderBuild, 0, "model-a", "", t0)

	blockingRepo.mu.Lock()
	blockingRepo.blockBase = make(chan struct{})
	blockingRepo.baseStarted = make(chan struct{}, 1)
	blockingRepo.mu.Unlock()

	discardDone := make(chan bool, 1)
	go func() {
		discardDone <- selectorDiscard.preRefreshBuildBaseAt(context.Background(), t0.Add(26*time.Second))
	}()
	<-blockingRepo.baseStarted

	// Invalidate while query blocked
	selectorDiscard.ApplyInvalidation(repository.InvalidationEvent{
		Kind:     repository.InvalidationAccountStateChanged,
		Provider: account.ProviderBuild,
	})
	blockingRepo.unblockBase()
	<-discardDone

	samplesDiscard := perfmetrics.Default.CollectAndReset()
	var hasDiscard bool
	for _, sample := range samplesDiscard {
		if sample.Labels.Subsystem == "gateway" && sample.Labels.Operation == "base_prerefresh" && sample.Labels.Provider == string(account.ProviderBuild) {
			if sample.Labels.Outcome == "discard" && sample.Name == "selector_base_prerefresh_total" {
				hasDiscard = true
			}
		}
	}
	if !hasDiscard {
		t.Fatalf("missing discard metrics in samples=%+v", samplesDiscard)
	}

	// Failure prerefresh metrics
	failRepo := newBlockingRoutingLayerRepo(nil)
	failRepo.unblockBase()
	failRepo.baseSleep = 200 * time.Microsecond
	selectorFail := NewSelector(failRepo, nil, nil, nil, time.Hour, time.Second, time.Minute)
	selectorFail.UpdateBuildBasePreRefreshPolicy(true, 5*time.Second, 5*time.Second)

	// Initial warm load succeeds
	_, _ = selectorFail.loadCandidates(context.Background(), account.ProviderBuild, 0, "model-a", "", t0)

	// Next call fails
	failRepo.mu.Lock()
	failRepo.returnErr = errors.New("simulated query error")
	failRepo.mu.Unlock()

	perfmetrics.Default.CollectAndReset()
	selectorFail.preRefreshBuildBaseAt(context.Background(), t0.Add(26*time.Second))

	samplesFailure := perfmetrics.Default.CollectAndReset()
	var hasFailure bool
	for _, sample := range samplesFailure {
		if sample.Labels.Subsystem == "gateway" && sample.Labels.Operation == "base_prerefresh" && sample.Labels.Provider == string(account.ProviderBuild) {
			if sample.Labels.Outcome == "failure" && sample.Name == "selector_base_prerefresh_total" {
				hasFailure = true
			}
		}
	}
	if !hasFailure {
		t.Fatalf("missing failure metrics in samples=%+v", samplesFailure)
	}
}
