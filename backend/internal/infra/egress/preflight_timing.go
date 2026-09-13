package egress

import (
	"context"
	"sync"
	"time"
)

type preflightTimingKey struct{}

// PreflightTiming records bounded request preparation metadata. All methods
// tolerate nil; disabled callers do not read the clock or allocate snapshots.
// Stages accumulate inclusive durations and may nest, so they are not additive.
type PreflightTiming struct {
	mu        sync.Mutex
	startedAt time.Time
	stages    map[string]PreflightStageTiming
	counters  map[string]int
}

// PreflightStageTiming aggregates completed observations of one allowed stage.
type PreflightStageTiming struct {
	DurationMS float64 `json:"duration_ms"`
	Count      int     `json:"count"`
}

// PreflightTimingSnapshot is detached from the collector for asynchronous logs.
// TotalMS spans gateway entry through snapshot creation, including retries.
// A *_shared_load stage measures the complete singleflight.Do call, including
// either its own loader or waiting for another caller. *_loader_executed identifies
// callers whose closure ran; query stages belong only to that caller. Missing
// stages are unobserved, not zero-duration operations.
type PreflightTimingSnapshot struct {
	TotalMS  float64                         `json:"total_ms"`
	Stages   map[string]PreflightStageTiming `json:"stages"`
	Counters map[string]int                  `json:"counters"`
}

// WithPreflightTiming starts metadata collection at the caller's gateway clock.
func WithPreflightTiming(ctx context.Context, startedAt time.Time) (context.Context, *PreflightTiming) {
	p := &PreflightTiming{startedAt: startedAt, stages: make(map[string]PreflightStageTiming), counters: make(map[string]int)}
	return context.WithValue(ctx, preflightTimingKey{}, p), p
}

// PreflightTimingFromContext returns nil when diagnostics were not enabled.
func PreflightTimingFromContext(ctx context.Context) *PreflightTiming {
	if ctx == nil {
		return nil
	}
	p, _ := ctx.Value(preflightTimingKey{}).(*PreflightTiming)
	return p
}

// Start reads the clock only when collection is enabled.
func (p *PreflightTiming) Start() time.Time {
	if p == nil {
		return time.Time{}
	}
	return time.Now()
}

// Observe records a completed allowed stage; arbitrary names are discarded.
func (p *PreflightTiming) Observe(stage string, startedAt time.Time) {
	if p == nil || startedAt.IsZero() || !preflightStageAllowed(stage) {
		return
	}
	duration := time.Since(startedAt)
	if duration < 0 {
		return
	}
	p.mu.Lock()
	value := p.stages[stage]
	value.DurationMS += float64(duration) / float64(time.Millisecond)
	value.Count++
	p.stages[stage] = value
	p.mu.Unlock()
}

// Add increments an allowed metadata counter. Negative updates are ignored.
func (p *PreflightTiming) Add(counter string, n int) {
	if p == nil || n < 0 || !preflightCounterAllowed(counter) {
		return
	}
	p.mu.Lock()
	p.counters[counter] += n
	p.mu.Unlock()
}

// Snapshot copies all maps; later observations cannot change a returned value.
func (p *PreflightTiming) Snapshot() *PreflightTimingSnapshot {
	if p == nil {
		return nil
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	snapshot := &PreflightTimingSnapshot{TotalMS: float64(time.Since(p.startedAt)) / float64(time.Millisecond), Stages: make(map[string]PreflightStageTiming, len(p.stages)), Counters: make(map[string]int, len(p.counters))}
	for key, value := range p.stages {
		snapshot.Stages[key] = value
	}
	for key, value := range p.counters {
		snapshot.Counters[key] = value
	}
	return snapshot
}

func preflightStageAllowed(stage string) bool {
	switch stage {
	case "route_resolve", "route_ownership", "preselection_identity", "preselection", "alias_rewrite", "media_summary", "session_identity", "billing_estimate", "billing_reserve", "selection_acquire", "credential",
		"candidate_load", "candidate_filter", "base_load", "overlay_load", "base_query", "overlay_query", "candidate_assemble", "bot_filter", "combined_query", "candidate_shared_load", "base_shared_load", "overlay_shared_load":
		return true
	}
	return false
}

func preflightCounterAllowed(counter string) bool {
	switch counter {
	case "adapter_calls", "candidate_cache_hit", "candidate_cache_miss", "candidate_loader_executed", "base_cache_hit", "base_cache_miss", "base_loader_executed", "overlay_cache_hit", "overlay_cache_miss", "overlay_loader_executed",
		"candidate_version_retry_after_load", "candidate_version_retry_before_store", "candidate_combined_fallback", "candidates_loaded", "candidates_normal", "candidates_probe", "base_rows_loaded", "candidates_assembled":
		return true
	}
	return false
}
