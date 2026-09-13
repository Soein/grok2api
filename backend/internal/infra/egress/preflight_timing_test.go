package egress

import (
	"context"
	"encoding/json"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestPreflightTimingDisabled(t *testing.T) {
	p := PreflightTimingFromContext(context.Background())
	if p != nil || !p.Start().IsZero() || p.Snapshot() != nil {
		t.Fatal("disabled collector allocated state")
	}
	p.Observe("route_resolve", time.Time{})
	p.Add("adapter_calls", 1)
}

func TestPreflightTimingBoundedDetached(t *testing.T) {
	ctx, p := WithPreflightTiming(context.Background(), time.Now().Add(-time.Second))
	if PreflightTimingFromContext(ctx) != p {
		t.Fatal("context collector differs")
	}
	start := p.Start()
	p.Observe("route_resolve", start)
	p.Add("adapter_calls", 1)
	p.Observe("secret-account-42", start)
	p.Add("secret-token", 9)
	p.Add("adapter_calls", -1)
	p.Observe("credential", time.Time{})
	first := p.Snapshot()
	if first.TotalMS < 1000 || first.Stages["route_resolve"].Count != 1 || first.Counters["adapter_calls"] != 1 {
		t.Fatalf("snapshot = %+v", first)
	}
	if len(first.Stages) != 1 || len(first.Counters) != 1 {
		t.Fatalf("unexpected keys: %+v", first)
	}
	p.Add("adapter_calls", 1)
	p.Observe("route_resolve", p.Start())
	first.Stages["route_resolve"] = PreflightStageTiming{Count: 99}
	first.Counters["adapter_calls"] = 99
	next := p.Snapshot()
	if next.Stages["route_resolve"].Count != 2 || next.Counters["adapter_calls"] != 2 {
		t.Fatal("snapshot is not detached")
	}
	data, err := json.Marshal(next)
	if err != nil || strings.Contains(string(data), "secret") {
		t.Fatalf("unsafe JSON: %s %v", data, err)
	}
}

func TestPreflightTimingConcurrent(t *testing.T) {
	_, p := WithPreflightTiming(context.Background(), time.Now())
	var wg sync.WaitGroup
	for range 20 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 100 {
				p.Add("adapter_calls", 1)
				p.Observe("credential", p.Start())
				_ = p.Snapshot()
			}
		}()
	}
	wg.Wait()
	snapshot := p.Snapshot()
	if snapshot.Counters["adapter_calls"] != 2000 || snapshot.Stages["credential"].Count != 2000 {
		t.Fatalf("snapshot = %+v", snapshot)
	}
}
