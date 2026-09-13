package egress

import (
	"context"
	"testing"
	"time"
)

func TestRequestTimingFreezesPreflightWithoutChangingAdapterOrigin(t *testing.T) {
	ctx, preflight := WithPreflightTiming(context.Background(), time.Now().Add(-time.Second))
	preflight.Add("adapter_calls", 1)
	_, timing := WithRequestTiming(ctx)
	first := timing.Snapshot()
	if first.Preflight == nil || first.Preflight.TotalMS < 1000 {
		t.Fatalf("missing preflight: %#v", first)
	}
	if first.TotalMS >= first.Preflight.TotalMS {
		t.Fatalf("adapter origin changed: %#v", first)
	}
	preflight.Add("adapter_calls", 1)
	first.Preflight.Counters["adapter_calls"] = 99
	second := timing.Snapshot()
	if second.Preflight.Counters["adapter_calls"] != 1 {
		t.Fatal("preflight snapshot mutated after adapter entry")
	}
	if first.Preflight.TotalMS != second.Preflight.TotalMS {
		t.Fatal("preflight duration includes adapter execution")
	}
}
