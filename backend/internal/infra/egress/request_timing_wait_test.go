package egress

import (
	"context"
	"testing"
	"time"
)

func TestRequestTimingMeasuresActualFailureProbeWait(t *testing.T) {
	manager := NewManager(nil, nil)
	done := make(chan struct{})
	manager.failureProbes[1] = failureProbeState{running: true, done: done}
	ctx, root := WithRequestTiming(context.Background())
	ctx, _ = BeginTimingCall(ctx)
	timer := time.AfterFunc(15*time.Millisecond, func() { close(done) })
	defer timer.Stop()
	completed, err := manager.waitForFailureProbe(ctx, 1)
	if err != nil || !completed {
		t.Fatalf("probe completion changed: completed=%t err=%v", completed, err)
	}
	s := root.Snapshot().Calls[0]
	if s.FailureProbeWaitCount != 1 || s.FailureProbeWaitMS < 14 {
		t.Fatalf("missing actual probe wait: %+v", s)
	}
}
