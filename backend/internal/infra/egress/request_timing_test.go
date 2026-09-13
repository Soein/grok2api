package egress

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"errors"
	"net/http/httptrace"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestRequestTimingDisabledAndCallLimit(t *testing.T) {
	if ctx, root := WithRequestTiming(nil); ctx != nil || root != nil || RequestTimingFromContext(nil) != nil || CallTimingFromContext(nil) != nil {
		t.Fatal("nil context enabled timing")
	}
	var rootNil *RequestTiming
	if len(rootNil.Snapshot().Calls) != 0 {
		t.Fatal("nil snapshot has calls")
	}
	var call *CallTiming
	call.RecordStage(TimingAcquire, time.Second)
	call.MarkTransportReturned()
	call.MarkHTTPReturned()
	call.ObserveBody([]byte("secret"))
	if call.HTTPTrace() != nil {
		t.Fatal("disabled trace enabled")
	}
	ctx := context.Background()
	if got, c := BeginTimingCall(ctx); got != ctx || c != nil {
		t.Fatal("disabled context changed")
	}
	ctx, root := WithRequestTiming(ctx)
	if got, again := WithRequestTiming(ctx); got != ctx || again != root {
		t.Fatal("root not reused")
	}
	var lastChild context.Context
	for i := 0; i < 18; i++ {
		child, c := BeginTimingCall(ctx)
		if i < 16 {
			if c == nil || CallTimingFromContext(child) != c {
				t.Fatal("missing child")
			}
			c.RecordStage(TimingAcquire, time.Duration(i)*time.Millisecond)
			lastChild = child
		} else if c != nil {
			t.Fatal("unbounded children")
		}
	}
	s := root.Snapshot()
	if len(s.Calls) != 16 || s.DroppedCalls != 2 || s.Calls[1].AcquireMS != 1 || s.Calls[0].AcquireMS != 0 {
		t.Fatalf("bad snapshot: %+v", s)
	}
	if s.Calls[0].FirstBodyByteMS != nil {
		t.Fatal("missing observation represented as zero")
	}
	masked, dropped := BeginTimingCall(lastChild)
	if dropped != nil || CallTimingFromContext(masked) != nil {
		t.Fatal("dropped child inherited prior call")
	}
}

func TestRequestTimingEmptyDeltaAndIndependentBodies(t *testing.T) {
	ctx, root := WithRequestTiming(context.Background())
	_, first := BeginTimingCall(ctx)
	_, second := BeginTimingCall(ctx)
	first.ObserveBody([]byte(": heartbeat\n\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"\"}\n\n"))
	second.ObserveBody([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n"))
	s := root.Snapshot()
	if s.Calls[0].FirstTextDeltaMS != nil || s.Calls[1].FirstTextDeltaMS == nil {
		t.Fatal("empty delta or call separation incorrect")
	}
	if s.Calls[0].FirstSSEEventType != "response.output_text.delta" {
		t.Fatal("comment counted as SSE data event")
	}
}

func TestRequestTimingSSEEventFieldFallbackAndReset(t *testing.T) {
	ctx, root := WithRequestTiming(context.Background())
	_, c := BeginTimingCall(ctx)
	c.ObserveBody([]byte("event: response.reasoning_text.delta\ndata: {\"delta\":\"thought\"}\n\n"))
	c.ObserveBody([]byte("data: {\"delta\":\"untyped\"}\n\n"))
	if root.Snapshot().Calls[0].FirstTextDeltaMS != nil {
		t.Fatal("untyped event classified as text")
	}
	c.ObserveBody([]byte("event: response.output_text.delta\ndata: {\"delta\":\"hello\"}\n\n"))
	s := root.Snapshot().Calls[0]
	if s.FirstSSEEventType != "response.reasoning_text.delta" || s.FirstReasoningDeltaMS == nil || s.FirstTextDeltaMS == nil {
		t.Fatalf("event fallback missing: %+v", s)
	}

	_, second := BeginTimingCall(ctx)
	second.ObserveBody([]byte("event: response.output_text.delta\ndata: {\"type\":\"response.created\",\"delta\":\"ignore\"}\n\n"))
	second.ObserveBody([]byte("data: {\"delta\":\"ignore\"}\n\n"))
	second.ObserveBody([]byte("event: secret-event\ndata: {\"delta\":\"secret-body\"}\n\n"))
	snapshot := root.Snapshot()
	if snapshot.Calls[1].FirstTextDeltaMS != nil || snapshot.Calls[1].FirstSSEEventType != "response.created" {
		t.Fatal("JSON type precedence or event reset incorrect")
	}
	encoded, _ := json.Marshal(snapshot)
	if strings.Contains(string(encoded), "secret") {
		t.Fatal("event field leaked")
	}
}

func TestRequestTimingSSEChunksAndSnapshotIsolation(t *testing.T) {
	ctx, root := WithRequestTiming(context.Background())
	_, c := BeginTimingCall(ctx)
	for _, chunk := range []string{"\xef", "\xbb\xbfdata: {\"type\":\"response.created\"}\r", "\n\r\n", "data: {\"type\":\"response.reasoning_summary_text.delta\",\n", "data: \"delta\":\"thinking-secret\"}\n\n", "data: {\"type\":\"response.output_text.delta\",\"delta\":\"", "body-secret\"}\r\n\r\n"} {
		c.ObserveBody([]byte(chunk))
	}
	s := root.Snapshot()
	a := s.Calls[0]
	if a.FirstBodyByteMS == nil || a.FirstSSEEventMS == nil || a.FirstSSEEventType != "response.created" || a.FirstTextDeltaMS == nil || a.FirstReasoningDeltaMS == nil {
		t.Fatalf("missing timings: %+v", a)
	}
	if *a.FirstTextDeltaMS < *a.FirstReasoningDeltaMS {
		t.Fatal("text recorded before reasoning")
	}
	*a.FirstTextDeltaMS = -1
	if *root.Snapshot().Calls[0].FirstTextDeltaMS < 0 {
		t.Fatal("mutable snapshot alias")
	}
	encoded, _ := json.Marshal(s)
	if strings.Contains(string(encoded), "secret") {
		t.Fatalf("sensitive data leaked: %s", encoded)
	}
}

func TestRequestTimingOversizedAndUnknownEvents(t *testing.T) {
	ctx, root := WithRequestTiming(context.Background())
	_, c := BeginTimingCall(ctx)
	c.ObserveBody([]byte("event: secret-event\ndata: {\"type\":\"secret-type\"}\n\n"))
	c.ObserveBody([]byte("data: " + strings.Repeat("x", 128*1024) + "\n\n"))
	c.ObserveBody([]byte("data: invalid-secret-json\n\n"))
	c.ObserveBody([]byte("data: {\"type\":\"response.output_text.delta\",\"delta\":\"safe\"}\n\n"))
	s := root.Snapshot()
	if s.Calls[0].FirstSSEEventType != "other" || s.Calls[0].FirstTextDeltaMS == nil {
		t.Fatalf("did not recover from oversized event: %+v", s)
	}
	encoded, _ := json.Marshal(s)
	if strings.Contains(string(encoded), "secret") {
		t.Fatal("unknown type leaked")
	}
}

func TestRequestTimingConcurrentTrace(t *testing.T) {
	ctx, root := WithRequestTiming(context.Background())
	_, c := BeginTimingCall(ctx)
	trace := c.HTTPTrace()
	var wg sync.WaitGroup
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			trace.GetConn("secret-host")
			trace.ConnectStart("tcp", "secret-address")
			trace.ConnectDone("tcp", "secret-address", errors.New("secret-error"))
			trace.TLSHandshakeStart()
			trace.TLSHandshakeDone(tls.ConnectionState{}, nil)
			trace.GotConn(httptrace.GotConnInfo{Reused: true})
			trace.WroteRequest(httptrace.WroteRequestInfo{})
			trace.GotFirstResponseByte()
			c.RecordStage(TimingFeedback, time.Millisecond)
			c.MarkTransportReturned()
			c.MarkHTTPReturned()
			_ = root.Snapshot()
		}()
	}
	wg.Wait()
	s := root.Snapshot()
	a := s.Calls[0]
	if a.ConnectionCount != 20 || a.ReusedConnectionCount != 20 || a.FeedbackCount != 20 || a.WroteRequestMS == nil || a.FirstResponseByteMS == nil || a.ConnectMS == nil || a.TLSMS == nil {
		t.Fatalf("missing trace observations: %+v", a)
	}
	encoded, _ := json.Marshal(s)
	if strings.Contains(string(encoded), "secret") {
		t.Fatal("trace leaked sensitive fields")
	}
}

func TestRequestTimingLogicalCallRetryCountersAndPlane(t *testing.T) {
	ctx := WithPhysicalCallTrace(context.Background(), "grok_build", "responses")
	ctx = WithPhysicalCallStage(WithPhysicalCallPlane(ctx, "xai"), "plane_fallback")
	ctx, root := WithRequestTiming(ctx)
	_, c := BeginTimingCall(ctx)
	trace := c.HTTPTrace()
	trace.WroteRequest(httptrace.WroteRequestInfo{Err: errors.New("secret-error")})
	trace.GotFirstResponseByte()
	first := root.Snapshot().Calls[0]
	trace.WroteRequest(httptrace.WroteRequestInfo{})
	trace.GotFirstResponseByte()
	s := root.Snapshot().Calls[0]
	if s.Plane != "xai" || s.Stage != "plane_fallback" || s.WriteAttemptCount != 2 || s.WriteErrorCount != 1 || s.FirstResponseByteCount != 2 {
		t.Fatalf("missing logical retry metadata: %+v", s)
	}
	if *s.WroteRequestMS != *first.WroteRequestMS || *s.FirstResponseByteMS != *first.FirstResponseByteMS {
		t.Fatal("retry replaced first observed timestamp")
	}
	_, _ = BeginTimingCall(WithPhysicalCallStage(WithPhysicalCallPlane(ctx, "secret-plane"), "secret-stage"))
	snapshot := root.Snapshot()
	if snapshot.Calls[1].Plane != "unknown" || snapshot.Calls[1].Stage != "other" {
		t.Fatal("plane or stage not bounded")
	}
	encoded, _ := json.Marshal(snapshot)
	if strings.Contains(string(encoded), "secret") {
		t.Fatal("retry metadata leaked")
	}
}
