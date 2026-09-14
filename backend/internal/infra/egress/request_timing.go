package egress

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"maps"
	"net/http/httptrace"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

const maxTimingCalls = 16
const maxTimingEventBytes = 64 * 1024
const maxPendingTraceEvents = 64

type requestTimingKey struct{}
type callTimingKey struct{}

// TimingStage names local work that may delay delivery of the upstream stream.
type TimingStage int

const (
	TimingAcquire TimingStage = iota
	TimingFeedback
	TimingFailureProbeWait
)

// RequestTiming contains bounded, observe-only timing for a downstream request.
// It never stores request or response contents in its snapshots.
type RequestTiming struct {
	mu        sync.Mutex
	start     time.Time
	calls     []*CallTiming
	dropped   int
	preflight *PreflightTimingSnapshot
}

// CallTiming records one logical adapter HTTP call, including any transport
// retries below that adapter invocation. All methods tolerate nil.
type CallTiming struct {
	mu             sync.Mutex
	start          time.Time
	snapshot       CallTimingSnapshot
	getConn        []time.Time
	connect        map[string][]time.Time
	connectPending int
	tls            []time.Time
	line           []byte
	data           []byte
	eventBytes     int
	skipEvent      bool
	skipLF         bool
	firstLine      bool
	hasData        bool
	eventType      string

	// Atomic read-boundary diagnostics (nonblocking, safe from contention):
	rawReadStart     atomic.Pointer[float64]
	rawReadDone      atomic.Pointer[float64]
	gzipInitStart    atomic.Pointer[float64]
	gzipInitEnd      atomic.Pointer[float64]
	decodedReadStart atomic.Pointer[float64]
	meta             atomic.Pointer[responseMetadata]
}

// RequestTimingSnapshot is a detached copy safe for structured logging.
type RequestTimingSnapshot struct {
	Preflight    *PreflightTimingSnapshot `json:"preflight,omitempty"`
	TotalMS      float64                  `json:"total_ms"`
	Calls        []CallTimingSnapshot     `json:"calls"`
	DroppedCalls int                      `json:"dropped_calls"`
}

// CallTimingSnapshot uses milliseconds relative to the logical adapter call, except
// StartMS which is relative to the root request. Durations aggregate completed
// observations and may overlap for racing connections. Nil means unobserved.
// Event timestamps retain the earliest observation. With retries they may span
// different wire attempts and must not be read as a final-attempt interval.
type CallTimingSnapshot struct {
	Plane                  string   `json:"plane"`
	Stage                  string   `json:"stage"`
	StartMS                float64  `json:"start_ms"`
	AcquireMS              float64  `json:"acquire_ms"`
	AcquireCount           int      `json:"acquire_count"`
	FeedbackMS             float64  `json:"feedback_ms"`
	FeedbackCount          int      `json:"feedback_count"`
	FailureProbeWaitMS     float64  `json:"failure_probe_wait_ms"`
	FailureProbeWaitCount  int      `json:"failure_probe_wait_count"`
	ConnectionAcquireMS    *float64 `json:"connection_acquire_ms"`
	ConnectionCount        int      `json:"connection_count"`
	ReusedConnectionCount  int      `json:"reused_connection_count"`
	ConnectMS              *float64 `json:"connect_ms"`
	ConnectCount           int      `json:"connect_count"`
	TLSMS                  *float64 `json:"tls_ms"`
	TLSCount               int      `json:"tls_count"`
	WroteRequestMS         *float64 `json:"wrote_request_ms"`
	WriteAttemptCount      int      `json:"write_attempt_count"`
	WriteErrorCount        int      `json:"write_error_count"`
	FirstResponseByteMS    *float64 `json:"first_response_byte_ms"`
	FirstResponseByteCount int      `json:"first_response_byte_count"`
	TransportReturnedMS    *float64 `json:"transport_returned_ms"`
	HTTPReturnedMS         *float64 `json:"http_returned_ms"`
	FirstBodyByteMS        *float64 `json:"first_body_read_ms"`
	FirstSSEEventMS        *float64 `json:"first_sse_event_ms"`
	FirstSSEEventType      string   `json:"first_sse_event_type,omitempty"`
	FirstTextDeltaMS       *float64 `json:"first_text_ms"`
	FirstReasoningDeltaMS  *float64 `json:"first_reasoning_ms"`

	// Read boundary diagnostics:
	// ContentEncoding is the finite class ("identity", "gzip", "other") of the HTTP response.
	// Protocol is the finite class ("http1", "http2", "other") of the HTTP response.
	// Uncompressed reports whether net/http transport decompressed the body before returning
	// (Uncompressed true means client received transparently decompressed stream, not raw wire gzip).
	// FirstRawBodyReadStartMS marks entry into the first Read on the response Body returned by net/http
	// before any adapter gzip normalization. This reflects client-visible read initiation, not wire arrival.
	// FirstRawBodyReadDoneMS marks the first return of n > 0 bytes from that raw body Read.
	// GzipInitStartMS and GzipInitEndMS bound gzip.NewReader initialization when Content-Encoding is gzip.
	// FirstDecodedBodyReadStartMS marks entry into the first Read on the decoded body presented to the caller.
	ContentEncoding             string   `json:"content_encoding,omitempty"`
	Protocol                    string   `json:"protocol,omitempty"`
	Uncompressed                bool     `json:"uncompressed,omitempty"`
	FirstRawBodyReadStartMS     *float64 `json:"first_raw_body_read_start_ms"`
	FirstRawBodyReadDoneMS      *float64 `json:"first_raw_body_read_done_ms"`
	GzipInitStartMS             *float64 `json:"gzip_init_start_ms"`
	GzipInitEndMS               *float64 `json:"gzip_init_end_ms"`
	FirstDecodedBodyReadStartMS *float64 `json:"first_decoded_body_read_start_ms"`
}

// WithRequestTiming enables timing once. A nil context leaves timing disabled.
func WithRequestTiming(ctx context.Context) (context.Context, *RequestTiming) {
	if ctx == nil {
		return ctx, nil
	}
	if root := RequestTimingFromContext(ctx); root != nil {
		return ctx, root
	}
	root := &RequestTiming{start: time.Now(), preflight: PreflightTimingFromContext(ctx).Snapshot()}
	return context.WithValue(ctx, requestTimingKey{}, root), root
}

// RequestTimingFromContext returns the optional request timer.
func RequestTimingFromContext(ctx context.Context) *RequestTiming {
	if ctx == nil {
		return nil
	}
	root, _ := ctx.Value(requestTimingKey{}).(*RequestTiming)
	return root
}

// BeginTimingCall creates a logical adapter-call record, retaining at most 16.
// A dropped child explicitly masks any inherited child in the returned context.
func BeginTimingCall(ctx context.Context) (context.Context, *CallTiming) {
	root := RequestTimingFromContext(ctx)
	if root == nil {
		return ctx, nil
	}
	root.mu.Lock()
	defer root.mu.Unlock()
	if len(root.calls) >= maxTimingCalls {
		root.dropped++
		return context.WithValue(ctx, callTimingKey{}, (*CallTiming)(nil)), nil
	}
	call := &CallTiming{start: time.Now(), firstLine: true}
	physical := physicalCallFromContext(ctx)
	call.snapshot.Plane = normalizePhysicalPlane(physical.plane)
	call.snapshot.Stage = normalizePhysicalStage(physical.stage)
	call.snapshot.StartMS = milliseconds(call.start.Sub(root.start))
	root.calls = append(root.calls, call)
	return context.WithValue(ctx, callTimingKey{}, call), call
}

// CallTimingFromContext returns the optional logical adapter-call timer.
func CallTimingFromContext(ctx context.Context) *CallTiming {
	if ctx == nil {
		return nil
	}
	call, _ := ctx.Value(callTimingKey{}).(*CallTiming)
	return call
}

// RecordStage accumulates local work; overlapping stages are not additive wall time.
func (c *CallTiming) RecordStage(stage TimingStage, duration time.Duration) {
	if c == nil || duration < 0 {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	ms := milliseconds(duration)
	switch stage {
	case TimingAcquire:
		c.snapshot.AcquireMS += ms
		c.snapshot.AcquireCount++
	case TimingFeedback:
		c.snapshot.FeedbackMS += ms
		c.snapshot.FeedbackCount++
	case TimingFailureProbeWait:
		c.snapshot.FailureProbeWaitMS += ms
		c.snapshot.FailureProbeWaitCount++
	}
}

func milliseconds(d time.Duration) float64 { return float64(d) / float64(time.Millisecond) }
func firstTiming(dst **float64, ms float64) {
	if *dst == nil {
		*dst = &ms
	}
}
func addTiming(dst **float64, d time.Duration) {
	ms := milliseconds(d)
	if *dst == nil {
		*dst = &ms
	} else {
		**dst += ms
	}
}

// HTTPTrace returns callbacks to compose with httptrace.WithClientTrace. Timing
// hooks ignore errors, addresses and connection state in exported observations.
// Pairing is FIFO per address for Connect, and FIFO for connection acquisition
// and TLS. Queues are bounded; unmatched completions contribute counts only.
func (c *CallTiming) HTTPTrace() *httptrace.ClientTrace {
	if c == nil {
		return nil
	}
	return &httptrace.ClientTrace{
		GetConn: func(string) {
			c.mu.Lock()
			defer c.mu.Unlock()
			if len(c.getConn) < maxPendingTraceEvents {
				c.getConn = append(c.getConn, time.Now())
			}
		},
		GotConn: func(info httptrace.GotConnInfo) {
			c.mu.Lock()
			defer c.mu.Unlock()
			c.snapshot.ConnectionCount++
			if info.Reused {
				c.snapshot.ReusedConnectionCount++
			}
			if len(c.getConn) > 0 {
				addTiming(&c.snapshot.ConnectionAcquireMS, time.Since(c.getConn[0]))
				c.getConn = c.getConn[1:]
			}
		},
		ConnectStart: func(network, address string) {
			c.mu.Lock()
			defer c.mu.Unlock()
			if c.connectPending >= maxPendingTraceEvents {
				return
			}
			if c.connect == nil {
				c.connect = make(map[string][]time.Time)
			}
			key := network + "\x00" + address
			c.connect[key] = append(c.connect[key], time.Now())
			c.connectPending++
		},
		ConnectDone: func(network, address string, _ error) {
			c.mu.Lock()
			defer c.mu.Unlock()
			c.snapshot.ConnectCount++
			key := network + "\x00" + address
			if pending := c.connect[key]; len(pending) > 0 {
				addTiming(&c.snapshot.ConnectMS, time.Since(pending[0]))
				c.connectPending--
				if len(pending) == 1 {
					delete(c.connect, key)
				} else {
					c.connect[key] = pending[1:]
				}
			}
		},
		TLSHandshakeStart: func() {
			c.mu.Lock()
			defer c.mu.Unlock()
			if len(c.tls) < maxPendingTraceEvents {
				c.tls = append(c.tls, time.Now())
			}
		},
		TLSHandshakeDone: func(tls.ConnectionState, error) {
			c.mu.Lock()
			defer c.mu.Unlock()
			c.snapshot.TLSCount++
			if len(c.tls) > 0 {
				addTiming(&c.snapshot.TLSMS, time.Since(c.tls[0]))
				c.tls = c.tls[1:]
			}
		},
		WroteRequest: func(info httptrace.WroteRequestInfo) {
			c.mu.Lock()
			defer c.mu.Unlock()
			c.snapshot.WriteAttemptCount++
			if info.Err != nil {
				c.snapshot.WriteErrorCount++
			}
			firstTiming(&c.snapshot.WroteRequestMS, milliseconds(time.Since(c.start)))
		},
		GotFirstResponseByte: func() {
			c.mu.Lock()
			defer c.mu.Unlock()
			c.snapshot.FirstResponseByteCount++
			firstTiming(&c.snapshot.FirstResponseByteMS, milliseconds(time.Since(c.start)))
		},
	}
}

// MarkTransportReturned observes return from the transport, before health feedback.
func (c *CallTiming) MarkTransportReturned() {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	firstTiming(&c.snapshot.TransportReturnedMS, milliseconds(time.Since(c.start)))
}

// MarkHTTPReturned observes return from the outer HTTP client, after feedback.
func (c *CallTiming) MarkHTTPReturned() {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	firstTiming(&c.snapshot.HTTPReturnedMS, milliseconds(time.Since(c.start)))
}

// ObserveBody inspects bytes already read by the caller without reading or
// modifying the stream. SSE timings mark complete event observation, not the
// upstream's generation time. Event parsing retains at most 64 KiB and skips
// malformed or oversized events. Only bounded type names survive parsing.
func (c *CallTiming) ObserveBody(chunk []byte) {
	if c == nil || len(chunk) == 0 {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	ms := milliseconds(time.Since(c.start))
	firstTiming(&c.snapshot.FirstBodyByteMS, ms)
	if c.snapshot.FirstTextDeltaMS != nil && c.snapshot.FirstReasoningDeltaMS != nil {
		return
	}
	for _, b := range chunk {
		if c.skipLF {
			c.skipLF = false
			if b == '\n' {
				continue
			}
		}
		if b == '\r' || b == '\n' {
			c.consumeLine(ms)
			c.skipLF = b == '\r'
			continue
		}
		c.eventBytes++
		if c.eventBytes > maxTimingEventBytes {
			c.skipEvent = true
			c.data = nil
		}
		if !c.skipEvent {
			c.line = append(c.line, b)
		} else if len(c.line) == 0 {
			c.line = append(c.line, 0)
		}
	}
}

func (c *CallTiming) consumeLine(ms float64) {
	line := c.line
	c.line = c.line[:0]
	if c.firstLine {
		line = []byte(strings.TrimPrefix(string(line), "\xef\xbb\xbf"))
		c.firstLine = false
	}
	if len(line) == 0 {
		if !c.skipEvent && c.hasData {
			c.observeEvent(ms)
		}
		c.data = nil
		c.hasData = false
		c.eventType = ""
		c.eventBytes = 0
		c.skipEvent = false
		return
	}
	if c.skipEvent {
		return
	}
	c.eventBytes++
	if c.eventBytes > maxTimingEventBytes {
		c.skipEvent = true
		c.data = nil
		return
	}
	if string(line) == "event" || strings.HasPrefix(string(line), "event:") {
		value := strings.TrimPrefix(string(line[len("event"):]), ":")
		c.eventType = safeTimingEventType(strings.TrimPrefix(value, " "))
	} else if string(line) == "data" || strings.HasPrefix(string(line), "data:") {
		value := line[len("data"):]
		if len(value) > 0 {
			value = value[1:]
			if len(value) > 0 && value[0] == ' ' {
				value = value[1:]
			}
		}
		if c.hasData {
			c.data = append(c.data, '\n')
		}
		c.data = append(c.data, value...)
		c.hasData = true
	}
}

func (c *CallTiming) observeEvent(ms float64) {
	var event struct {
		Type    string          `json:"type"`
		Delta   json.RawMessage `json:"delta"`
		Choices []struct {
			Delta struct {
				Content   string `json:"content"`
				Reasoning string `json:"reasoning_content"`
			} `json:"delta"`
		} `json:"choices"`
	}
	valid := json.Unmarshal(c.data, &event) == nil
	kind := "other"
	if valid {
		kind = safeTimingEventType(event.Type)
		if event.Type == "" && c.eventType != "" {
			kind = c.eventType
		}
	}
	if c.snapshot.FirstSSEEventMS == nil {
		firstTiming(&c.snapshot.FirstSSEEventMS, ms)
		c.snapshot.FirstSSEEventType = kind
	}
	if !valid {
		return
	}
	var delta string
	_ = json.Unmarshal(event.Delta, &delta)
	if delta != "" {
		switch kind {
		case "response.output_text.delta":
			firstTiming(&c.snapshot.FirstTextDeltaMS, ms)
		case "response.reasoning_text.delta", "response.reasoning_summary_text.delta":
			firstTiming(&c.snapshot.FirstReasoningDeltaMS, ms)
		}
	}
	for _, choice := range event.Choices {
		if choice.Delta.Content != "" {
			firstTiming(&c.snapshot.FirstTextDeltaMS, ms)
		}
		if choice.Delta.Reasoning != "" {
			firstTiming(&c.snapshot.FirstReasoningDeltaMS, ms)
		}
	}
}

type responseMetadata struct {
	contentEncoding string
	protocol        string
	uncompressed    bool
}

// RecordResponseMetadata captures bounded protocol and encoding metadata for
// an HTTP response. Arbitrary header values and URLs are never recorded.
func (c *CallTiming) RecordResponseMetadata(contentEncoding, proto string, protoMajor int, uncompressed bool) {
	if c == nil {
		return
	}
	m := &responseMetadata{
		contentEncoding: safeContentEncodingClass(contentEncoding),
		protocol:        safeProtocolClass(proto, protoMajor),
		uncompressed:    uncompressed,
	}
	c.meta.Store(m)
}

func recordAtomicTiming(target *atomic.Pointer[float64], start time.Time, observedAt time.Time) {
	if target.Load() != nil {
		return
	}
	if observedAt.IsZero() {
		observedAt = time.Now()
	}
	ms := milliseconds(observedAt.Sub(start))
	target.CompareAndSwap(nil, &ms)
}

func loadDetachedTiming(dst **float64, src *atomic.Pointer[float64]) {
	if p := src.Load(); p != nil {
		v := *p
		*dst = &v
	} else if *dst != nil {
		v := **dst
		*dst = &v
	}
}

// RecordRawBodyReadStart marks entry into the first Read on the raw HTTP response body.
// Recording is nonblocking and does not acquire legacy timing locks.
func (c *CallTiming) RecordRawBodyReadStart(observedAt time.Time) {
	if c == nil {
		return
	}
	recordAtomicTiming(&c.rawReadStart, c.start, observedAt)
}

// RecordRawBodyReadDone marks the first return of n > 0 bytes from the raw response body.
func (c *CallTiming) RecordRawBodyReadDone(observedAt time.Time) {
	if c == nil {
		return
	}
	recordAtomicTiming(&c.rawReadDone, c.start, observedAt)
}

// RecordGzipInitStart marks the start of gzip normalization (gzip.NewReader).
func (c *CallTiming) RecordGzipInitStart(observedAt time.Time) {
	if c == nil {
		return
	}
	recordAtomicTiming(&c.gzipInitStart, c.start, observedAt)
}

// RecordGzipInitEnd marks the completion of gzip normalization, even on failure.
func (c *CallTiming) RecordGzipInitEnd(observedAt time.Time) {
	if c == nil {
		return
	}
	recordAtomicTiming(&c.gzipInitEnd, c.start, observedAt)
}

// RecordDecodedBodyReadStart marks entry into the first Read on the decoded body.
func (c *CallTiming) RecordDecodedBodyReadStart(observedAt time.Time) {
	if c == nil {
		return
	}
	recordAtomicTiming(&c.decodedReadStart, c.start, observedAt)
}

func safeContentEncodingClass(raw string) string {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "", "identity":
		return "identity"
	case "gzip":
		return "gzip"
	default:
		return "other"
	}
}

func safeProtocolClass(proto string, protoMajor int) string {
	switch protoMajor {
	case 1:
		return "http1"
	case 2:
		return "http2"
	}
	normalized := strings.ToLower(strings.TrimSpace(proto))
	switch {
	case strings.HasPrefix(normalized, "http/1.") || normalized == "http/1":
		return "http1"
	case strings.HasPrefix(normalized, "http/2.") || normalized == "http/2" || normalized == "h2":
		return "http2"
	default:
		return "other"
	}
}

func safeTimingEventType(kind string) string {
	switch kind {
	case "response.created", "response.in_progress", "response.completed", "response.failed", "response.output_item.added", "response.output_text.delta", "response.reasoning_text.delta", "response.reasoning_summary_text.delta", "response.function_call_arguments.delta":
		return kind
	default:
		return "other"
	}
}

// Snapshot copies all mutable observations; absent timing events remain nil.
func (r *RequestTiming) Snapshot() RequestTimingSnapshot {
	if r == nil {
		return RequestTimingSnapshot{}
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	result := RequestTimingSnapshot{TotalMS: milliseconds(time.Since(r.start)), DroppedCalls: r.dropped, Calls: make([]CallTimingSnapshot, 0, len(r.calls))}
	// Freeze Gateway work at adapter entry; subsequent account attempts must not
	// change earlier records or the established adapter-relative clock.
	if r.preflight != nil {
		preflight := *r.preflight
		preflight.Stages = maps.Clone(r.preflight.Stages)
		preflight.Counters = maps.Clone(r.preflight.Counters)
		result.Preflight = &preflight
	}
	for _, c := range r.calls {
		c.mu.Lock()
		s := c.snapshot
		for _, ptr := range []**float64{
			&s.ConnectionAcquireMS, &s.ConnectMS, &s.TLSMS, &s.WroteRequestMS,
			&s.FirstResponseByteMS, &s.TransportReturnedMS, &s.HTTPReturnedMS,
			&s.FirstBodyByteMS, &s.FirstSSEEventMS, &s.FirstTextDeltaMS, &s.FirstReasoningDeltaMS,
		} {
			if *ptr != nil {
				value := **ptr
				*ptr = &value
			}
		}
		if m := c.meta.Load(); m != nil {
			s.ContentEncoding = m.contentEncoding
			s.Protocol = m.protocol
			s.Uncompressed = m.uncompressed
		}
		loadDetachedTiming(&s.FirstRawBodyReadStartMS, &c.rawReadStart)
		loadDetachedTiming(&s.FirstRawBodyReadDoneMS, &c.rawReadDone)
		loadDetachedTiming(&s.GzipInitStartMS, &c.gzipInitStart)
		loadDetachedTiming(&s.GzipInitEndMS, &c.gzipInitEnd)
		loadDetachedTiming(&s.FirstDecodedBodyReadStartMS, &c.decodedReadStart)
		c.mu.Unlock()
		result.Calls = append(result.Calls, s)
	}
	return result
}
