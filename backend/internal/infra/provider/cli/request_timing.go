package cli

import (
	"io"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	infraegress "github.com/chenyme/grok2api/backend/internal/infra/egress"
	"github.com/chenyme/grok2api/backend/internal/infra/provider"
)

func isTimingTextOperation(operation string) bool {
	switch operation {
	case "", "responses", "chat", "messages":
		return true
	default:
		return false
	}
}

func (a *Adapter) attachRequestTiming(response *provider.Response, forwardErr error, request provider.ResponseResourceRequest, timing *infraegress.RequestTiming) {
	logger := a.logger
	if logger == nil {
		logger = slog.Default()
	}
	record := requestTimingLogRecord{
		logger: logger, requestID: request.RequestID, operation: request.Operation,
		forwardFailed: forwardErr != nil,
	}
	if response != nil {
		record.statusCode = response.StatusCode
	}
	finish := func() {
		record.timing = timing.Snapshot()
		record.completedAt = time.Now()
		a.timingLogs.submit(record)
	}
	if response == nil || response.Body == nil {
		finish()
		return
	}
	response.Body = &timingFinalBody{ReadCloser: response.Body, finish: finish}
}

type timingFinalBody struct {
	io.ReadCloser
	once     sync.Once
	finish   func()
	closeErr error
}

func (b *timingFinalBody) Close() error {
	b.once.Do(func() { b.closeErr = b.ReadCloser.Close(); b.finish() })
	return b.closeErr
}

func isGzipResponse(response *http.Response) bool {
	return response != nil && response.Body != nil && strings.EqualFold(strings.TrimSpace(response.Header.Get("Content-Encoding")), "gzip")
}

// Observe the decoded upstream representation before compatibility transforms.
func normalizeTimedBuildResponse(call responseCall) error {
	timingApplicable := call.timing != nil && call.response != nil && call.response.Body != nil && isHTTPSuccess(call.response.StatusCode)
	if timingApplicable {
		call.timing.RecordResponseMetadata(
			call.response.Header.Get("Content-Encoding"),
			call.response.Proto,
			call.response.ProtoMajor,
			call.response.Uncompressed,
		)
		call.response.Body = &timingRawBody{ReadCloser: call.response.Body, timing: call.timing}
	}

	isGzip := isGzipResponse(call.response)
	if isGzip && timingApplicable {
		call.timing.RecordGzipInitStart(time.Now())
	}
	err := normalizeGzipResponse(call.response)
	if isGzip && timingApplicable {
		call.timing.RecordGzipInitEnd(time.Now())
	}
	if err != nil {
		return err
	}

	if timingApplicable && call.response != nil && call.response.Body != nil {
		call.response.Body = &timingObservedBody{ReadCloser: call.response.Body, timing: call.timing}
	}
	return nil
}

type timingRawBody struct {
	io.ReadCloser
	timing      *infraegress.CallTiming
	readStarted atomic.Bool
	readDone    atomic.Bool
}

func (b *timingRawBody) Read(p []byte) (int, error) {
	if !b.readStarted.Load() {
		observedAt := time.Now()
		if b.readStarted.CompareAndSwap(false, true) {
			b.timing.RecordRawBodyReadStart(observedAt)
		}
	}
	n, err := b.ReadCloser.Read(p)
	if n > 0 && !b.readDone.Load() {
		observedAt := time.Now()
		if b.readDone.CompareAndSwap(false, true) {
			b.timing.RecordRawBodyReadDone(observedAt)
		}
	}
	return n, err
}

type timingObservedBody struct {
	io.ReadCloser
	timing      *infraegress.CallTiming
	readStarted atomic.Bool
}

func (b *timingObservedBody) Read(p []byte) (int, error) {
	if !b.readStarted.Load() {
		observedAt := time.Now()
		if b.readStarted.CompareAndSwap(false, true) {
			b.timing.RecordDecodedBodyReadStart(observedAt)
		}
	}
	n, err := b.ReadCloser.Read(p)
	if n > 0 {
		b.timing.ObserveBody(p[:n])
	}
	return n, err
}
