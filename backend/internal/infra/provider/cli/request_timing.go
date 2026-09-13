package cli

import (
	"io"
	"log/slog"
	"sync"
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

// Observe the decoded upstream representation before compatibility transforms.
func normalizeTimedBuildResponse(call responseCall) error {
	if err := normalizeGzipResponse(call.response); err != nil {
		return err
	}
	if call.timing != nil && call.response != nil && call.response.Body != nil && isHTTPSuccess(call.response.StatusCode) {
		call.response.Body = &timingObservedBody{ReadCloser: call.response.Body, timing: call.timing}
	}
	return nil
}

type timingObservedBody struct {
	io.ReadCloser
	timing *infraegress.CallTiming
}

func (b *timingObservedBody) Read(p []byte) (int, error) {
	n, err := b.ReadCloser.Read(p)
	if n > 0 {
		b.timing.ObserveBody(p[:n])
	}
	return n, err
}
