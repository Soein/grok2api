package cli

import (
	"context"
	"log/slog"
	"sync"
	"sync/atomic"
	"time"

	infraegress "github.com/chenyme/grok2api/backend/internal/infra/egress"
	"github.com/chenyme/grok2api/backend/internal/pkg/perfmetrics"
)

const requestTimingLogQueueCapacity = 256

// Only detached metadata enters the queue; never retain request contexts,
// bodies, raw errors, or the mutable timing collector here.
type requestTimingLogRecord struct {
	logger        *slog.Logger
	requestID     string
	operation     string
	statusCode    int
	forwardFailed bool
	completedAt   time.Time
	timing        infraegress.RequestTimingSnapshot
}

type requestTimingLogWriter struct {
	mu      sync.Mutex
	queue   chan requestTimingLogRecord
	done    chan struct{}
	closed  bool
	aborted atomic.Bool
	dropped atomic.Uint64
}

func (w *requestTimingLogWriter) submit(record requestTimingLogRecord) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed {
		w.drop(1, "closed")
		return
	}
	// Lazy initialization avoids workers on disabled adapters and during startup
	// failures. This lock also prevents a late request from restarting after Close.
	if w.queue == nil {
		w.queue = make(chan requestTimingLogRecord, requestTimingLogQueueCapacity)
		w.done = make(chan struct{})
		go w.run()
	}
	select {
	case w.queue <- record:
	default:
		w.drop(1, "queue_full")
	}
}

func (w *requestTimingLogWriter) drop(count uint64, outcome string) {
	if count == 0 {
		return
	}
	w.dropped.Add(count)
	perfmetrics.Default.Add("build_request_timing_log_dropped", perfmetrics.Labels{Subsystem: "diagnostics", Provider: "build", Outcome: outcome}, int64(count))
}

func (w *requestTimingLogWriter) run() {
	defer close(w.done)
	for record := range w.queue {
		if w.aborted.Load() {
			w.drop(1, "shutdown")
			continue
		}
		record.logger.Info("build_request_timing",
			"request_id", record.requestID, "operation", record.operation,
			"status_code", record.statusCode, "forward_failed", record.forwardFailed,
			"completed_at", record.completedAt, "timing", record.timing,
			"timing_logs_dropped_total", w.dropped.Load())
	}
}

// CloseRequestTimingLogs stops accepting diagnostics and drains accepted records
// until ctx expires. Call after requests stop. A blocked logging handler cannot
// be interrupted: the caller returns on timeout, pending records are discarded,
// and at most the existing single worker remains until the handler returns.
// Repeated or concurrent calls are safe; the adapter never restarts this writer.
func (a *Adapter) CloseRequestTimingLogs(ctx context.Context) error {
	return a.timingLogs.close(ctx)
}

func (w *requestTimingLogWriter) close(ctx context.Context) error {
	w.mu.Lock()
	if !w.closed {
		w.closed = true
		if w.queue != nil {
			close(w.queue)
		}
	}
	done, queue := w.done, w.queue
	w.mu.Unlock()
	if done == nil {
		return nil
	}
	select {
	case <-done:
		return nil
	default:
	}
	select {
	case <-done:
		return nil
	case <-ctx.Done():
		w.aborted.Store(true)
		// The channel is closed and producers are rejected, so draining is bounded.
		// Release queued snapshots even if the worker is stuck inside a log handler.
		var dropped uint64
		for range queue {
			dropped++
		}
		w.drop(dropped, "shutdown")
		return ctx.Err()
	}
}
