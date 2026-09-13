package cli

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/chenyme/grok2api/backend/internal/domain/account"
	infraegress "github.com/chenyme/grok2api/backend/internal/infra/egress"
	"github.com/chenyme/grok2api/backend/internal/infra/provider"
)

type blockedTimingOutput struct {
	sink    io.Writer
	entered chan struct{}
	release chan struct{}
	once    sync.Once
}

func (w *blockedTimingOutput) Write(p []byte) (int, error) {
	w.once.Do(func() { close(w.entered) })
	<-w.release
	if w.sink != nil {
		return w.sink.Write(p)
	}
	return len(p), nil
}

func TestBuildRequestTimingBlockedOutputDoesNotBlockRequest(t *testing.T) {
	for _, failed := range []bool{false, true} {
		name := "body_close"
		if failed {
			name = "transport_failure"
		}
		t.Run(name, func(t *testing.T) {
			adapter, token := newTimingTestAdapter(t)
			cfg := adapter.config()
			cfg.RequestTimingEnabled = true
			adapter.UpdateConfig(cfg)
			output := &blockedTimingOutput{entered: make(chan struct{}), release: make(chan struct{})}
			var releaseOnce sync.Once
			release := func() { releaseOnce.Do(func() { close(output.release) }) }
			defer release()
			adapter.SetLogger(slog.New(slog.NewJSONHandler(output, nil)))
			adapter.http.Transport = roundTripFunc(func(req *http.Request) (*http.Response, error) {
				if failed {
					return nil, io.ErrUnexpectedEOF
				}
				return &http.Response{StatusCode: 200, Header: http.Header{}, Body: io.NopCloser(strings.NewReader("data: [DONE]\n\n")), Request: req}, nil
			})
			done := make(chan error, 1)
			go func() {
				response, err := adapter.ForwardResponse(context.Background(), provider.ResponseResourceRequest{RequestID: "blocked-output", Credential: account.Credential{ID: 1, Provider: account.ProviderBuild, EncryptedAccessToken: token}, Method: http.MethodPost, Path: "/responses", Model: "grok-4.6", Streaming: true, Operation: "responses"})
				if failed {
					done <- err
					return
				}
				if err != nil {
					done <- err
					return
				}
				_, _ = io.Copy(io.Discard, response.Body)
				done <- response.Body.Close()
			}()
			select {
			case err := <-done:
				if (err != nil) != failed {
					t.Fatalf("request result changed: %v", err)
				}
			case <-time.After(300 * time.Millisecond):
				release()
				<-done
				t.Fatal("diagnostic log output blocked the request")
			}
			select {
			case <-output.entered:
			case <-time.After(time.Second):
				t.Fatal("background writer did not write the timing record")
			}
		})
	}
}

func queueTimingTestRecord(adapter *Adapter, id string) *infraegress.CallTiming {
	ctx, timing := infraegress.WithRequestTiming(context.Background())
	_, call := infraegress.BeginTimingCall(ctx)
	adapter.attachRequestTiming(nil, nil, provider.ResponseResourceRequest{RequestID: id, Operation: "responses"}, timing)
	return call
}

func TestBuildRequestTimingFullQueueDropsWithoutBlocking(t *testing.T) {
	adapter, _ := newTimingTestAdapter(t)
	var logs bytes.Buffer
	output := &blockedTimingOutput{entered: make(chan struct{}), release: make(chan struct{}), sink: &logs}
	var once sync.Once
	release := func() { once.Do(func() { close(output.release) }) }
	defer release()
	adapter.SetLogger(slog.New(slog.NewJSONHandler(output, nil)))
	queueTimingTestRecord(adapter, "in-flight")
	<-output.entered
	done := make(chan struct{})
	go func() {
		for i := 0; i < requestTimingLogQueueCapacity+7; i++ {
			queueTimingTestRecord(adapter, fmt.Sprintf("queued-%d", i))
		}
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(time.Second):
		release()
		<-done
		t.Fatal("full diagnostic queue blocked enqueue")
	}
	if got := adapter.timingLogs.dropped.Load(); got != 7 {
		t.Fatalf("dropped=%d, want 7", got)
	}
	if got := len(adapter.timingLogs.queue); got != requestTimingLogQueueCapacity {
		t.Fatalf("queue size=%d", got)
	}
	release()
	closeTimingTestLogs(t, adapter)
	lines := strings.Split(strings.TrimSpace(logs.String()), "\n")
	if len(lines) != requestTimingLogQueueCapacity+1 {
		t.Fatalf("written=%d, want %d", len(lines), requestTimingLogQueueCapacity+1)
	}
	if !strings.Contains(lines[len(lines)-1], `"timing_logs_dropped_total":7`) {
		t.Fatal("drop count not observable after output recovers")
	}
}

func TestBuildRequestTimingCloseDeadlineDropsPendingAndCannotRestart(t *testing.T) {
	adapter, _ := newTimingTestAdapter(t)
	var logs bytes.Buffer
	output := &blockedTimingOutput{entered: make(chan struct{}), release: make(chan struct{}), sink: &logs}
	var once sync.Once
	release := func() { once.Do(func() { close(output.release) }) }
	defer release()
	adapter.SetLogger(slog.New(slog.NewJSONHandler(output, nil)))
	queueTimingTestRecord(adapter, "in-flight")
	<-output.entered
	for i := 0; i < 3; i++ {
		queueTimingTestRecord(adapter, "pending")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- adapter.CloseRequestTimingLogs(ctx) }()
	select {
	case err := <-done:
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("close=%v", err)
		}
	case <-time.After(time.Second):
		release()
		<-done
		t.Fatal("close ignored deadline")
	}
	if got := len(adapter.timingLogs.queue); got != 0 {
		t.Fatalf("retained %d pending snapshots", got)
	}
	queueTimingTestRecord(adapter, "after-close")
	if got := adapter.timingLogs.dropped.Load(); got != 4 {
		t.Fatalf("dropped=%d, want 4", got)
	}
	release()
	closeTimingTestLogs(t, adapter)
	if got := strings.Count(logs.String(), `"msg":"build_request_timing"`); got != 1 {
		t.Fatalf("written=%d, want only in-flight record", got)
	}
}

func TestBuildRequestTimingQueuesDetachedSnapshot(t *testing.T) {
	adapter, _ := newTimingTestAdapter(t)
	var logs bytes.Buffer
	output := &blockedTimingOutput{entered: make(chan struct{}), release: make(chan struct{}), sink: &logs}
	var once sync.Once
	release := func() { once.Do(func() { close(output.release) }) }
	defer release()
	adapter.SetLogger(slog.New(slog.NewJSONHandler(output, nil)))
	queueTimingTestRecord(adapter, "blocking-first")
	<-output.entered
	queuedAt := time.Now()
	call := queueTimingTestRecord(adapter, "snapshot")
	// The queued metadata must not reference the live collector or include queue delay.
	call.ObserveBody([]byte("data: {\"type\":\"response.output_text.delta\",\"delta\":\"private-late-text\"}\n\n"))
	time.Sleep(15 * time.Millisecond)
	releasedAt := time.Now()
	release()
	closeTimingTestLogs(t, adapter)
	lines := strings.Split(strings.TrimSpace(logs.String()), "\n")
	var record struct {
		CompletedAt time.Time                         `json:"completed_at"`
		Timing      infraegress.RequestTimingSnapshot `json:"timing"`
	}
	if len(lines) != 2 {
		t.Fatalf("logs=%s", logs.String())
	}
	if err := json.Unmarshal([]byte(lines[1]), &record); err != nil {
		t.Fatal(err)
	}
	if len(record.Timing.Calls) != 1 || record.Timing.Calls[0].FirstTextDeltaMS != nil {
		t.Fatal("queued snapshot mutated after enqueue")
	}
	if record.CompletedAt.Before(queuedAt) || !record.CompletedAt.Before(releasedAt) {
		t.Fatal("completion timestamp includes background queue delay")
	}
	if strings.Contains(logs.String(), "private-late-text") {
		t.Fatal("body leaked into queue")
	}
}

func TestBuildRequestTimingConcurrentCloseAndEnqueue(t *testing.T) {
	adapter, _ := newTimingTestAdapter(t)
	adapter.SetLogger(slog.New(slog.NewJSONHandler(io.Discard, nil)))
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				queueTimingTestRecord(adapter, "concurrent")
			}
		}()
	}
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ctx, cancel := context.WithTimeout(context.Background(), time.Second)
			defer cancel()
			if err := adapter.CloseRequestTimingLogs(ctx); err != nil {
				t.Errorf("close: %v", err)
			}
		}()
	}
	wg.Wait()
	closeTimingTestLogs(t, adapter)
}

func TestBuildRequestTimingCloseBeforeFirstRecordDoesNotStartWorker(t *testing.T) {
	adapter, _ := newTimingTestAdapter(t)
	closeTimingTestLogs(t, adapter)
	queueTimingTestRecord(adapter, "late")
	if adapter.timingLogs.queue != nil || adapter.timingLogs.done != nil {
		t.Fatal("closed writer restarted")
	}
	if got := adapter.timingLogs.dropped.Load(); got != 1 {
		t.Fatalf("late drops=%d", got)
	}
}
