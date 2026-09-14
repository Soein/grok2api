package cli

import (
	"bytes"
	"compress/gzip"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/chenyme/grok2api/backend/internal/domain/account"
	infraegress "github.com/chenyme/grok2api/backend/internal/infra/egress"
	"github.com/chenyme/grok2api/backend/internal/infra/provider"
	"github.com/chenyme/grok2api/backend/internal/infra/security"
)

func TestBuildRequestTimingRecordsRawStreamWithoutSecrets(t *testing.T) {
	adapter, token := newTimingTestAdapter(t)
	cfg := adapter.config()
	if err := json.Unmarshal([]byte(`{"RequestTimingEnabled":true}`), &cfg); err != nil {
		t.Fatal(err)
	}
	adapter.UpdateConfig(cfg)
	var logs bytes.Buffer
	adapter.SetLogger(slog.New(slog.NewJSONHandler(&logs, nil)))
	const stream = "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"private-output-sentinel\"}\n\nevent: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_test\",\"status\":\"completed\"}}\n\n"
	adapter.http.Transport = roundTripFunc(func(req *http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: 200, Status: "200 OK", Header: http.Header{"Content-Type": []string{"text/event-stream"}, "Set-Cookie": []string{"private-cookie-sentinel"}}, Body: io.NopCloser(strings.NewReader(stream)), Request: req}, nil
	})
	request := provider.ResponseResourceRequest{Credential: account.Credential{ID: 1, Provider: account.ProviderBuild, EncryptedAccessToken: token}, Method: http.MethodPost, Path: "/responses", Model: "grok-4.6", Body: []byte(`{"model":"grok-4.6","input":"private-input-sentinel","stream":true}`), Streaming: true, NormalizeBody: true, Operation: "responses"}
	if err := json.Unmarshal([]byte(`{"RequestID":"timing-test-1"}`), &request); err != nil {
		t.Fatal(err)
	}
	response, err := adapter.ForwardResponse(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	output, err := io.ReadAll(response.Body)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(output, []byte("private-output-sentinel")) {
		t.Fatal("instrumentation changed forwarded content")
	}
	if logs.Len() != 0 {
		t.Fatal("timing log emitted before response body close")
	}
	if err := response.Body.Close(); err != nil {
		t.Fatal(err)
	}
	if err := response.Body.Close(); err != nil {
		t.Fatal(err)
	}
	closeTimingTestLogs(t, adapter)
	lines := strings.Split(strings.TrimSpace(logs.String()), "\n")
	if len(lines) != 1 || !strings.Contains(lines[0], `"msg":"build_request_timing"`) {
		t.Fatalf("expected one timing record; got %q", logs.String())
	}
	for _, secret := range []string{"private-input-sentinel", "private-output-sentinel", "private-cookie-sentinel", "private-access-sentinel", "synthetic.invalid"} {
		if strings.Contains(logs.String(), secret) {
			t.Fatalf("timing leaked %s", secret)
		}
	}
	var record struct {
		RequestID string `json:"request_id"`
		Timing    struct {
			Calls []struct {
				HTTPReturnedMS  *float64 `json:"http_returned_ms"`
				FirstBodyReadMS *float64 `json:"first_body_read_ms"`
				FirstTextMS     *float64 `json:"first_text_ms"`
			} `json:"calls"`
		} `json:"timing"`
	}
	if err := json.Unmarshal([]byte(lines[0]), &record); err != nil {
		t.Fatal(err)
	}
	if record.RequestID != "timing-test-1" || len(record.Timing.Calls) != 1 {
		t.Fatalf("missing correlation/call: %+v", record)
	}
	call := record.Timing.Calls[0]
	if call.HTTPReturnedMS == nil || call.FirstBodyReadMS == nil || call.FirstTextMS == nil {
		t.Fatalf("missing phase timestamps: %+v", call)
	}
	if *call.FirstTextMS < *call.HTTPReturnedMS {
		t.Fatal("first text preceded header handoff")
	}
}

func TestBuildRequestTimingDisabledProducesNoRecord(t *testing.T) {
	adapter, token := newTimingTestAdapter(t)
	var logs bytes.Buffer
	adapter.SetLogger(slog.New(slog.NewJSONHandler(&logs, nil)))
	adapter.http.Transport = roundTripFunc(func(req *http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: 200, Header: http.Header{}, Body: io.NopCloser(strings.NewReader("data: [DONE]\n\n")), Request: req}, nil
	})
	response, err := adapter.ForwardResponse(context.Background(), provider.ResponseResourceRequest{Credential: account.Credential{ID: 1, Provider: account.ProviderBuild, EncryptedAccessToken: token}, Method: http.MethodPost, Path: "/responses", Model: "grok-4.6", Streaming: true, Operation: "responses"})
	if err != nil {
		t.Fatal(err)
	}
	_, _ = io.Copy(io.Discard, response.Body)
	_ = response.Body.Close()
	closeTimingTestLogs(t, adapter)
	if adapter.timingLogs.queue != nil {
		t.Fatal("disabled timing started a writer")
	}
	if logs.Len() != 0 {
		t.Fatalf("disabled timing logged: %q", logs.String())
	}
}

func newTimingTestAdapter(t *testing.T) (*Adapter, string) {
	t.Helper()
	cipher, err := security.NewCipher(base64.StdEncoding.EncodeToString(make([]byte, 32)))
	if err != nil {
		t.Fatal(err)
	}
	token, err := cipher.Encrypt("private-access-sentinel")
	if err != nil {
		t.Fatal(err)
	}
	adapter := NewAdapter(Config{BaseURL: "https://synthetic.invalid/v1"}, cipher)
	t.Cleanup(func() { closeTimingTestLogs(t, adapter) })
	return adapter, token
}

func TestBuildRequestTimingRecordsTransportFailureWithoutErrorText(t *testing.T) {
	adapter, token := newTimingTestAdapter(t)
	cfg := adapter.config()
	cfg.RequestTimingEnabled = true
	adapter.UpdateConfig(cfg)
	var logs bytes.Buffer
	adapter.SetLogger(slog.New(slog.NewJSONHandler(&logs, nil)))
	adapter.http.Transport = roundTripFunc(func(*http.Request) (*http.Response, error) { return nil, io.ErrUnexpectedEOF })
	response, err := adapter.ForwardResponse(context.Background(), provider.ResponseResourceRequest{RequestID: "timing-failure-1", Credential: account.Credential{ID: 1, Provider: account.ProviderBuild, EncryptedAccessToken: token}, Method: http.MethodPost, Path: "/responses", Model: "grok-4.6", Streaming: true, Operation: "responses"})
	if err == nil || response != nil {
		t.Fatal("transport failure changed")
	}
	closeTimingTestLogs(t, adapter)
	var record struct {
		ForwardFailed bool `json:"forward_failed"`
		Timing        struct {
			Calls []struct {
				HTTPReturnedMS  *float64 `json:"http_returned_ms"`
				FirstBodyReadMS *float64 `json:"first_body_read_ms"`
			} `json:"calls"`
		} `json:"timing"`
	}
	if err := json.Unmarshal(logs.Bytes(), &record); err != nil {
		t.Fatal(err)
	}
	if !record.ForwardFailed || len(record.Timing.Calls) != 1 || record.Timing.Calls[0].HTTPReturnedMS == nil || record.Timing.Calls[0].FirstBodyReadMS != nil {
		t.Fatalf("incorrect failure observation: %+v", record)
	}
	if strings.Contains(logs.String(), "synthetic.invalid") || strings.Contains(logs.String(), "unexpected EOF") {
		t.Fatal("raw transport error leaked")
	}
}

func closeTimingTestLogs(t *testing.T, adapter *Adapter) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if err := adapter.CloseRequestTimingLogs(ctx); err != nil {
		t.Errorf("close timing logs: %v", err)
	}
}

type channelSyncStream struct {
	readEntered chan struct{}
	releaseData chan struct{}
	data        []byte
	closeCalled chan struct{}
	readOnce    sync.Once
	closeOnce   sync.Once
	mu          sync.Mutex
	offset      int
}

func (s *channelSyncStream) Read(p []byte) (int, error) {
	s.readOnce.Do(func() {
		close(s.readEntered)
	})
	<-s.releaseData
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.offset >= len(s.data) {
		return 0, io.EOF
	}
	n := copy(p, s.data[s.offset:])
	s.offset += n
	return n, nil
}

func (s *channelSyncStream) Close() error {
	s.closeOnce.Do(func() {
		close(s.closeCalled)
	})
	return nil
}

func TestBuildRequestTimingIdentityStreamChannelSync(t *testing.T) {
	adapter, token := newTimingTestAdapter(t)
	cfg := adapter.config()
	cfg.RequestTimingEnabled = true
	adapter.UpdateConfig(cfg)

	stream := &channelSyncStream{
		readEntered: make(chan struct{}),
		releaseData: make(chan struct{}),
		data:        []byte("event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"synced-hello\"}\n\n"),
		closeCalled: make(chan struct{}),
	}

	adapter.http.Transport = roundTripFunc(func(req *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: 200,
			Status:     "200 OK",
			Proto:      "HTTP/1.1",
			ProtoMajor: 1,
			Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
			Body:       stream,
			Request:    req,
		}, nil
	})

	ctx, root := infraegress.WithRequestTiming(context.Background())
	response, err := adapter.ForwardResponse(ctx, provider.ResponseResourceRequest{
		RequestID:  "identity-sync-1",
		Credential: account.Credential{ID: 1, Provider: account.ProviderBuild, EncryptedAccessToken: token},
		Method:     http.MethodPost,
		Path:       "/responses",
		Model:      "grok-4.6",
		Streaming:  true,
		Operation:  "responses",
	})
	if err != nil {
		t.Fatal(err)
	}

	// Response headers handed off; Read has not been entered yet.
	preSnapshot := root.Snapshot().Calls[0]
	if preSnapshot.FirstRawBodyReadStartMS != nil || preSnapshot.FirstDecodedBodyReadStartMS != nil {
		t.Fatal("read start recorded before read invocation")
	}
	if preSnapshot.ContentEncoding != "identity" {
		t.Fatalf("expected identity content encoding, got %q", preSnapshot.ContentEncoding)
	}
	if preSnapshot.Protocol != "http1" {
		t.Fatalf("expected http1 protocol, got %q", preSnapshot.Protocol)
	}

	// Begin reading in a background goroutine. It will enter Read and block waiting on releaseData.
	readResult := make(chan struct {
		n   int
		err error
		buf []byte
	}, 1)
	go func() {
		buf := make([]byte, 1024)
		n, rErr := response.Body.Read(buf)
		readResult <- struct {
			n   int
			err error
			buf []byte
		}{n: n, err: rErr, buf: buf[:n]}
	}()

	// Wait until Read has actively entered the underlying reader.
	select {
	case <-stream.readEntered:
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for read entry")
	}

	// Read has entered, but data is NOT yet released.
	// Verify that entry timestamps are recorded while byte timestamps are NOT recorded.
	duringSnapshot := root.Snapshot().Calls[0]
	if duringSnapshot.FirstDecodedBodyReadStartMS == nil {
		t.Fatal("decoded body read start not recorded on entry")
	}
	if duringSnapshot.FirstRawBodyReadStartMS == nil {
		t.Fatal("raw body read start not recorded on entry")
	}
	if duringSnapshot.FirstRawBodyReadDoneMS != nil {
		t.Fatal("raw body read done recorded before data released")
	}
	if duringSnapshot.FirstBodyByteMS != nil {
		t.Fatal("decoded body read done recorded before data released")
	}

	// Now release data to the blocked reader.
	close(stream.releaseData)

	select {
	case res := <-readResult:
		if res.err != nil && res.err != io.EOF {
			t.Fatalf("read failed: %v", res.err)
		}
		remaining, rErr := io.ReadAll(response.Body)
		if rErr != nil {
			t.Fatalf("read remaining failed: %v", rErr)
		}
		full := append(res.buf, remaining...)
		if !bytes.Equal(full, stream.data) {
			t.Fatalf("data mismatch: got %q, want %q", full, stream.data)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for read completion")
	}

	// After bytes returned, both return timestamps are recorded.
	postSnapshot := root.Snapshot().Calls[0]
	if postSnapshot.FirstRawBodyReadDoneMS == nil {
		t.Fatal("raw body read done not recorded after data return")
	}
	if postSnapshot.FirstBodyByteMS == nil {
		t.Fatal("decoded body read done not recorded after data return")
	}
	if *postSnapshot.FirstRawBodyReadStartMS > *postSnapshot.FirstRawBodyReadDoneMS {
		t.Fatal("raw body read start after raw body read done")
	}
	if *postSnapshot.FirstDecodedBodyReadStartMS > *postSnapshot.FirstBodyByteMS {
		t.Fatal("decoded body read start after decoded body read done")
	}

	// Close response body and verify Close called on source.
	if err := response.Body.Close(); err != nil {
		t.Fatal(err)
	}
	select {
	case <-stream.closeCalled:
	case <-time.After(time.Second):
		t.Fatal("close was not propagated to underlying body")
	}
}

func TestBuildRequestTimingGzipNormalizationAndErrorHandling(t *testing.T) {
	adapter, token := newTimingTestAdapter(t)
	cfg := adapter.config()
	cfg.RequestTimingEnabled = true
	adapter.UpdateConfig(cfg)

	t.Run("successful_gzip", func(t *testing.T) {
		plainText := "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"gzip-payload\"}\n\n"
		var gzipBuf bytes.Buffer
		gw := gzip.NewWriter(&gzipBuf)
		if _, err := gw.Write([]byte(plainText)); err != nil {
			t.Fatal(err)
		}
		if err := gw.Close(); err != nil {
			t.Fatal(err)
		}

		adapter.http.Transport = roundTripFunc(func(req *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: 200,
				Status:     "200 OK",
				Proto:      "HTTP/2.0",
				ProtoMajor: 2,
				Header:     http.Header{"Content-Type": []string{"text/event-stream"}, "Content-Encoding": []string{"gzip"}},
				Body:       io.NopCloser(bytes.NewReader(gzipBuf.Bytes())),
				Request:    req,
			}, nil
		})

		ctx, root := infraegress.WithRequestTiming(context.Background())
		response, err := adapter.ForwardResponse(ctx, provider.ResponseResourceRequest{
			RequestID:  "gzip-success",
			Credential: account.Credential{ID: 1, Provider: account.ProviderBuild, EncryptedAccessToken: token},
			Method:     http.MethodPost,
			Path:       "/responses",
			Model:      "grok-4.6",
			Streaming:  true,
			Operation:  "responses",
		})
		if err != nil {
			t.Fatal(err)
		}

		// Gzip initialization occurred during ForwardResponse (inside normalizeTimedBuildResponse).
		// Raw read start/done happened during gzip.NewReader reading the header!
		s := root.Snapshot().Calls[0]
		if s.ContentEncoding != "gzip" {
			t.Fatalf("expected gzip content encoding, got %q", s.ContentEncoding)
		}
		if s.Protocol != "http2" {
			t.Fatalf("expected http2, got %q", s.Protocol)
		}
		if s.GzipInitStartMS == nil || s.GzipInitEndMS == nil {
			t.Fatalf("gzip init timestamps missing: start=%v end=%v", s.GzipInitStartMS, s.GzipInitEndMS)
		}
		if *s.GzipInitStartMS > *s.GzipInitEndMS {
			t.Fatal("gzip init start after end")
		}
		if s.FirstRawBodyReadStartMS == nil || s.FirstRawBodyReadDoneMS == nil {
			t.Fatalf("raw body read not observed during gzip init: start=%v done=%v", s.FirstRawBodyReadStartMS, s.FirstRawBodyReadDoneMS)
		}
		if s.FirstDecodedBodyReadStartMS != nil || s.FirstBodyByteMS != nil {
			t.Fatal("decoded body read observed before decoded read started")
		}

		// Read decoded body
		decoded, err := io.ReadAll(response.Body)
		if err != nil {
			t.Fatal(err)
		}
		if string(decoded) != plainText {
			t.Fatalf("decoded mismatch: got %q, want %q", string(decoded), plainText)
		}
		_ = response.Body.Close()

		post := root.Snapshot().Calls[0]
		if post.FirstDecodedBodyReadStartMS == nil || post.FirstBodyByteMS == nil {
			t.Fatalf("decoded body timestamps missing: %+v", post)
		}
	})

	t.Run("source_failing_during_gzip_header", func(t *testing.T) {
		var closed atomic.Bool
		simulatedErr := errors.New("underlying header read failure")
		failingSource := &fnReadCloser{
			read: func(p []byte) (int, error) {
				if len(p) >= 2 {
					copy(p, []byte{0x1f, 0x8b}) // gzip magic only (not enough for 10-byte header)
					return 2, simulatedErr
				}
				return 0, simulatedErr
			},
			close: func() error {
				closed.Store(true)
				return nil
			},
		}

		adapter.http.Transport = roundTripFunc(func(req *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: 200,
				Status:     "200 OK",
				Proto:      "HTTP/1.1",
				ProtoMajor: 1,
				Header:     http.Header{"Content-Encoding": []string{"gzip"}},
				Body:       failingSource,
				Request:    req,
			}, nil
		})

		ctx, root := infraegress.WithRequestTiming(context.Background())
		response, err := adapter.ForwardResponse(ctx, provider.ResponseResourceRequest{
			RequestID:  "gzip-failure",
			Credential: account.Credential{ID: 1, Provider: account.ProviderBuild, EncryptedAccessToken: token},
			Method:     http.MethodPost,
			Path:       "/responses",
			Model:      "grok-4.6",
			Streaming:  true,
			Operation:  "responses",
		})
		if err == nil || response != nil {
			t.Fatal("expected forward failure on invalid gzip header")
		}
		if !closed.Load() {
			t.Fatal("source body was not closed on gzip header failure")
		}

		s := root.Snapshot().Calls[0]
		if s.ContentEncoding != "gzip" {
			t.Fatalf("expected gzip, got %q", s.ContentEncoding)
		}
		if s.GzipInitStartMS == nil || s.GzipInitEndMS == nil {
			t.Fatalf("expected gzip init start and end recorded: %+v", s)
		}
		if s.FirstRawBodyReadStartMS == nil {
			t.Fatal("expected raw body read start recorded during gzip init")
		}
		// Decoded body was never wrapped or read because gzip init failed.
		if s.FirstDecodedBodyReadStartMS != nil || s.FirstBodyByteMS != nil {
			t.Fatal("decoded body read recorded on failed gzip init")
		}
	})
}

func TestBuildRequestTimingEmptyReadsAndDisabledTiming(t *testing.T) {
	adapter, token := newTimingTestAdapter(t)
	cfg := adapter.config()
	cfg.RequestTimingEnabled = true
	adapter.UpdateConfig(cfg)

	t.Run("empty_stream_does_not_pretend_bytes_arrived", func(t *testing.T) {
		adapter.http.Transport = roundTripFunc(func(req *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: 200,
				Status:     "200 OK",
				Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
				Body:       io.NopCloser(strings.NewReader("")),
				Request:    req,
			}, nil
		})

		ctx, root := infraegress.WithRequestTiming(context.Background())
		response, err := adapter.ForwardResponse(ctx, provider.ResponseResourceRequest{
			RequestID:  "empty-1",
			Credential: account.Credential{ID: 1, Provider: account.ProviderBuild, EncryptedAccessToken: token},
			Method:     http.MethodPost,
			Path:       "/responses",
			Model:      "grok-4.6",
			Streaming:  true,
			Operation:  "responses",
		})
		if err != nil {
			t.Fatal(err)
		}

		buf := make([]byte, 128)
		n, rErr := response.Body.Read(buf)
		if n != 0 || rErr != io.EOF {
			t.Fatalf("expected (0, EOF), got (%d, %v)", n, rErr)
		}
		_ = response.Body.Close()

		s := root.Snapshot().Calls[0]
		if s.FirstDecodedBodyReadStartMS == nil || s.FirstRawBodyReadStartMS == nil {
			t.Fatal("read start not recorded on empty stream")
		}
		if s.FirstRawBodyReadDoneMS != nil || s.FirstBodyByteMS != nil {
			t.Fatalf("empty read pretended bytes arrived: rawDone=%v decodedDone=%v", s.FirstRawBodyReadDoneMS, s.FirstBodyByteMS)
		}
	})

	t.Run("disabled_timing_wraps_nothing_and_starts_no_queue", func(t *testing.T) {
		disabledAdapter, disToken := newTimingTestAdapter(t)
		// RequestTimingEnabled is false by default
		disabledAdapter.http.Transport = roundTripFunc(func(req *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: 200,
				Status:     "200 OK",
				Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
				Body:       io.NopCloser(strings.NewReader("data: [DONE]\n\n")),
				Request:    req,
			}, nil
		})

		response, err := disabledAdapter.ForwardResponse(context.Background(), provider.ResponseResourceRequest{
			RequestID:  "disabled-1",
			Credential: account.Credential{ID: 1, Provider: account.ProviderBuild, EncryptedAccessToken: disToken},
			Method:     http.MethodPost,
			Path:       "/responses",
			Model:      "grok-4.6",
			Streaming:  true,
			Operation:  "responses",
		})
		if err != nil {
			t.Fatal(err)
		}

		_, _ = io.ReadAll(response.Body)
		_ = response.Body.Close()

		if disabledAdapter.timingLogs.queue != nil {
			t.Fatal("disabled timing started a queue")
		}
	})
}

func TestBuildRequestTimingFiniteMetadataSecretRedaction(t *testing.T) {
	adapter, token := newTimingTestAdapter(t)
	cfg := adapter.config()
	cfg.RequestTimingEnabled = true
	adapter.UpdateConfig(cfg)

	var logs bytes.Buffer
	adapter.SetLogger(slog.New(slog.NewJSONHandler(&logs, nil)))

	adapter.http.Transport = roundTripFunc(func(req *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: 200,
			Status:     "200 OK",
			Proto:      "SECRET-PROTOCOL-V9",
			ProtoMajor: 9,
			Header: http.Header{
				"Content-Encoding": []string{"secret-compress-sentinel"},
				"Content-Type":     []string{"text/event-stream"},
			},
			Uncompressed: true,
			Body:         io.NopCloser(strings.NewReader("data: [DONE]\n\n")),
			Request:      req,
		}, nil
	})

	response, err := adapter.ForwardResponse(context.Background(), provider.ResponseResourceRequest{
		RequestID:  "finite-metadata-1",
		Credential: account.Credential{ID: 1, Provider: account.ProviderBuild, EncryptedAccessToken: token},
		Method:     http.MethodPost,
		Path:       "/responses",
		Model:      "grok-4.6",
		Streaming:  true,
		Operation:  "responses",
	})
	if err != nil {
		t.Fatal(err)
	}
	_, _ = io.ReadAll(response.Body)
	_ = response.Body.Close()
	closeTimingTestLogs(t, adapter)

	logStr := logs.String()
	if strings.Contains(logStr, "secret-compress-sentinel") {
		t.Fatal("secret encoding leaked into logs")
	}
	if strings.Contains(logStr, "SECRET-PROTOCOL") {
		t.Fatal("secret protocol leaked into logs")
	}
	if !strings.Contains(logStr, `"content_encoding":"other"`) {
		t.Fatalf("expected content_encoding other in log, got: %s", logStr)
	}
	if !strings.Contains(logStr, `"protocol":"other"`) {
		t.Fatalf("expected protocol other in log, got: %s", logStr)
	}
	if !strings.Contains(logStr, `"uncompressed":true`) {
		t.Fatalf("expected uncompressed true in log, got: %s", logStr)
	}
}

func TestBuildRequestTimingConcurrentSnapshotReadCloseRaces(t *testing.T) {
	adapter, token := newTimingTestAdapter(t)
	cfg := adapter.config()
	cfg.RequestTimingEnabled = true
	adapter.UpdateConfig(cfg)

	for iter := 0; iter < 5; iter++ {
		pr, pw := io.Pipe()
		adapter.http.Transport = roundTripFunc(func(req *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: 200,
				Status:     "200 OK",
				Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
				Body:       pr,
				Request:    req,
			}, nil
		})

		ctx, root := infraegress.WithRequestTiming(context.Background())
		response, err := adapter.ForwardResponse(ctx, provider.ResponseResourceRequest{
			RequestID:  "concurrent-race",
			Credential: account.Credential{ID: 1, Provider: account.ProviderBuild, EncryptedAccessToken: token},
			Method:     http.MethodPost,
			Path:       "/responses",
			Model:      "grok-4.6",
			Streaming:  true,
			Operation:  "responses",
		})
		if err != nil {
			t.Fatal(err)
		}

		go func() {
			for j := 0; j < 5; j++ {
				_, _ = io.WriteString(pw, "data: {\"type\":\"response.output_text.delta\",\"delta\":\"chunk\"}\n\n")
				time.Sleep(time.Millisecond)
			}
			_ = pw.Close()
		}()

		var wg sync.WaitGroup
		// Goroutine reading
		wg.Add(1)
		go func() {
			defer wg.Done()
			buf := make([]byte, 64)
			for {
				n, rErr := response.Body.Read(buf)
				if n > 0 {
					_ = root.Snapshot()
				}
				if rErr != nil {
					break
				}
			}
		}()

		// Goroutines snapshotting
		for k := 0; k < 3; k++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for m := 0; m < 10; m++ {
					_ = root.Snapshot()
				}
			}()
		}

		wg.Wait()
		_ = response.Body.Close()
	}
}

type fnReadCloser struct {
	read  func(p []byte) (int, error)
	close func() error
}

func (f *fnReadCloser) Read(p []byte) (int, error) {
	if f.read != nil {
		return f.read(p)
	}
	return 0, io.EOF
}

func (f *fnReadCloser) Close() error {
	if f.close != nil {
		return f.close()
	}
	return nil
}
