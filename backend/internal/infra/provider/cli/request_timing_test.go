package cli

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/chenyme/grok2api/backend/internal/domain/account"
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
