package cli

import (
	"bufio"
	"context"
	"encoding/base64"
	"fmt"
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

// A paused upstream must not prevent header handoff or a complete first text event.
// This diagnostic uses only synthetic data and never contacts an external service.
func TestBuildRequestTimingStreamsTextBeforeUpstreamEnd(t *testing.T) {
	for _, operation := range []string{"responses", "chat", "messages"} {
		t.Run(operation, func(t *testing.T) {
			cipher, err := security.NewCipher(base64.StdEncoding.EncodeToString(make([]byte, 32)))
			if err != nil {
				t.Fatal(err)
			}
			token, err := cipher.Encrypt("synthetic-token")
			if err != nil {
				t.Fatal(err)
			}
			source, emitter := io.Pipe()
			defer source.Close()
			defer emitter.Close()
			adapter := NewAdapter(Config{RequestTimingEnabled: true, BaseURL: "https://synthetic.invalid/v1", StreamIdleTimeout: time.Second}, cipher)
			t.Cleanup(func() { closeTimingTestLogs(t, adapter) })
			adapter.SetLogger(slog.New(slog.NewTextHandler(io.Discard, nil)))
			adapter.http.Transport = roundTripFunc(func(req *http.Request) (*http.Response, error) {
				if req.Header.Get("Accept-Encoding") != "identity" {
					return nil, fmt.Errorf("stream compression not disabled")
				}
				return &http.Response{StatusCode: 200, Status: "200 OK", Header: http.Header{"Content-Type": []string{"text/event-stream"}}, Body: source, Request: req}, nil
			})
			body := `{"model":"grok-4.6","input":"Say hello","stream":true}`
			if operation != "responses" {
				body = `{"model":"grok-4.6","messages":[{"role":"user","content":"Say hello"}],"max_tokens":64,"stream":true}`
			}
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()
			type result struct {
				response *provider.Response
				err      error
			}
			ready := make(chan result, 1)
			started := time.Now()
			go func() {
				r, e := adapter.ForwardResponse(ctx, provider.ResponseResourceRequest{Credential: account.Credential{ID: 1, Provider: account.ProviderBuild, EncryptedAccessToken: token}, Method: http.MethodPost, Path: "/responses", Model: "grok-4.6", PromptCacheKey: "synthetic-first-byte", Body: []byte(body), Streaming: true, NormalizeBody: true, Operation: operation})
				ready <- result{r, e}
			}()
			var response *provider.Response
			select {
			case r := <-ready:
				if r.err != nil {
					t.Fatal(r.err)
				}
				response = r.response
			case <-time.After(300 * time.Millisecond):
				t.Fatal("adapter waited for upstream body before handing off streaming response")
			}
			defer response.Body.Close()
			headersElapsed := time.Since(started)
			gotText := make(chan error, 1)
			go func() {
				scanner := bufio.NewScanner(response.Body)
				for scanner.Scan() {
					if strings.Contains(scanner.Text(), "FIRSTBYTEPROBE") {
						gotText <- nil
						return
					}
				}
				gotText <- fmt.Errorf("missing first text: %v", scanner.Err())
			}()
			sentAt := time.Now()
			go func() {
				_, _ = io.WriteString(emitter, "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_synthetic\",\"model\":\"grok-4.6\",\"created_at\":1,\"status\":\"in_progress\"}}\n\nevent: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"item_id\":\"msg_synthetic\",\"output_index\":0,\"content_index\":0,\"delta\":\"FIRSTBYTEPROBE\"}\n\n")
				// Keep upstream open: no completed event and no EOF until test cleanup.
			}()
			select {
			case e := <-gotText:
				if e != nil {
					t.Fatal(e)
				}
			case <-time.After(300 * time.Millisecond):
				t.Fatal("complete text delta was held while upstream remained open")
			}
			t.Logf("adapter_handoff=%s first_complete_text_event=%s before_terminal=true", headersElapsed, time.Since(sentAt))
		})
	}
}
