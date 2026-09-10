package web

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/chenyme/grok2api/backend/internal/domain/account"
	infraegress "github.com/chenyme/grok2api/backend/internal/infra/egress"
	"github.com/chenyme/grok2api/backend/internal/infra/provider"
	"github.com/chenyme/grok2api/backend/internal/infra/security"
)

func TestGenerateVideoStopsOnSigningFailure(t *testing.T) {
	for _, test := range []struct {
		name         string
		signerStatus int
		rejectCount  int
		wantPosts    int
		wantStatus   int
		rejectStatus int
		rejectBody   string
		wantStage    provider.VideoStage
	}{
		{name: "unavailable signer", signerStatus: http.StatusServiceUnavailable, wantStatus: http.StatusServiceUnavailable},
		{name: "rejected signature refresh unavailable", signerStatus: http.StatusBadGateway, rejectCount: 1, wantPosts: 1, wantStatus: http.StatusServiceUnavailable},
		{name: "fresh signature still rejected", signerStatus: http.StatusOK, rejectCount: 2, wantPosts: 2, wantStatus: http.StatusForbidden},
		{name: "cloudflare remains retryable", signerStatus: http.StatusOK, rejectCount: 1, wantPosts: 1, wantStatus: http.StatusForbidden, rejectStatus: http.StatusForbidden, rejectBody: `<html><title>Just a moment...</title>cf-chl- challenge</html>`, wantStage: provider.VideoStageCreate},
		{name: "unauthorized remains account scoped", signerStatus: http.StatusOK, rejectCount: 1, wantPosts: 1, wantStatus: http.StatusUnauthorized, rejectStatus: http.StatusUnauthorized, rejectBody: `{"error":"unauthorized"}`, wantStage: provider.VideoStageCreate},
		{name: "quota rejection remains retryable", signerStatus: http.StatusOK, rejectCount: 1, wantPosts: 1, wantStatus: http.StatusTooManyRequests, rejectStatus: http.StatusTooManyRequests, rejectBody: `{"error":"quota exhausted"}`, wantStage: provider.VideoStageCreate},
	} {
		t.Run(test.name, func(t *testing.T) {
			if test.rejectStatus == 0 {
				test.rejectStatus = http.StatusForbidden
			}
			if test.rejectBody == "" {
				test.rejectBody = `{"code":7,"message":"This page is out of date. Reload to continue."}`
			}
			if test.wantStage == "" {
				test.wantStage = provider.VideoStagePrepare
			}
			posts, signs := 0, 0
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/sign":
					signs++
					if test.signerStatus != http.StatusOK && (test.rejectCount == 0 || signs > 1) {
						w.WriteHeader(test.signerStatus)
						return
					}
					_ = json.NewEncoder(w).Encode(map[string]string{"x-statsig-id": base64.RawStdEncoding.EncodeToString(bytes.Repeat([]byte{byte(signs)}, 70))})
				case "/rest/app-chat/conversations/new":
					posts++
					if posts <= test.rejectCount {
						w.WriteHeader(test.rejectStatus)
						_, _ = io.WriteString(w, test.rejectBody)
						return
					}
					_, _ = io.WriteString(w, `data: {"result":{"response":{"streamingVideoGenerationResponse":{"progress":100,"videoUrl":"/videos/final.mp4"}}}}`+"\n\n")
				default:
					http.NotFound(w, r)
				}
			}))
			defer server.Close()
			cipher, err := security.NewCipher(base64.StdEncoding.EncodeToString(make([]byte, 32)))
			if err != nil {
				t.Fatal(err)
			}
			encrypted, err := cipher.Encrypt("test-sso")
			if err != nil {
				t.Fatal(err)
			}
			adapter := NewAdapter(Config{BaseURL: server.URL, StatsigMode: "url", StatsigSignerURL: server.URL + "/sign", VideoTimeoutSeconds: 5}, infraegress.NewManager(egressRepositoryStub{}, cipher), cipher, nil, nil)
			adapter.statsig.fetchMeta = func(context.Context, string, string, *infraegress.Lease) (string, error) { return "page-meta", nil }
			adapter.statsig.validateEndpoint = func(context.Context, string) error { return nil }
			_, err = adapter.GenerateVideo(context.Background(), provider.VideoRequest{Credential: account.Credential{ID: 1, Provider: account.ProviderWeb, EncryptedAccessToken: encrypted}, Prompt: "test", Duration: 5})
			if posts != test.wantPosts {
				t.Errorf("generation POSTs = %d, want %d", posts, test.wantPosts)
			}
			stage, ok := provider.VideoErrorStage(err)
			if !ok || stage != test.wantStage {
				t.Errorf("stage = %q, err = %v; want %s", stage, err, test.wantStage)
			}
			if test.wantStatus == http.StatusUnauthorized {
				if !errors.Is(err, provider.ErrUnauthorized) {
					t.Errorf("err = %v; want unauthorized sentinel", err)
				}
				return
			}
			if status, ok := provider.ErrorHTTPStatus(err); !ok || status != test.wantStatus {
				t.Errorf("status = %d, err = %v; want %d", status, err, test.wantStatus)
			}
		})
	}
}
