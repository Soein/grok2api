package localstatsig

import (
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func request(t *testing.T, h http.Handler, method, path, body string) *httptest.ResponseRecorder {
	t.Helper()
	r := httptest.NewRequest(method, path, strings.NewReader(body))
	r.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	return w
}

func requestBody(meta string) string {
	b, _ := json.Marshal(map[string]any{"method": "POST", "path": "/rest/app-chat/conversations/new", "environment": map[string]string{"metaContent": meta}})
	return string(b)
}

func TestHTTPHealthAndReadyAreDistinct(t *testing.T) {
	path := filepath.Join(t.TempDir(), "params.json")
	h := NewHandler(path)
	if w := request(t, h, "GET", "/healthz", ""); w.Code != 200 {
		t.Fatalf("health: %d", w.Code)
	}
	if w := request(t, h, "GET", "/readyz", ""); w.Code != 503 {
		t.Fatalf("missing config ready: %d", w.Code)
	}
	cfg := fixtureConfig()
	cfg.ExpiresAt = time.Now().Add(time.Hour)
	writeConfig(t, path, cfg)
	if w := request(t, h, "GET", "/readyz", ""); w.Code != 200 || !strings.Contains(w.Body.String(), "configuration") {
		t.Fatalf("ready: %d %s", w.Code, w.Body)
	}
	cfg.ExpiresAt = time.Now().Add(-time.Second)
	writeConfig(t, path, cfg)
	if w := request(t, h, "GET", "/readyz", ""); w.Code != 503 {
		t.Fatalf("expired config ready: %d", w.Code)
	}
}

func TestHTTPCompatibleResponseAndFailClosedReload(t *testing.T) {
	path := filepath.Join(t.TempDir(), "params.json")
	h := NewHandler(path)
	cfg := fixtureConfig()
	cfg.ExpiresAt = time.Now().Add(time.Hour)
	writeConfig(t, path, cfg)
	w := request(t, h, "POST", "/sign", requestBody(fixtureMeta))
	if w.Code != 200 {
		t.Fatalf("sign: %d %s", w.Code, w.Body)
	}
	var result map[string]string
	if err := json.Unmarshal(w.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	decoded, err := base64.RawStdEncoding.DecodeString(result["x-statsig-id"])
	if err != nil || len(decoded) != 70 {
		t.Fatalf("incompatible response: %v length=%d", err, len(decoded))
	}
	if w.Header().Get("Cache-Control") != "no-store" {
		t.Fatal("signature may be cached")
	}
	other := base64.StdEncoding.EncodeToString([]byte(strings.Repeat("x", 48)))
	w = request(t, h, "POST", "/sign", requestBody(other))
	if w.Code != 503 || strings.Contains(w.Body.String(), fixtureMeta) || strings.Contains(w.Body.String(), other) {
		t.Fatalf("mismatch: %d %s", w.Code, w.Body)
	}
	if err := os.WriteFile(path, []byte("invalid"), 0600); err != nil {
		t.Fatal(err)
	}
	w = request(t, h, "POST", "/sign", requestBody(fixtureMeta))
	if w.Code != 503 || strings.Contains(w.Body.String(), "x-statsig-id") {
		t.Fatalf("invalid update signed: %d %s", w.Code, w.Body)
	}
}

func TestHTTPRejectsMalformedAndOversizedRequests(t *testing.T) {
	path := filepath.Join(t.TempDir(), "params.json")
	h := NewHandler(path)
	cfg := fixtureConfig()
	cfg.ExpiresAt = time.Now().Add(time.Hour)
	writeConfig(t, path, cfg)
	for _, tc := range []struct {
		name, body string
		status     int
	}{
		{"malformed", "{", 400},
		{"missing", "{}", 400},
		{"multiple", requestBody(fixtureMeta) + "{}", 400},
		{"unknown-field", `{"method":"POST","path":"/x","environment":{"metaContent":"x"},"cookie":"never-log-this"}`, 400},
		{"oversized", strings.Repeat(" ", maxRequestBytes+1), 413},
	} {
		t.Run(tc.name, func(t *testing.T) {
			w := request(t, h, "POST", "/sign", tc.body)
			if w.Code != tc.status {
				t.Fatalf("status=%d want=%d body=%s", w.Code, tc.status, w.Body)
			}
			if strings.Contains(w.Body.String(), "never-log-this") {
				t.Fatal("request echoed")
			}
		})
	}
	if w := request(t, h, "GET", "/sign", ""); w.Code != 405 {
		t.Fatalf("GET sign: %d", w.Code)
	}
}

func TestHTTPDynamicAcceptsChangingMetaAndRejectsInvalidTable(t *testing.T) {
	path := filepath.Join(t.TempDir(), "params.json")
	h := NewHandler(path)
	cfg := fixtureDynamic()
	cfg.ExpiresAt = time.Now().Add(time.Hour)
	writeConfig(t, path, cfg)
	for _, meta := range []string{fixtureMeta, base64.StdEncoding.EncodeToString([]byte(strings.Repeat("x", 48)))} {
		w := request(t, h, "POST", "/sign", requestBody(meta))
		if w.Code != 200 {
			t.Fatalf("dynamic sign: %d %s", w.Code, w.Body)
		}
		var result map[string]string
		_ = json.Unmarshal(w.Body.Bytes(), &result)
		raw, err := base64.RawStdEncoding.DecodeString(result["x-statsig-id"])
		if err != nil || len(raw) != 70 {
			t.Fatal("invalid dynamic signature format")
		}
		key := raw[0]
		for i := range raw {
			raw[i] ^= key
		}
		if base64.StdEncoding.EncodeToString(raw[1:49]) != meta {
			t.Fatal("dynamic signature header does not bind request meta")
		}
	}
	cfg.Fingerprints[3][15][181] = ""
	writeConfig(t, path, cfg)
	if w := request(t, h, "POST", "/sign", requestBody(fixtureMeta)); w.Code != 503 {
		t.Fatalf("incomplete dynamic replacement signed: %d", w.Code)
	}
}
