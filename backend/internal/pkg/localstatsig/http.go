package localstatsig

import (
	"bytes"
	"crypto/rand"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"time"
)

const maxRequestBytes = 8192

// NewHandler exposes the Grok2API signer contract without logging inputs or
// signatures. /readyz checks configuration validity, not upstream acceptance.
// Serve it only on loopback or a private Docker network.
func NewHandler(path string) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]bool{"alive": true})
	})
	mux.HandleFunc("GET /readyz", func(w http.ResponseWriter, r *http.Request) {
		if _, err := Load(path, time.Now()); err != nil {
			writeError(w, http.StatusServiceUnavailable, ErrUnavailable)
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"ready": true, "scope": "configuration"})
	})
	mux.HandleFunc("POST /sign", func(w http.ResponseWriter, r *http.Request) {
		r.Body = http.MaxBytesReader(w, r.Body, maxRequestBytes)
		body, err := io.ReadAll(r.Body)
		if err != nil {
			var maxErr *http.MaxBytesError
			if errors.As(err, &maxErr) {
				writeError(w, http.StatusRequestEntityTooLarge, ErrInvalidRequest)
			} else {
				writeError(w, http.StatusBadRequest, ErrInvalidRequest)
			}
			return
		}
		var req struct {
			Method      string `json:"method"`
			Path        string `json:"path"`
			Environment struct {
				MetaContent string `json:"metaContent"`
			} `json:"environment"`
		}
		d := json.NewDecoder(bytes.NewReader(body))
		d.DisallowUnknownFields()
		if d.Decode(&req) != nil || d.Decode(new(any)) != io.EOF || ValidateRequest(req.Method, req.Path, req.Environment.MetaContent) != nil {
			writeError(w, http.StatusBadRequest, ErrInvalidRequest)
			return
		}
		now := time.Now()
		cfg, err := Load(path, now)
		if err != nil {
			writeError(w, http.StatusServiceUnavailable, ErrUnavailable)
			return
		}
		signature, err := cfg.Sign(req.Method, req.Path, req.Environment.MetaContent, now, rand.Reader)
		if err != nil {
			writeError(w, http.StatusServiceUnavailable, err)
			return
		}
		writeJSON(w, http.StatusOK, map[string]string{"x-statsig-id": signature})
	})
	return mux
}

func writeError(w http.ResponseWriter, status int, err error) {
	writeJSON(w, status, map[string]string{"detail": err.Error()})
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}
