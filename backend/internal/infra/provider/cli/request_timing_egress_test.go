package cli

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/chenyme/grok2api/backend/internal/domain/account"
	domainegress "github.com/chenyme/grok2api/backend/internal/domain/egress"
	infraegress "github.com/chenyme/grok2api/backend/internal/infra/egress"
	"github.com/chenyme/grok2api/backend/internal/repository"
)

type timingNodeRepository struct {
	repository.EgressRepository
	node  domainegress.Node
	delay time.Duration
}

func (r *timingNodeRepository) GetEgressNode(ctx context.Context, id uint64) (domainegress.Node, error) {
	timer := time.NewTimer(r.delay)
	defer timer.Stop()
	select {
	case <-timer.C:
		return r.node, nil
	case <-ctx.Done():
		return domainegress.Node{}, ctx.Err()
	}
}

func TestBuildRequestTimingSeparatesAcquireAndFeedbackFromHTTP(t *testing.T) {
	proxy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer proxy.Close()
	adapter, _ := newTimingTestAdapter(t)
	encrypted, err := adapter.cipher.Encrypt(proxy.URL)
	if err != nil {
		t.Fatal(err)
	}
	repo := &timingNodeRepository{node: domainegress.Node{ID: 1, Enabled: true, Scope: domainegress.ScopeBuild, EncryptedProxyURL: encrypted, Health: 1}, delay: 15 * time.Millisecond}
	manager := infraegress.NewManager(repo, adapter.cipher)
	ctx, root := infraegress.WithRequestTiming(context.Background())
	ctx, call := infraegress.BeginTimingCall(ctx)
	ctx = infraegress.WithCredential(ctx, account.Credential{ID: 1, Provider: account.ProviderBuild, EgressNodeID: 1})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, "http://synthetic.invalid/v1/responses", strings.NewReader(`{}`))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Accept", "text/event-stream")
	transport := &egressTransport{manager: manager, fallback: http.DefaultTransport}
	response, err := transport.RoundTrip(req)
	if err != nil {
		t.Fatal(err)
	}
	call.MarkHTTPReturned()
	_, _ = io.Copy(io.Discard, response.Body)
	_ = response.Body.Close()
	s := root.Snapshot().Calls[0]
	if s.AcquireCount != 1 || s.FeedbackCount != 1 || s.AcquireMS < 14 || s.FeedbackMS < 14 {
		t.Fatalf("local delays not isolated: %+v", s)
	}
	if s.TransportReturnedMS == nil || s.HTTPReturnedMS == nil || *s.HTTPReturnedMS-*s.TransportReturnedMS < 14 {
		t.Fatalf("headers/feedback boundary missing: %+v", s)
	}
}
