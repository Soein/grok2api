package app

import (
	"context"
	"log/slog"
	"testing"
	"time"

	"github.com/chenyme/grok2api/backend/internal/application/gateway"
)

func TestApplicationBuildBasePreRefreshSupervisedWorkerLifecycle(t *testing.T) {
	selector := gateway.NewSelector(nil, nil, nil, nil, time.Hour, time.Second, time.Minute)
	selector.UpdateBuildBasePreRefreshPolicy(true, 5*time.Second, 5*time.Second)

	app := &Application{
		selector: selector,
		logger:   slog.Default(),
	}

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})

	go func() {
		defer close(done)
		app.runSupervisedTask(ctx, "build_base_prerefresh", app.selector.RunBasePreRefresh)
	}()

	// Give worker a moment to enter RunBasePreRefresh ticker loop
	time.Sleep(50 * time.Millisecond)

	cancel()

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("supervised build_base_prerefresh worker did not exit cleanly within bounded time on context cancellation")
	}
}
