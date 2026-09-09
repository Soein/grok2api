package app

import (
	"context"
	"testing"

	"github.com/chenyme/grok2api/backend/internal/application/accountrecovery"
)

func TestUnifiedRecoveryDoesNotQueueLegacyStartupProbes(t *testing.T) {
	// Nil legacy dependencies make any attempt to bypass the unified scheduler
	// fail immediately instead of quietly spawning a second recovery queue.
	a := &Application{accountRecovery: &accountrecovery.Service{}}
	a.queueDueWebQuotaRefresh(context.Background())
}
