package repository

import (
	"context"
	"time"

	"github.com/chenyme/grok2api/backend/internal/domain/account"
)

// RecoveryActivationRevoker removes an explicit migration recovery grant before
// an administrator applies enablement, including a repeated disable operation.
type RecoveryActivationRevoker interface {
	RevokeRecoveryActivations(context.Context, account.Provider, []uint64) error
}

// RecoveryActivationRepository queues quota verification for an externally
// authorized, identity-bound account. Authentication recovery alone never enables
// business traffic. Only a successful claimed quota probe consumes the grant.
type RecoveryActivationRepository interface {
	RecoveryActivationRevoker
	ListPendingRecoveryActivationIDs(context.Context, int) ([]uint64, error)
	PrepareRecoveredActivationQuota(context.Context, uint64, time.Time) (bool, error)
}
