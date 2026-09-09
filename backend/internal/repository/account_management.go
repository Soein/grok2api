package repository

import (
	"context"
	"time"

	"github.com/chenyme/grok2api/backend/internal/domain/account"
)

// ManagementAccountUpdates contains only explicitly requested administrative
// fields. Nil leaves a field unchanged; credential identity and OAuth state are
// deliberately absent so a stale management view cannot revert token rotation.
type ManagementAccountUpdates struct {
	Name                      *string
	Enabled                   *bool
	Priority                  *int
	MaxConcurrent             *int
	MinimumRemaining          *float64
	EncryptedCloudflareCookie *string
	BuildSuperEntitled        *bool
	BuildRouteMode            *account.BuildRouteMode
}

type AccountManagementWriter interface {
	UpdateAccountManagement(context.Context, uint64, ManagementAccountUpdates) (account.Credential, error)
}

// AccountAuthenticationStateWriter rejects only the credential snapshot that
// produced the authentication failure, and never writes credential material.
type AccountAuthenticationStateWriter interface {
	MarkAccountReauthRequired(context.Context, account.Credential, string, time.Time) (bool, error)
}
