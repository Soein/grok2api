package repository

import (
	"context"
	"time"

	"github.com/chenyme/grok2api/backend/internal/domain/account"
)

// ReauthRepository maintains linked Build credentials without changing account
// identity, routing preferences, enablement or quota state.
type ReauthRepository interface {
	ListBuildReauthCandidates(context.Context, time.Time, bool, int) ([]uint64, error)
	ClaimBuildReauth(context.Context, account.Credential, account.Credential, time.Time, time.Time, bool) (bool, error)
	SaveBuildReauth(context.Context, account.Credential, account.Credential, string, string, string, time.Time, int) (bool, error)
	FailBuildReauth(context.Context, account.Credential, time.Time, time.Time, string) error
	RejectReauthSSO(context.Context, account.Credential) error
}
