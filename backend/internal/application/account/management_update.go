package account

import (
	"context"

	accountdomain "github.com/chenyme/grok2api/backend/internal/domain/account"
	"github.com/chenyme/grok2api/backend/internal/repository"
)

func (s *Service) markCredentialReauthRequired(ctx context.Context, value accountdomain.Credential, reason string) error {
	if writer, ok := s.accounts.(repository.AccountAuthenticationStateWriter); ok {
		changed, err := writer.MarkAccountReauthRequired(ctx, value, reason, s.now())
		if err != nil {
			return mapRepositoryError(err)
		}
		if !changed {
			return nil
		}
	} else {
		// Compatibility for repository adapters without the optional narrow writer.
		// The production relational repository always implements the guarded path.
		value.AuthStatus = accountdomain.AuthStatusReauthRequired
		value.LastError = reason
		if len(value.LastError) > 512 {
			value.LastError = value.LastError[:512]
		}
		if _, err := s.accounts.Update(ctx, value); err != nil {
			return mapRepositoryError(err)
		}
	}
	if s.sticky != nil {
		_ = s.sticky.DeleteByAccount(ctx, value.ID)
	}
	return nil
}
