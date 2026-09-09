package model

import (
	"context"

	modeldomain "github.com/chenyme/grok2api/backend/internal/domain/model"
)

// ListConfiguredEnabled returns configured routes before account availability
// filtering, so maintenance can probe an unavailable pool without enabling it.
func (s *Service) ListConfiguredEnabled(ctx context.Context) ([]modeldomain.Route, error) {
	return s.models.ListConfiguredEnabled(ctx)
}
