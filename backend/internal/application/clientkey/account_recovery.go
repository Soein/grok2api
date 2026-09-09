package clientkey

import (
	"context"
	"errors"
	"fmt"

	clientkeydomain "github.com/chenyme/grok2api/backend/internal/domain/clientkey"
	"github.com/chenyme/grok2api/backend/internal/infra/security"
	"github.com/chenyme/grok2api/backend/internal/repository"
)

const (
	accountRecoveryInternalPrefix = "account-recovery-internal"
	accountRecoveryInternalName   = "[system] Account Recovery"
)

// EnsureAccountRecoveryIdentity 协调维护探针的不可导出内部身份，保留重启前的 ID 和计费归属。
// enabled 为 false 时不创建身份；已存在的身份会停用。此身份不能用于公开 API 鉴权。
func (s *Service) EnsureAccountRecoveryIdentity(ctx context.Context, enabled bool) (clientkeydomain.Key, error) {
	value, err := s.keys.GetByPrefix(ctx, accountRecoveryInternalPrefix)
	if errors.Is(err, repository.ErrNotFound) {
		if !enabled {
			return clientkeydomain.Key{}, nil
		}
		value, err = s.createAccountRecoveryIdentity(ctx)
		if errors.Is(err, repository.ErrConflict) {
			value, err = s.keys.GetByPrefix(ctx, accountRecoveryInternalPrefix)
		}
	}
	if err != nil {
		return clientkeydomain.Key{}, fmt.Errorf("读取系统账号恢复身份: %w", err)
	}
	if value.InternalKind != clientkeydomain.InternalKindAccountRecovery {
		return clientkeydomain.Key{}, fmt.Errorf("%w: 保留前缀已被占用", ErrConflict)
	}
	value.Name = accountRecoveryInternalName
	value.Enabled = enabled
	value.ExpiresAt = nil
	value.RPMLimit = 0
	value.MaxConcurrent = 0
	value.BillingLimitUSDTicks = 0
	value.AllowModelAliases = false
	value.AllowedModels = nil
	value.ProviderScope = clientkeydomain.ProviderScopeBuild
	value.TierScope = clientkeydomain.TierScopeAll
	updated, err := s.keys.Update(ctx, value)
	if err != nil {
		return clientkeydomain.Key{}, fmt.Errorf("更新系统账号恢复身份: %w", err)
	}
	s.authCache.deleteID(updated.ID)
	return updated, nil
}

func (s *Service) createAccountRecoveryIdentity(ctx context.Context) (clientkeydomain.Key, error) {
	if s.cipher == nil {
		return clientkeydomain.Key{}, errors.New("客户端 Key 加密器未配置")
	}
	secretPart, err := security.NewOpaqueToken(32)
	if err != nil {
		return clientkeydomain.Key{}, err
	}
	raw := security.FormatClientKey(accountRecoveryInternalPrefix, secretPart)
	encrypted, err := s.cipher.Encrypt(raw)
	if err != nil {
		return clientkeydomain.Key{}, fmt.Errorf("加密系统账号恢复身份: %w", err)
	}
	return s.keys.Create(ctx, clientkeydomain.Key{
		Name: accountRecoveryInternalName, Prefix: accountRecoveryInternalPrefix,
		SecretHash: security.HashToken(raw), EncryptedSecret: encrypted,
		InternalKind: clientkeydomain.InternalKindAccountRecovery, Enabled: true,
		ProviderScope: clientkeydomain.ProviderScopeBuild, TierScope: clientkeydomain.TierScopeAll,
	})
}
