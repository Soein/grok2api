package clientkey

import (
	"context"
	"errors"
	"path/filepath"
	"testing"
	"time"

	clientkeydomain "github.com/chenyme/grok2api/backend/internal/domain/clientkey"
	"github.com/chenyme/grok2api/backend/internal/infra/persistence/relational"
	"github.com/chenyme/grok2api/backend/internal/infra/security"
	"github.com/chenyme/grok2api/backend/internal/repository"
)

func TestAccountRecoveryIdentityIsStableHiddenAndSystemManaged(t *testing.T) {
	ctx := context.Background()
	database, err := relational.OpenSQLite(ctx, filepath.Join(t.TempDir(), "account-recovery-identity.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer database.Close()
	if err := database.InitializeSchema(ctx); err != nil {
		t.Fatal(err)
	}
	repo := relational.NewClientKeyRepository(database)
	cipher := testCipher(t)
	service := NewService(repo, nil, nil, 60, 5, cipher)
	unused, err := service.EnsureAccountRecoveryIdentity(ctx, false)
	if err != nil || unused.ID != 0 {
		t.Fatalf("disabled fresh identity ID=%d, err=%v", unused.ID, err)
	}
	if _, err := repo.GetByPrefix(ctx, accountRecoveryInternalPrefix); !errors.Is(err, repository.ErrNotFound) {
		t.Fatalf("disabled identity unexpectedly persisted: %v", err)
	}
	first, err := service.EnsureAccountRecoveryIdentity(ctx, true)
	if err != nil {
		t.Fatal(err)
	}
	if first.ID == 0 || first.InternalKind != clientkeydomain.InternalKindAccountRecovery || !first.Enabled || first.Name != "[system] Account Recovery" || first.Prefix != "account-recovery-internal" {
		t.Fatal("unexpected identity metadata")
	}
	if first.ProviderScope != clientkeydomain.ProviderScopeBuild || first.TierScope != clientkeydomain.TierScopeAll || first.RPMLimit != 0 || first.MaxConcurrent != 0 || first.BillingLimitUSDTicks != 0 || first.AllowModelAliases || len(first.AllowedModels) != 0 || first.ExpiresAt != nil {
		t.Fatal("unexpected identity policy")
	}
	if values, total, err := service.List(ctx, 1, 20, "", ListFilter{}); err != nil || total != 0 || len(values) != 0 {
		t.Fatalf("internal identity leaked in list: count=%d total=%d err=%v", len(values), total, err)
	}
	raw, err := cipher.Decrypt(first.EncryptedSecret)
	if err != nil {
		t.Fatal(err)
	}
	// Repeat to cover cached as well as repository-backed public authentication.
	for range 2 {
		if _, release, err := service.Authenticate(ctx, raw); !errors.Is(err, ErrInvalidKey) || release != nil {
			t.Fatalf("internal identity authenticated externally: %v", err)
		}
	}
	if raw, err := service.RevealSecret(ctx, first.ID); !errors.Is(err, ErrSystemManaged) || raw != "" {
		t.Fatalf("reveal error=%v", err)
	}
	if _, err := service.Update(ctx, first.ID, UpdateInput{}); !errors.Is(err, ErrSystemManaged) {
		t.Fatalf("update error=%v", err)
	}
	if err := service.Delete(ctx, first.ID); !errors.Is(err, ErrSystemManaged) {
		t.Fatalf("delete error=%v", err)
	}
	if _, err := service.BatchSetEnabled(ctx, []uint64{first.ID}, false); !errors.Is(err, ErrSystemManaged) {
		t.Fatalf("batch update error=%v", err)
	}
	if _, err := service.BatchDelete(ctx, []uint64{first.ID}); !errors.Is(err, ErrSystemManaged) {
		t.Fatalf("batch delete error=%v", err)
	}

	// Trusted storage may contain an older policy; restart reconciles policy without replacing identity or credentials.
	expired := time.Now().Add(-time.Hour)
	changed := first
	changed.Name, changed.Enabled, changed.ExpiresAt = "old name", false, &expired
	changed.RPMLimit, changed.MaxConcurrent, changed.BillingLimitUSDTicks = 1, 1, 1
	changed.AllowModelAliases = true
	changed.ProviderScope, changed.TierScope = clientkeydomain.ProviderScopeWeb, clientkeydomain.TierScopeFree
	if _, err := repo.Update(ctx, changed); err != nil {
		t.Fatal(err)
	}
	restarted := NewService(repo, nil, nil, 60, 5, cipher)
	second, err := restarted.EnsureAccountRecoveryIdentity(ctx, true)
	if err != nil {
		t.Fatal(err)
	}
	if second.ID != first.ID || second.SecretHash != first.SecretHash || second.EncryptedSecret != first.EncryptedSecret {
		t.Fatal("restart replaced identity or credentials")
	}
	if second.Name != first.Name || !second.Enabled || second.ExpiresAt != nil || second.RPMLimit != 0 || second.MaxConcurrent != 0 || second.BillingLimitUSDTicks != 0 || second.AllowModelAliases || len(second.AllowedModels) != 0 || second.ProviderScope != first.ProviderScope || second.TierScope != first.TierScope {
		t.Fatal("restart did not reconcile policy")
	}
	disabled, err := restarted.EnsureAccountRecoveryIdentity(ctx, false)
	if err != nil || disabled.ID != first.ID || disabled.Enabled {
		t.Fatalf("disable ID=%d err=%v", disabled.ID, err)
	}
	reenabled, err := restarted.EnsureAccountRecoveryIdentity(ctx, true)
	if err != nil || reenabled.ID != first.ID || !reenabled.Enabled {
		t.Fatalf("reenable ID=%d err=%v", reenabled.ID, err)
	}
	quality, err := restarted.EnsureQualityGuardIdentity(ctx, true)
	if err != nil || quality.ID == first.ID {
		t.Fatalf("quality guard identity collided: %v", err)
	}
}

func TestAccountRecoveryIdentityRejectsReservedPrefixCollision(t *testing.T) {
	ctx := context.Background()
	database, err := relational.OpenSQLite(ctx, filepath.Join(t.TempDir(), "account-recovery-collision.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer database.Close()
	if err := database.InitializeSchema(ctx); err != nil {
		t.Fatal(err)
	}
	repo := relational.NewClientKeyRepository(database)
	cipher := testCipher(t)
	raw := security.FormatClientKey(accountRecoveryInternalPrefix, "test-reserved-prefix-secret")
	encrypted, err := cipher.Encrypt(raw)
	if err != nil {
		t.Fatal(err)
	}
	other, err := repo.Create(ctx, clientkeydomain.Key{Name: "user", Prefix: accountRecoveryInternalPrefix, SecretHash: security.HashToken(raw), EncryptedSecret: encrypted, Enabled: true})
	if err != nil {
		t.Fatal(err)
	}
	service := NewService(repo, nil, nil, 60, 5, cipher)
	if _, err := service.EnsureAccountRecoveryIdentity(ctx, true); !errors.Is(err, ErrConflict) {
		t.Fatalf("collision error=%v", err)
	}
	current, err := repo.Get(ctx, other.ID)
	if err != nil || current.InternalKind != "" || current.Name != "user" {
		t.Fatalf("existing user identity changed: %v", err)
	}
}
