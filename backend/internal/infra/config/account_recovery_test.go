package config

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestAccountRecoveryDefaults(t *testing.T) {
	got := defaultConfig().AccountRecovery
	want := AccountRecoveryConfig{
		Interval: Duration(5 * time.Minute), BatchSize: 10, Concurrency: 1,
		ProbeTimeout: Duration(90 * time.Second), Build: true, Web: true, Console: true,
		ReauthBatchSize: 1, ReauthBackoffBase: Duration(time.Hour), ReauthBackoffMax: Duration(24 * time.Hour),
		BuildModels: []string{"grok-4.5", "grok-4.6"},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("defaults = %#v, want %#v", got, want)
	}
}

func TestAccountRecoveryValidation(t *testing.T) {
	tests := []struct {
		name   string
		change func(*AccountRecoveryConfig)
	}{
		{"interval short", func(c *AccountRecoveryConfig) { c.Interval = Duration(time.Minute - 1) }},
		{"interval long", func(c *AccountRecoveryConfig) { c.Interval = Duration(24*time.Hour + 1) }},
		{"batch zero", func(c *AccountRecoveryConfig) { c.BatchSize = 0 }},
		{"batch large", func(c *AccountRecoveryConfig) { c.BatchSize = 101 }},
		{"concurrency zero", func(c *AccountRecoveryConfig) { c.Concurrency = 0 }},
		{"concurrency large", func(c *AccountRecoveryConfig) { c.BatchSize = 100; c.Concurrency = 11 }},
		{"concurrency exceeds batch", func(c *AccountRecoveryConfig) { c.BatchSize = 1; c.Concurrency = 2 }},
		{"probe short", func(c *AccountRecoveryConfig) { c.ProbeTimeout = Duration(10*time.Second - 1) }},
		{"probe long", func(c *AccountRecoveryConfig) { c.ProbeTimeout = Duration(2*time.Minute + 1) }},
		{"reauth batch zero", func(c *AccountRecoveryConfig) { c.ReauthBatchSize = 0 }},
		{"reauth batch large", func(c *AccountRecoveryConfig) { c.ReauthBatchSize = 11 }},
		{"backoff short", func(c *AccountRecoveryConfig) { c.ReauthBackoffBase = Duration(5*time.Minute - 1) }},
		{"backoff reversed", func(c *AccountRecoveryConfig) { c.ReauthBackoffMax = Duration(30 * time.Minute) }},
		{"backoff long", func(c *AccountRecoveryConfig) { c.ReauthBackoffMax = Duration(7*24*time.Hour + 1) }},
		{"models empty", func(c *AccountRecoveryConfig) { c.BuildModels = nil }},
		{"models large", func(c *AccountRecoveryConfig) { c.BuildModels = make([]string, 9) }},
		{"model blank", func(c *AccountRecoveryConfig) { c.BuildModels = []string{" "} }},
		{"model whitespace", func(c *AccountRecoveryConfig) { c.BuildModels = []string{"grok-4.5\tfast"} }},
		{"model unicode whitespace", func(c *AccountRecoveryConfig) { c.BuildModels = []string{"grok-4.5\u00a0fast"} }},
		{"model control", func(c *AccountRecoveryConfig) { c.BuildModels = []string{"grok-4.5\x00"} }},
		{"model source prefix", func(c *AccountRecoveryConfig) { c.BuildModels = []string{"web/grok-4.5"} }},
		{"model colon prefix", func(c *AccountRecoveryConfig) { c.BuildModels = []string{"console:grok-4.5"} }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cfg := defaultConfig()
			cfg.Secrets.JWTSecret = "12345678901234567890123456789012"
			cfg.Secrets.CredentialEncryptionKey = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="
			test.change(&cfg.AccountRecovery)
			if err := cfg.Validate(); err == nil || !strings.Contains(err.Error(), "accountRecovery.") {
				t.Fatalf("expected accountRecovery validation error, got %v", err)
			}
		})
	}
}

func TestAccountRecoveryLoadOverridesAndNormalizesModels(t *testing.T) {
	t.Setenv(DatabaseURLEnv, "")
	path := filepath.Join(t.TempDir(), "config.yaml")
	data := `secrets:
  jwtSecret: "12345678901234567890123456789012"
  credentialEncryptionKey: "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="
accountRecovery:
  enabled: true
  interval: 1m
  batchSize: 100
  concurrency: 10
  probeTimeout: 10s
  build: false
  web: false
  console: false
  includeDisabled: true
  ssoReauth: true
  reauthBatchSize: 100
  reauthBackoffBase: 5m
  reauthBackoffMax: 168h
  buildModels: [" grok-4.5 ", "grok-4.6"]
`
	if err := os.WriteFile(path, []byte(data), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	want := AccountRecoveryConfig{
		Enabled: true, Interval: Duration(time.Minute), BatchSize: 100, Concurrency: 10,
		ProbeTimeout: Duration(10 * time.Second), IncludeDisabled: true, SSOReauth: true,
		ReauthBatchSize: 100, ReauthBackoffBase: Duration(5 * time.Minute), ReauthBackoffMax: Duration(7 * 24 * time.Hour),
		BuildModels: []string{"grok-4.5", "grok-4.6"},
	}
	if !reflect.DeepEqual(cfg.AccountRecovery, want) {
		t.Fatalf("loaded = %#v, want %#v", cfg.AccountRecovery, want)
	}
}
