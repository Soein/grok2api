package localstatsig

import (
	"encoding/base64"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

const fixtureMeta = "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8gISIjJCUmJygpKissLS4v"

var fixtureNow = time.Unix(1682924400+123456, 0)

func fixtureConfig() Config {
	trailer := 3
	return Config{SchemaVersion: 1, MetaContent: fixtureMeta,
		HeaderHex: "00000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f",
		Suffix:    "fixture-suffix", Trailer: &trailer, ExpiresAt: fixtureNow.Add(time.Hour)}
}

func TestSignFixedVector(t *testing.T) {
	cfg := fixtureConfig()
	sig, err := cfg.Sign("POST", "/rest/app-chat/conversations/new", fixtureMeta, fixtureNow, strings.NewReader("Z"))
	if err != nil {
		t.Fatal(err)
	}
	// Independently calculated with Python hashlib/struct; this tests the format,
	// not acceptance by the upstream service.
	want := "WlpbWFleX1xdUlNQUVZXVFVKS0hJTk9MTUJDQEFGR0RFent4eX5/fH1yc3Bxdnd0dRq4W1pmmltdEhYHlrQA4VqsqZUbWQ"
	if sig != want {
		t.Fatalf("signature = %q, want fixed vector", sig)
	}
	b, err := base64.RawStdEncoding.DecodeString(sig)
	if err != nil || len(b) != 70 {
		t.Fatalf("format: length=%d err=%v", len(b), err)
	}
}

func TestSignBindsMethodPathTimeAndSuffix(t *testing.T) {
	cfg := fixtureConfig()
	base, _ := cfg.Sign("POST", "/x", fixtureMeta, fixtureNow, strings.NewReader("Z"))
	for _, tc := range []struct {
		name, method, path, suffix string
		now                        time.Time
	}{
		{"method", "GET", "/x", cfg.Suffix, fixtureNow},
		{"path", "POST", "/y", cfg.Suffix, fixtureNow},
		{"time", "POST", "/x", cfg.Suffix, fixtureNow.Add(time.Second)},
		{"suffix", "POST", "/x", "other-suffix", fixtureNow},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg.Suffix = tc.suffix
			sig, err := cfg.Sign(tc.method, tc.path, fixtureMeta, tc.now, strings.NewReader("Z"))
			if err != nil {
				t.Fatal(err)
			}
			if sig == base {
				t.Fatal("changed signed input did not change signature")
			}
		})
	}
}

func TestRejectInvalidConfig(t *testing.T) {
	for _, tc := range []struct {
		name   string
		change func(*Config)
	}{
		{"version", func(c *Config) { c.SchemaVersion = 2 }},
		{"meta", func(c *Config) { c.MetaContent = "bad" }},
		{"header-length", func(c *Config) { c.HeaderHex = "00" }},
		{"header-not-normalized", func(c *Config) { c.HeaderHex = "ff" + c.HeaderHex[2:] }},
		{"header-meta-mismatch", func(c *Config) { c.HeaderHex = c.HeaderHex[:96] + "ff" }},
		{"empty-suffix", func(c *Config) { c.Suffix = "" }},
		{"missing-trailer", func(c *Config) { c.Trailer = nil }},
		{"invalid-trailer", func(c *Config) { v := 256; c.Trailer = &v }},
		{"expired", func(c *Config) { c.ExpiresAt = fixtureNow }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := fixtureConfig()
			tc.change(&cfg)
			if _, err := cfg.Sign("POST", "/x", fixtureMeta, fixtureNow, strings.NewReader("Z")); err == nil {
				t.Fatal("invalid config signed")
			}
		})
	}
}

func TestSignRejectsMismatchInvalidInputAndRandomFailure(t *testing.T) {
	cfg := fixtureConfig()
	for _, tc := range []struct {
		name, method, path, meta, random string
		now                              time.Time
	}{
		{"meta-mismatch", "POST", "/x", "other", "Z", fixtureNow},
		{"method", "post", "/x", fixtureMeta, "Z", fixtureNow},
		{"absolute-url", "POST", "https://grok.com/x", fixtureMeta, "Z", fixtureNow},
		{"query", "POST", "/x?foo=bar", fixtureMeta, "Z", fixtureNow},
		{"control", "POST", "/x\n", fixtureMeta, "Z", fixtureNow},
		{"random-failure", "POST", "/x", fixtureMeta, "", fixtureNow},
		{"pre-epoch", "POST", "/x", fixtureMeta, "Z", time.Unix(1682924399, 0)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := cfg.Sign(tc.method, tc.path, tc.meta, tc.now, strings.NewReader(tc.random)); err == nil {
				t.Fatal("invalid request signed")
			}
		})
	}
}

func writeConfig(t *testing.T, path string, c Config) {
	t.Helper()
	b, err := json.Marshal(c)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(path+".new", b, 0600); err != nil {
		t.Fatal(err)
	}
	if err = os.Rename(path+".new", path); err != nil {
		t.Fatal(err)
	}
}

func TestLoadReloadsAtomicReplacementAndRejectsInvalidUpdate(t *testing.T) {
	path := filepath.Join(t.TempDir(), "params.json")
	cfg := fixtureConfig()
	writeConfig(t, path, cfg)
	first, err := Load(path, fixtureNow)
	if err != nil {
		t.Fatal(err)
	}
	cfg.Suffix = "second"
	writeConfig(t, path, cfg)
	second, err := Load(path, fixtureNow)
	if err != nil {
		t.Fatal(err)
	}
	if first.Suffix == second.Suffix {
		t.Fatal("atomic replacement not reloaded")
	}
	if err = os.WriteFile(path, []byte(`{"schemaVersion":1}`), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err = Load(path, fixtureNow); err == nil {
		t.Fatal("invalid update retained old configuration")
	}
	if err = os.WriteFile(path, []byte(strings.Repeat(" ", maxConfigBytes+1)), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err = Load(path, fixtureNow); err == nil {
		t.Fatal("oversize configuration accepted")
	}
}
