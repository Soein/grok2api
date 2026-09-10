package localstatsig

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"
)

func fixtureDynamic() Config {
	trailer := 3
	timeSet := map[int]bool{}
	for a := 0; a < 16; a++ {
		for b := 0; b < 16; b++ {
			for c := 0; c < 16; c++ {
				timeSet[((a*b*c+5)/10)*10] = true
			}
		}
	}
	times := make([]int, 0, len(timeSet))
	for tm := range timeSet {
		times = append(times, tm)
	}
	sort.Ints(times)
	frames := make([][][]string, 4)
	for s := range frames {
		frames[s] = make([][]string, 16)
		for r := range frames[s] {
			frames[s][r] = make([]string, len(times))
			for i, tm := range times {
				frames[s][r][i] = fmt.Sprintf("%x%x%x", s, r, tm)
			}
		}
	}
	return Config{SchemaVersion: 2, BuildID: "synthetic-build", SuffixPrefix: "obfiowerehiring", Trailer: &trailer, ExpiresAt: fixtureNow.Add(3600000000000), Times: times, Fingerprints: frames}
}

func TestDynamicMatchesCurrentOfficialModule(t *testing.T) {
	// Captured from the official browser module on 2026-09-10, with its random
	// byte fixed to zero. This validates parity independently of this Go code.
	const meta = "txmPq7hfP422YvjDfO+x6tFMwOuDY2FW7GqV+VgFXWa67FkOvbM9NBvDEckXp6Vt"
	const want = "ALcZj6u4Xz+NtmL4w3zvserRTMDrg2NhVuxqlflYBV1muuxZDr2zPTQbwxHJF6elbSwYUwZ0jz/A76m1QMUR61ptMYyhAw"
	now := time.Unix(1789033372, 0)
	c := fixtureDynamic()
	c.ExpiresAt = now.Add(time.Hour)
	c.Fingerprints[3][8][sort.SearchInts(c.Times, 850)] = "29bad70f851eb851eb8504040f851eb851eb8500"
	sig, err := c.Sign("POST", "/rest/app-chat/conversations/new", meta, now, bytes.NewReader([]byte{0}))
	if err != nil {
		t.Fatal(err)
	}
	if sig != want {
		t.Fatal("signature differs from official module vector")
	}
}

func TestDynamicUsesAllLowBitsAndLegalTimes(t *testing.T) {
	c := fixtureDynamic()
	if len(c.Times) != 182 || c.Times[0] != 0 || c.Times[len(c.Times)-1] != 3380 {
		t.Fatal("unexpected legal time set")
	}
	for set := 0; set < 4; set++ {
		for row := 0; row < 16; row++ {
			for a := 0; a < 16; a++ {
				for b := 0; b < 16; b++ {
					for d := 0; d < 16; d++ {
						seed := make([]byte, 48)
						seed[5] = byte(128 + set)
						seed[4] = byte(240 + row)
						seed[44] = byte(128 + a)
						seed[40] = byte(96 + b)
						seed[3] = byte(240 + d)
						// Quotient/remainder computes the nearest ten independently of the
						// implementation's offset-before-division expression, including ties.
						product := a * b * d
						tm := product - product%10
						if product%10 >= 5 {
							tm += 10
						}
						want := fmt.Sprintf("%x%x%x", set, row, tm)
						if got := c.fingerprintForSeed(seed); got != want {
							t.Fatalf("seed selected %q, want %q", got, want)
						}
					}
				}
			}
		}
	}
}

func TestDynamicBrowserOracleVectors(t *testing.T) {
	// Fingerprints are output from the current browser CSS oracle, not a Go
	// float emulation. Expected signatures were independently packed/hashed with
	// Python. This comparison alone does not establish upstream acceptance.
	for _, tc := range []struct {
		set, row, tm    int
		meta, hex, want string
	}{
		{0, 0, 0, "AAECAAAABgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8gISIjJCUmJwApKisALS4v", "1230cd100100", "WlpbWFpaWlxdUlNQUVZXVFVKS0hJTk9MTUJDQEFGR0RFent4eX5/fH1ac3BxWnd0dRq4W1qDbQQ4hEdfp0K55m+HFUROWQ"},
		{3, 15, 3380, "AAECDw8DBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8gISIjJCUmJw8pKisPLS4v", "f2d59407ae147ae147ae0e147ae147ae1480e147ae147ae14807ae147ae147ae00", "WlpbWFVVWVxdUlNQUVZXVFVKS0hJTk9MTUJDQEFGR0RFent4eX5/fH1Vc3BxVXd0dRq4W1pjWR0usm3eePcttRfr3eqjWQ"},
		{2, 7, 10, "AAECAQcCBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8gISIjJCUmJwUpKisBLS4v", "bbc22b100100", "WlpbWFtdWFxdUlNQUVZXVFVKS0hJTk9MTUJDQEFGR0RFent4eX5/fH1fc3BxW3d0dRq4W1pTQ+8gtFzvQSMhCEj0tmRzWQ"},
		{1, 9, 1000, "AAECDQkBBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8gISIjJCUmJwspKisHLS4v", "da54bc0f851eb851eb8504040f851eb851eb8500", "WlpbWFdTW1xdUlNQUVZXVFVKS0hJTk9MTUJDQEFGR0RFent4eX5/fH1Rc3BxXXd0dRq4W1qH50Vls+QL9n3lQ4GauGUjWQ"},
	} {
		t.Run(fmt.Sprintf("%d-%d-%d", tc.set, tc.row, tc.tm), func(t *testing.T) {
			cfg := fixtureDynamic()
			cfg.Fingerprints[tc.set][tc.row][sort.SearchInts(cfg.Times, tc.tm)] = tc.hex
			sig, err := cfg.Sign("POST", "/rest/app-chat/conversations/new", tc.meta, fixtureNow, strings.NewReader("Z"))
			if err != nil {
				t.Fatal(err)
			}
			if sig != tc.want {
				t.Fatalf("signature differs from independent oracle vector: %q", sig)
			}
		})
	}
}

func TestDynamicRejectsIncompleteOrAmbiguousConfig(t *testing.T) {
	for _, tc := range []struct {
		name   string
		change func(*Config)
	}{
		{"no-build", func(c *Config) { c.BuildID = "" }},
		{"prefix", func(c *Config) { c.SuffixPrefix = "" }},
		{"wrong-prefix", func(c *Config) { c.SuffixPrefix = "other" }},
		{"wrong-trailer", func(c *Config) { v := 4; c.Trailer = &v }},
		{"missing-times", func(c *Config) { c.Times = nil }},
		{"missing-time", func(c *Config) { c.Times = c.Times[:181] }},
		{"duplicate-time", func(c *Config) { c.Times[1] = 0 }},
		{"illegal-time", func(c *Config) { c.Times[1] = 1 }},
		{"order", func(c *Config) { c.Times[1], c.Times[2] = c.Times[2], c.Times[1] }},
		{"no-table", func(c *Config) { c.Fingerprints = nil }},
		{"missing-set", func(c *Config) { c.Fingerprints = c.Fingerprints[:3] }},
		{"missing-row", func(c *Config) { c.Fingerprints[1] = c.Fingerprints[1][:15] }},
		{"missing-frame", func(c *Config) { c.Fingerprints[3][15] = c.Fingerprints[3][15][:181] }},
		{"empty-hex", func(c *Config) { c.Fingerprints[3][15][181] = "" }},
		{"uppercase-hex", func(c *Config) { c.Fingerprints[3][15][181] = "AB" }},
		{"invalid-hex", func(c *Config) { c.Fingerprints[3][15][181] = "0x5" }},
		{"long-hex", func(c *Config) { c.Fingerprints[3][15][181] = strings.Repeat("a", 257) }},
		{"mixed-legacy", func(c *Config) { c.MetaContent = fixtureMeta }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := fixtureDynamic()
			tc.change(&c)
			if _, err := c.Sign("POST", "/x", fixtureMeta, fixtureNow, strings.NewReader("Z")); err == nil {
				t.Fatal("invalid schema2 config signed")
			}
		})
	}
}

func TestDynamicRequiresCanonicalMeta(t *testing.T) {
	for _, meta := range []string{"bad", strings.Repeat("A", 63), base64.StdEncoding.EncodeToString(make([]byte, 49)), fixtureMeta + "\n"} {
		c := fixtureDynamic()
		if _, err := c.Sign("POST", "/x", meta, fixtureNow, strings.NewReader("Z")); err == nil {
			t.Fatal("noncanonical meta signed")
		}
	}
}

func TestDynamicReloadAndSchemaSeparation(t *testing.T) {
	path := filepath.Join(t.TempDir(), "params.json")
	c := fixtureDynamic()
	writeConfig(t, path, c)
	if _, err := Load(path, fixtureNow); err != nil {
		t.Fatal(err)
	}
	for _, version := range []int{1, 2} {
		var data map[string]any
		c := fixtureDynamic()
		if version == 1 {
			c = fixtureConfig()
		}
		b, _ := json.Marshal(c)
		_ = json.Unmarshal(b, &data)
		if version == 1 {
			data["buildID"] = ""
		} else {
			data["headerHex"] = ""
		}
		b, _ = json.Marshal(data)
		if err := os.WriteFile(path, b, 0600); err != nil {
			t.Fatal(err)
		}
		if _, err := Load(path, fixtureNow); err == nil {
			t.Fatalf("schema %d accepted mixed fields", version)
		}
	}
}
