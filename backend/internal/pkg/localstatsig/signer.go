package localstatsig

import (
	"bytes"
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"math"
	"net/url"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode"
)

const maxConfigBytes = 2 << 20
const epoch = 1682924400

// Config holds either a normalized snapshot bound to one verification meta
// (schema 1), or a complete browser fingerprint table for one build (schema 2).
// Valid configuration does not prove upstream acceptance.
type Config struct {
	SchemaVersion int          `json:"schemaVersion"`
	MetaContent   string       `json:"metaContent,omitempty"`
	HeaderHex     string       `json:"headerHex,omitempty"`
	Suffix        string       `json:"suffix,omitempty"`
	BuildID       string       `json:"buildID,omitempty"`
	SuffixPrefix  string       `json:"suffixPrefix,omitempty"`
	Times         []int        `json:"times,omitempty"`
	Fingerprints  [][][]string `json:"fingerprints,omitempty"`
	Trailer       *int         `json:"trailer"`
	ExpiresAt     time.Time    `json:"expiresAt"`
}

var (
	ErrUnavailable    = errors.New("signing configuration unavailable")
	ErrMetaMismatch   = errors.New("signing configuration does not match meta")
	ErrInvalidRequest = errors.New("invalid signing request")
)

func (c Config) validate(now time.Time) ([]byte, error) {
	if !now.Before(c.ExpiresAt) || c.Trailer == nil || *c.Trailer < 0 || *c.Trailer > 255 {
		return nil, ErrUnavailable
	}
	if c.SchemaVersion == 2 {
		return nil, c.validateDynamic()
	}
	if c.SchemaVersion != 1 || c.Suffix == "" || len(c.Suffix) > 4096 || c.BuildID != "" || c.SuffixPrefix != "" || c.Times != nil || c.Fingerprints != nil {
		return nil, ErrUnavailable
	}
	meta, err := decodeMeta(c.MetaContent)
	if err != nil {
		return nil, ErrUnavailable
	}
	header, err := hex.DecodeString(c.HeaderHex)
	if err != nil || len(header) != 49 || header[0] != 0 || !bytes.Equal(header[1:], meta) {
		return nil, ErrUnavailable
	}
	return header, nil
}

func decodeMeta(value string) ([]byte, error) {
	meta, err := base64.StdEncoding.DecodeString(value)
	if err != nil || len(meta) != 48 || base64.StdEncoding.EncodeToString(meta) != value {
		return nil, ErrInvalidRequest
	}
	return meta, nil
}

// legalFrameTimes contains every result of the browser's rounded product of
// three four-bit seed values. It is independent of any supplied configuration.
var legalFrameTimes = func() []int {
	set := map[int]bool{}
	for a := 0; a < 16; a++ {
		for b := 0; b < 16; b++ {
			for c := 0; c < 16; c++ {
				set[((a*b*c+5)/10)*10] = true
			}
		}
	}
	times := make([]int, 0, len(set))
	for tm := range set {
		times = append(times, tm)
	}
	sort.Ints(times)
	return times
}()

func (c Config) validateDynamic() error {
	if c.MetaContent != "" || c.HeaderHex != "" || c.Suffix != "" || c.BuildID == "" || len(c.BuildID) > 256 || strings.TrimSpace(c.BuildID) != c.BuildID || c.SuffixPrefix != "obfiowerehiring" || *c.Trailer != 3 {
		return ErrUnavailable
	}
	for _, r := range c.BuildID {
		if unicode.IsControl(r) {
			return ErrUnavailable
		}
	}
	if len(c.Times) != len(legalFrameTimes) || len(c.Fingerprints) != 4 {
		return ErrUnavailable
	}
	for i, tm := range c.Times {
		if tm != legalFrameTimes[i] {
			return ErrUnavailable
		}
	}
	for _, set := range c.Fingerprints {
		if len(set) != 16 {
			return ErrUnavailable
		}
		for _, row := range set {
			if len(row) != len(legalFrameTimes) {
				return ErrUnavailable
			}
			for _, frame := range row {
				if len(frame) == 0 || len(frame) > 256 {
					return ErrUnavailable
				}
				for _, r := range frame {
					if !((r >= '0' && r <= '9') || (r >= 'a' && r <= 'f')) {
						return ErrUnavailable
					}
				}
			}
		}
	}
	return nil
}

func (c Config) fingerprintForSeed(seed []byte) string {
	// All operands are nonnegative. Adding five before integer division matches
	// Math.round(product / 10) without introducing floating-point differences.
	product := int(seed[44]%16) * int(seed[40]%16) * int(seed[3]%16)
	tm := ((product + 5) / 10) * 10
	return c.Fingerprints[seed[5]%4][seed[4]%16][sort.SearchInts(c.Times, tm)]
}

// ValidateRequest rejects ambiguous methods and URLs. The protocol signs the
// exact uppercase method and absolute path, without query or fragment.
func ValidateRequest(method, path, meta string) error {
	if method == "" || len(method) > 16 || meta == "" || len(meta) > 256 {
		return ErrInvalidRequest
	}
	if _, err := decodeMeta(meta); err != nil {
		return err
	}
	for _, r := range method {
		if r < 'A' || r > 'Z' {
			return ErrInvalidRequest
		}
	}
	if len(path) == 0 || len(path) > 4096 || !strings.HasPrefix(path, "/") || strings.HasPrefix(path, "//") || strings.ContainsAny(path, "?#") {
		return ErrInvalidRequest
	}
	for _, r := range path {
		if unicode.IsControl(r) || unicode.IsSpace(r) {
			return ErrInvalidRequest
		}
	}
	u, err := url.ParseRequestURI(path)
	if err != nil || u.IsAbs() || u.Host != "" {
		return ErrInvalidRequest
	}
	return nil
}

// Sign generates a 70-byte challenge using a matching, unexpired snapshot.
// random must provide one cryptographically random byte in production.
// Its layout is adapted from the MIT-licensed grok-web-api challenge algorithm;
// see LICENSE and README.md in this directory for provenance and limitations.
func (c Config) Sign(method, path, meta string, now time.Time, random io.Reader) (string, error) {
	if err := ValidateRequest(method, path, meta); err != nil {
		return "", err
	}
	header, err := c.validate(now)
	if err != nil {
		return "", err
	}
	suffix := c.Suffix
	if c.SchemaVersion == 2 {
		seed, _ := decodeMeta(meta) // ValidateRequest already enforces canonical 48-byte input.
		header = make([]byte, 49)
		copy(header[1:], seed)
		suffix = c.SuffixPrefix + c.fingerprintForSeed(seed)
	} else if meta != c.MetaContent {
		return "", ErrMetaMismatch
	}
	counter := now.Unix() - epoch
	if counter < 0 || counter > math.MaxUint32 {
		return "", ErrUnavailable
	}
	if random == nil {
		return "", ErrUnavailable
	}
	var xor [1]byte
	if _, err = io.ReadFull(random, xor[:]); err != nil {
		return "", ErrUnavailable
	}
	hash := sha256.Sum256([]byte(method + "!" + path + "!" + strconv.FormatInt(counter, 10) + suffix))
	var raw [70]byte
	copy(raw[:49], header)
	binary.LittleEndian.PutUint32(raw[49:53], uint32(counter))
	copy(raw[53:69], hash[:16])
	raw[69] = byte(*c.Trailer)
	for i := range raw {
		raw[i] ^= xor[0]
	}
	return base64.RawStdEncoding.EncodeToString(raw[:]), nil
}

// Load reads a complete snapshot on every request, so an atomic rename becomes
// visible immediately. Invalid replacements fail closed instead of using a
// cached snapshot. Mount the containing directory to preserve rename semantics.
func Load(path string, now time.Time) (Config, error) {
	f, err := os.Open(path)
	if err != nil {
		return Config{}, ErrUnavailable
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil || !info.Mode().IsRegular() || info.Mode().Perm()&0077 != 0 {
		return Config{}, ErrUnavailable
	}
	b, err := io.ReadAll(io.LimitReader(f, maxConfigBytes+1))
	if err != nil || len(b) > maxConfigBytes {
		return Config{}, ErrUnavailable
	}
	var c Config
	d := json.NewDecoder(bytes.NewReader(b))
	d.DisallowUnknownFields()
	if d.Decode(&c) != nil {
		return Config{}, ErrUnavailable
	}
	if d.Decode(new(any)) != io.EOF {
		return Config{}, ErrUnavailable
	}
	var fields map[string]json.RawMessage
	if json.Unmarshal(b, &fields) != nil {
		return Config{}, ErrUnavailable
	}
	forbidden := []string{"buildID", "suffixPrefix", "times", "fingerprints"}
	if c.SchemaVersion == 2 {
		forbidden = []string{"metaContent", "headerHex", "suffix"}
	}
	for _, field := range forbidden {
		if _, ok := fields[field]; ok {
			return Config{}, ErrUnavailable
		}
	}
	if _, err = c.validate(now); err != nil {
		return Config{}, err
	}
	return c, nil
}
