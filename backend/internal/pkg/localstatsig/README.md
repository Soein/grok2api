# Local Statsig signer prototype

The 70-byte challenge algorithm is adapted from the MIT-licensed
[`imjustprism/grok-web-api` challenge.rs](https://github.com/imjustprism/grok-web-api/blob/8ae029d701db1ef569127c2fbbb7aa147fed97b9/crates/grok-client/src/challenge.rs),
commit `8ae029d701db1ef569127c2fbbb7aa147fed97b9`.
The upstream license is reproduced in `LICENSE` in this directory.

This is an isolated signature prototype. Passing format tests or `/readyz`
does **not** prove that Grok accepts a signature or that an account can generate
video. Current browser parameters and an independent upstream acceptance test
are required before production use. No default or captured live constants are
embedded in the source.

## Configuration contract

Both schemas require `schemaVersion`, an explicit `trailer`, and an `expiresAt`
RFC3339 timestamp. Files are limited to 2 MiB. Schema-specific fields may not be
mixed, even when a forbidden field is empty. Unknown fields are rejected.

### Schema 1: one normalized meta snapshot

The private JSON file contains these required fields:

| Field | Constraint |
|---|---|
| `schemaVersion` | Integer `1` |
| `metaContent` | Canonical standard base64 encoding of 48 bytes |
| `headerHex` | 49 bytes encoded in hex; first byte `00`, remaining bytes exactly match decoded `metaContent` |
| `suffix` | Complete hash suffix obtained from the matching browser build; 1–4096 bytes |
| `trailer` | Integer 0–255; explicitly required |
| `expiresAt` | RFC3339 timestamp; signatures are refused at/after this time |

Obtain the normalized header with the browser's random XOR byte set to zero;
arbitrary captured, already-XORed headers are rejected. Parameter extraction is
outside this command. The expiry is an operator-imposed validity window and
does not predict when Grok will change its protocol.

### Schema 2: browser fingerprint table for a build

This schema handles different meta values using the current build's complete
browser-generated CSS fingerprint table. It does not approximate browser float,
color or transform serialization in Go. It has these required fields:

| Field | Constraint |
|---|---|
| `schemaVersion` | Integer `2` |
| `buildID` | Nonempty build identifier, at most 256 bytes; no surrounding whitespace or control characters |
| `suffixPrefix` | Exactly `obfiowerehiring` |
| `trailer` | Exactly `3` |
| `expiresAt` | RFC3339 timestamp; signatures are refused at/after this time |
| `times` | All 182 legal animation times, strictly increasing |
| `fingerprints` | Exactly `[4][16][182]` strings, indexed by set, row and position in `times`; each string contains 1–256 lowercase hexadecimal characters |

The legal time set is every distinct value of `round(a*b*c/10)*10` for integer
`a`, `b`, `c` in `[0,15]`. All entries must be present, including currently unused
frames; the range is 0–3380. The table must come from the matching browser build.
`metaContent`, `headerHex`, and `suffix` are forbidden in schema 2.

Each request's canonical base64 meta decodes to 48 bytes. The header is zero
followed by those bytes. The fingerprint selection is:

```text
set = seed[5] % 4
row = seed[4] % 16
time = round((seed[44]%16)*(seed[40]%16)*(seed[3]%16)/10)*10
suffix = suffixPrefix + fingerprints[set][row][indexOf(times, time)]
```

The finite table and selector were verified against the current official module
and browser CSS output on 2026-09-10. Unit tests include an independent official
module signature vector. This proves parity for that captured build, not ongoing
upstream acceptance. An external collector/validator must verify the current
build and curve data and publish an expired snapshot if they change. This binary
does not fetch a webpage, check a live build ID, generate tables, or extend expiry.

### Publishing either schema

The file must be a regular file with no group/other permission bits (for example
0600 or 0400). Each request opens, bounds, parses and validates the entire file.
Publish updates by writing a temporary file in the same directory and atomically
renaming it. Mount the **directory**, not the individual file, into Docker so
atomic replacements are visible. Invalid, missing, expired or mismatching
configuration returns HTTP 503; no previous snapshot is reused.

## Command and endpoints

Build from `backend` with `go build -o grok-signer ./cmd/grok-signer` and run:

```sh
./grok-signer --listen 127.0.0.1:8788 --config /private/params.json
```

For Docker use `--listen 0.0.0.0:8788` only on a private container network; do not
publish this unauthenticated helper to the Internet. It has no upstream network
client, SSO credentials, logging of request bodies, or logging of signatures.

- `GET /healthz`: process liveness, independent of configuration.
- `GET /readyz`: currently valid configuration, with `scope: "configuration"`;
  it does not independently check that `buildID` is still current.
- `POST /sign`: accepts at most 8192 body bytes using the existing Grok2API
  `{method, path, environment: {metaContent}}` contract. Method must be uppercase;
  path is absolute without a query or fragment. On success returns
  `{"x-statsig-id":"<70-byte raw-standard-base64 signature>"}` with `no-store`.

The server limits request headers and read/write/idle timeouts. Shutdown on
SIGTERM/SIGINT allows five seconds for in-flight HTTP requests to finish.
