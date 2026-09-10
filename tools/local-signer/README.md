# Local Statsig signer

The Go service implements the existing `/sign` contract using public signing parameters. It never needs account cookies or a proxy. See `backend/internal/pkg/localstatsig/README.md` for the algorithm and schema; the MIT attribution must accompany binary distributions.

Build from repository root with `docker build -f tools/local-signer/Dockerfile .`. Compose requires a pinned `GROK_SIGNER_IMAGE` and the existing `GROK2API_NETWORK`. It publishes no host port. Set the Web signer URL to `http://grok-signer:8788/sign`; internal signer signatures cache for one minute and refresh failures do not reuse stale signatures.

Install the guard script and units under `/opt/grok-signer` and `/etc/systemd/system`. Supply a verified schema-2 `parameters/template.json` and matching `parameters/manifest.json`. The manifest records buildID and the SHA256 of canonical public curves JSON. Templates contain a complete browser-generated 4×16×182 fingerprint table; no credentials belong in these files. Keep provenance and the browser oracle results with deployment evidence.

Provision a dedicated, non-login system account and group with UID/GID 10001 on the host before starting the systemd service (first verify those IDs are available or already belong to the intended service). Numeric systemd User/Group settings still require resolvable account records on some hosts. Use UID/GID 10001 for both signer and guard. The parent directories must be traversable, input files readable (0400), and `runtime` writable (0700) by that UID. The output is atomically replaced at mode 0600; mount its directory, not the individual file. Run the oneshot as its configured user and verify `/readyz` before enabling the timer.

The guard fetches only the anonymous public homepage, without cookies, environment proxy, or redirects. Every five minutes it compares the public build and curve digest, then renews a 15-minute parameter expiry. Fetch/parse/mismatch failures revoke the output. Configuration readiness is not proof of upstream acceptance: perform a separately authorized real request when introducing a new parameter table.

Normal revocation can take about five minutes to discover plus one minute of app cache. If the guard stops, expiry limits use to about 15 minutes plus one minute of cache. Changed upstream builds require regenerating and verifying a table; this guard deliberately does not run downloaded JavaScript or silently accept new parameters.

Rollback the app image and signer URL independently, preserving the current database and rotated credentials. Restoring the external URL will also restore its external caching policy; it does not repair an unavailable external signing service.
