#!/usr/bin/env python3
"""Renew a verified signer table only while Grok's public build and curves match."""

import argparse
import datetime as dt
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import tempfile
import urllib.error
import urllib.request


PAGE_URL = "https://grok.com/"
MAX_BYTES = 4 * 1024 * 1024
PUSH = re.compile(r"self\.__next_f\.push\s*\(")


class Rejected(ValueError):
    """A public, non-sensitive reason that the table must not be renewed."""


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise Rejected("page_redirect_rejected")


def fetch_page():
    """Fetch only the fixed public page, without cookies or environment proxies."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    request = urllib.request.Request(PAGE_URL, headers={
        "User-Agent": "Mozilla/5.0", "Referer": PAGE_URL,
    })
    try:
        with opener.open(request, timeout=20) as response:
            if response.status != 200:
                raise Rejected("page_http_status_" + str(response.status))
            body = response.read(MAX_BYTES + 1)
    except urllib.error.HTTPError as exc:
        raise Rejected("page_http_status_" + str(exc.code)) from None
    except (OSError, urllib.error.URLError):
        raise Rejected("page_fetch_failed") from None
    if len(body) > MAX_BYTES:
        raise Rejected("page_too_large")
    return body


class FlightParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.inline = False
        self.script = []
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            self.inline = not any(name == "src" for name, _ in attrs)
            self.script = []

    def handle_data(self, data):
        if self.inline:
            self.script.append(data)

    def handle_endtag(self, tag):
        if tag != "script" or not self.inline:
            return
        script = ''.join(self.script)
        decoder = json.JSONDecoder()
        for match in PUSH.finditer(script):
            tail = script[match.end():].lstrip()
            try:
                item, end = decoder.raw_decode(tail)
            except (ValueError, RecursionError):
                raise Rejected("flight_payload_invalid") from None
            if not tail[end:].lstrip().startswith(")") or not isinstance(item, list):
                raise Rejected("flight_payload_invalid")
            if item and type(item[0]) is int and item[0] == 1:
                if len(item) != 2 or not isinstance(item[1], str):
                    raise Rejected("flight_payload_invalid")
                self.parts.append(item[1])
        self.inline = False
        self.script = []


def curve_digest(curves):
    if not isinstance(curves, list) or len(curves) != 4:
        raise Rejected("curves_shape_invalid")
    for group in curves:
        if not isinstance(group, list) or len(group) != 16:
            raise Rejected("curves_shape_invalid")
        for curve in group:
            if not isinstance(curve, dict) or set(curve) != {"color", "deg", "bezier"}:
                raise Rejected("curves_shape_invalid")
            color, bezier = curve["color"], curve["bezier"]
            if not isinstance(color, list) or len(color) != 6 or not isinstance(bezier, list) or len(bezier) != 4:
                raise Rejected("curves_shape_invalid")
            if any(type(value) is not int or not 0 <= value <= 255
                   for value in color + [curve["deg"]] + bezier):
                raise Rejected("curves_value_invalid")
    return hashlib.sha256(json.dumps(curves, separators=(',', ':'), sort_keys=True).encode()).hexdigest()


def parse_page(body):
    """Return the single unambiguous build and animation digest from Flight data."""
    if not isinstance(body, bytes) or len(body) > MAX_BYTES:
        raise Rejected("page_size_or_type_invalid")
    try:
        html = body.decode("utf-8")
    except UnicodeError:
        raise Rejected("page_encoding_invalid") from None
    parser = FlightParser()
    parser.feed(html)
    parser.close()
    builds, digests = set(), set()

    def walk(value):
        if isinstance(value, list):
            if len(value) == 4 and value[0] == "$" and value[2] is None and isinstance(value[3], dict):
                props = value[3]
                if props.get("css_class") == "r-1rsvi":
                    digests.add(curve_digest(props.get("curves")))
            for child in value:
                walk(child)
        elif isinstance(value, dict):
            for child in value.values():
                walk(child)

    for line in ''.join(parser.parts).splitlines():
        match = re.match(r"([0-9a-fA-F]+):(.*)$", line)
        if not match:
            continue
        record_id, payload = match.groups()
        try:
            value = json.loads(payload)
        except (ValueError, RecursionError):
            if record_id == "0":
                raise Rejected("build_record_invalid") from None
            # Flight also carries tagged module, hint and text records.
            continue
        if record_id == "0":
            build = value.get("b") if isinstance(value, dict) else None
            if not isinstance(build, str) or not build or len(build) > 256:
                raise Rejected("build_record_invalid")
            builds.add(build)
        walk(value)
    if len(builds) != 1:
        raise Rejected("build_missing_or_conflicting")
    if len(digests) != 1:
        raise Rejected("curves_missing_or_conflicting")
    return builds.pop(), digests.pop()


def protected_output(template, manifest, output):
    for source in (template, manifest):
        if output.resolve() == source.resolve():
            return True
        if output.exists() and source.exists() and output.samefile(source):
            return True
    return False


def write_atomic(output, config):
    temporary = None
    try:
        descriptor, temporary = tempfile.mkstemp(prefix=".page-guard-", dir=output.parent)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            os.fchmod(handle.fileno(), 0o600)
            json.dump(config, handle, separators=(',', ':'), allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def refresh(template, manifest, output, *, ttl_seconds=900, fetch=None, now=None):
    """Validate and atomically renew the table; revoke output on any failure.

    Input aliases are never unlinked. ``fetch`` is solely an offline test seam;
    the CLI always uses the fixed public URL and has no endpoint override.
    """
    template, manifest, output = map(Path, (template, manifest, output))
    can_revoke = False
    try:
        if protected_output(template, manifest, output):
            raise Rejected("output_conflicts_with_input")
        can_revoke = True
        if output.is_symlink():
            raise Rejected("output_symlink_rejected")
        if type(ttl_seconds) is not int or not 60 <= ttl_seconds <= 3600:
            raise Rejected("ttl_out_of_range")
        config = json.loads(template.read_text(encoding="utf-8"))
        expected = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(expected, dict) or not isinstance(expected.get("buildID"), str) or not expected["buildID"]:
            raise Rejected("manifest_invalid")
        if not isinstance(expected.get("curvesSHA256"), str) or not re.fullmatch(r"[0-9a-f]{64}", expected["curvesSHA256"]):
            raise Rejected("manifest_invalid")
        if not isinstance(config, dict) or type(config.get("schemaVersion")) is not int or config["schemaVersion"] != 2:
            raise Rejected("template_schema_invalid")
        if config.get("buildID") != expected["buildID"]:
            raise Rejected("template_build_mismatch")
        build, digest = parse_page((fetch or fetch_page)())
        if build != expected["buildID"]:
            raise Rejected("page_build_changed")
        if digest != expected["curvesSHA256"]:
            raise Rejected("page_curves_changed")
        checked_at = now if now is not None else dt.datetime.now(dt.timezone.utc)
        if checked_at.tzinfo is None:
            raise Rejected("clock_timezone_missing")
        config["expiresAt"] = (checked_at.astimezone(dt.timezone.utc) + dt.timedelta(seconds=ttl_seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")
        write_atomic(output, config)
        return {"status": "updated", "buildID": build, "curvesSHA256": digest,
                "expiresAt": config["expiresAt"]}
    except Exception as exc:
        reason = str(exc) if isinstance(exc, Rejected) else "validation_or_io_failed"
        revoked = False
        if can_revoke:
            try:
                output.unlink(missing_ok=True)
                revoked = True
            except OSError:
                reason += ":output_revoke_failed"
        return {"status": "rejected", "reason": reason, "revoked": revoked}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ttl-seconds", default="900")
    args = parser.parse_args()
    try:
        ttl = int(args.ttl_seconds)
    except ValueError:
        ttl = None
    result = refresh(args.template, args.manifest, args.output, ttl_seconds=ttl)
    print(json.dumps(result, separators=(',', ':')))
    return 0 if result["status"] == "updated" else 1


if __name__ == "__main__":
    raise SystemExit(main())
