import datetime as dt
import hashlib
import json
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import page_guard


def curves():
    return [[{"color": [1, 2, 3, 4, 5, 6], "deg": row,
              "bezier": [7, 8, 9, 10]} for row in range(16)] for _ in range(4)]


def page(build="build-a", shape=None, extra=""):
    shape = curves() if shape is None else shape
    flight = '0:' + json.dumps({"b": build}) + '\n'
    flight += '9:' + json.dumps(["$", "$L1", None,
                               {"curves": shape, "css_class": "r-1rsvi"}]) + '\n' + extra
    # Split inside a JSON token, as Flight chunks need not end on record boundaries.
    parts = [flight[:13], flight[13:70], flight[70:]]
    return ''.join('<script>self.__next_f.push(' + json.dumps([1, part]) + ')</script>'
                   for part in parts).encode()


class PageGuardTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.template, self.manifest, self.output = [root / name for name in
                                                   ("template.json", "manifest.json", "params.json")]
        self.template.write_text(json.dumps({"schemaVersion": 2, "buildID": "build-a",
                                            "fingerprints": [[["abc"]]], "expiresAt": "old"}))
        digest = hashlib.sha256(json.dumps(curves(), separators=(',', ':'),
                                          sort_keys=True).encode()).hexdigest()
        self.manifest.write_text(json.dumps({"buildID": "build-a", "curvesSHA256": digest,
                                            "provenance": "accepted extra field"}))
        self.now = dt.datetime(2026, 9, 10, 12, 0, tzinfo=dt.timezone.utc)

    def run_guard(self, content=None, **kwargs):
        return page_guard.refresh(self.template, self.manifest, self.output,
                                  fetch=lambda: page() if content is None else content,
                                  now=self.now, **kwargs)

    def assert_revoked(self, content):
        self.output.write_text("old still valid configuration")
        result = self.run_guard(content)
        self.assertEqual(result["status"], "rejected")
        self.assertTrue(result["revoked"])
        self.assertFalse(self.output.exists())

    def test_segmented_flight_atomic_update_and_permissions(self):
        original = self.template.read_bytes()
        self.output.write_text("old configuration")
        replace = page_guard.os.replace
        observed = []
        def verify_replace(source, target):
            observed.append(Path(target).read_text())
            self.assertEqual(stat.S_IMODE(Path(source).stat().st_mode), 0o600)
            replace(source, target)
        with mock.patch.object(page_guard.os, "replace", side_effect=verify_replace):
            result = self.run_guard()
        self.assertEqual(result["status"], "updated")
        self.assertEqual(observed, ["old configuration"])
        self.assertEqual(self.template.read_bytes(), original)
        output = json.loads(self.output.read_text())
        self.assertEqual(output["expiresAt"], "2026-09-10T12:15:00Z")
        self.assertEqual(output["fingerprints"], [[["abc"]]])
        self.assertEqual(stat.S_IMODE(self.output.stat().st_mode), 0o600)
        self.assertEqual(set(p.name for p in self.output.parent.iterdir()),
                         {"template.json", "manifest.json", "params.json"})

    def test_build_change_revokes_existing_output(self):
        self.assert_revoked(page("build-b"))

    def test_curve_change_revokes_existing_output(self):
        changed = curves()
        changed[0][0]["color"][0] = 200
        self.assert_revoked(page(shape=changed))

    def test_invalid_payloads_revoke_existing_output(self):
        invalid = []
        for mutate in [lambda x: x.pop(), lambda x: x[0].pop(),
                       lambda x: x[0][0]["color"].append(1),
                       lambda x: x[0][0].update(deg=True),
                       lambda x: x[0][0].update(deg=256),
                       lambda x: x[0][0].update(deg=1.5),
                       lambda x: x[0][0].update(extra=0)]:
            shape = curves()
            mutate(shape)
            invalid.append(page(shape=shape))
        invalid += [b"<html>challenge</html>", b'<script>self.__next_f.push([1,"broken)</script>',
                    b"x" * (4 * 1024 * 1024 + 1)]
        for content in invalid:
            with self.subTest(size=len(content)):
                self.assert_revoked(content)

    def test_conflicting_duplicates_rejected_identical_allowed(self):
        self.assert_revoked(page(extra='0:{"b":"other"}\n'))
        other = curves()
        other[0][0]["deg"] = 200
        record = lambda value: 'a:' + json.dumps(["$", "$L2", None,
                           {"css_class": "r-1rsvi", "curves": value}]) + '\n'
        self.assert_revoked(page(extra=record(other)))
        self.assertEqual(self.run_guard(page(extra=record(curves())))['status'], 'updated')

    def test_output_conflicts_do_not_delete_inputs(self):
        for source in [self.template, self.manifest]:
            original = source.read_bytes()
            result = page_guard.refresh(self.template, self.manifest, source, fetch=lambda: page(), now=self.now)
            self.assertEqual(result["status"], "rejected")
            self.assertEqual(source.read_bytes(), original)
        self.output.hardlink_to(self.template)
        result = self.run_guard()
        self.assertEqual(result["status"], "rejected")
        self.assertTrue(self.template.exists())

    def test_template_build_and_schema_must_match(self):
        for update in [{"buildID": "other"}, {"schemaVersion": 1}, {"schemaVersion": True}]:
            template = json.loads(self.template.read_text())
            template.update(update)
            self.template.write_text(json.dumps(template))
            self.assert_revoked(page())

    def test_fetch_failure_and_invalid_ttl_revoke(self):
        self.output.write_text("old")
        with mock.patch.object(page_guard, "fetch_page", side_effect=OSError("sensitive URL text")):
            result = page_guard.refresh(self.template, self.manifest, self.output, now=self.now)
        self.assertEqual(result['status'], 'rejected')
        self.assertNotIn('sensitive', json.dumps(result))
        self.assertFalse(self.output.exists())
        for ttl in [59, 3601]:
            self.output.write_text("old")
            self.assertEqual(self.run_guard(ttl_seconds=ttl)['status'], 'rejected')
            self.assertFalse(self.output.exists())

    def test_atomic_write_failure_revokes_and_cleans_temporary_file(self):
        self.output.write_text("old")
        with mock.patch.object(page_guard.os, "replace", side_effect=OSError("private")):
            result = self.run_guard()
        self.assertEqual(result['status'], 'rejected')
        self.assertFalse(self.output.exists())
        self.assertEqual(set(p.name for p in self.output.parent.iterdir()),
                         {"template.json", "manifest.json"})

    def test_cli_invalid_ttl_revokes_without_network(self):
        self.output.write_text("old")
        result = subprocess.run([sys.executable, str(Path(page_guard.__file__)),
                                 "--template", str(self.template), "--manifest", str(self.manifest),
                                 "--output", str(self.output), "--ttl-seconds", "not-an-integer"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stdout)['status'], 'rejected')
        self.assertFalse(self.output.exists())

    def test_fetch_has_fixed_origin_no_proxy_cookie_or_redirect(self):
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.status = 200
        response.read.return_value = page()
        opener = mock.Mock()
        opener.open.return_value = response
        with mock.patch.object(page_guard.urllib.request, 'build_opener', return_value=opener) as build:
            self.assertEqual(page_guard.fetch_page(), page())
        handlers = build.call_args.args
        self.assertEqual(handlers[0].proxies, {})
        request = opener.open.call_args.args[0]
        self.assertEqual(request.full_url, 'https://grok.com/')
        self.assertEqual(request.get_method(), 'GET')
        self.assertEqual(dict(request.header_items()),
                         {'User-agent': 'Mozilla/5.0', 'Referer': 'https://grok.com/'})
        self.assertEqual(opener.open.call_args.kwargs, {'timeout': 20})
        response.read.assert_called_once_with(4 * 1024 * 1024 + 1)
        with self.assertRaises(page_guard.Rejected):
            handlers[1].redirect_request(None, None, 302, 'redirect', {}, 'https://example.com/')

    def test_fetch_rejects_status_and_oversized_body(self):
        response = mock.MagicMock()
        response.__enter__.return_value = response
        opener = mock.Mock()
        opener.open.return_value = response
        with mock.patch.object(page_guard.urllib.request, 'build_opener', return_value=opener):
            response.status = 503
            with self.assertRaisesRegex(page_guard.Rejected, 'page_http_status_503'):
                page_guard.fetch_page()
            response.status = 200
            response.read.return_value = b'x' * (4 * 1024 * 1024 + 1)
            with self.assertRaisesRegex(page_guard.Rejected, 'page_too_large'):
                page_guard.fetch_page()


if __name__ == '__main__':
    unittest.main()
