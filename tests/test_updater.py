from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from bob_der.updater import UpdateError, fetch_update, format_outcome


class FakeResponse(io.BytesIO):
    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class UpdaterTests(unittest.TestCase):
    def opener_for(self, package: bytes, checksum: str | None = None):
        asset_name = "bob-der_0.9.7_all.deb"
        payload = {
            "tag_name": "v0.9.7",
            "html_url": "https://github.com/bobwdmai/bob-der/releases/tag/v0.9.7",
            "assets": [
                {
                    "name": asset_name,
                    "browser_download_url": "https://download.test/package",
                },
                {
                    "name": f"{asset_name}.sha256",
                    "browser_download_url": "https://download.test/checksum",
                },
            ],
        }
        responses = {
            "https://api.github.com/repos/bobwdmai/bob-der/releases/latest": json.dumps(
                payload
            ).encode(),
            "https://download.test/package": package,
            "https://download.test/checksum": (
                checksum or hashlib.sha256(package).hexdigest()
            ).encode(),
        }

        def opener(request, *, timeout):
            self.assertGreater(timeout, 0)
            return FakeResponse(responses[request.full_url])

        return opener

    def test_check_reports_new_release_without_downloading(self) -> None:
        outcome = fetch_update(download=False, opener=self.opener_for(b"package"))
        self.assertTrue(outcome.available)
        self.assertIsNone(outcome.downloaded_path)
        self.assertIn("0.9.7 is available", format_outcome(outcome))

    def test_download_verifies_checksum_and_writes_package(self) -> None:
        package = b"valid deb package fixture"
        with tempfile.TemporaryDirectory() as directory:
            outcome = fetch_update(
                download=True,
                destination=Path(directory),
                opener=self.opener_for(package),
            )
            self.assertEqual(outcome.downloaded_path.read_bytes(), package)
            self.assertIn("sudo apt install", format_outcome(outcome))

    def test_download_rejects_checksum_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(UpdateError, "checksum mismatch"):
                fetch_update(
                    download=True,
                    destination=Path(directory),
                    opener=self.opener_for(b"package", checksum="0" * 64),
                )


if __name__ == "__main__":
    unittest.main()
