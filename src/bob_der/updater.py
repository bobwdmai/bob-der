from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from . import __version__

GITHUB_REPOSITORY = "bobwdmai/bob-der"
LATEST_RELEASE_URL = (
    f"https://api.github.com/repos/{GITHUB_REPOSITORY}/releases/latest"
)
USER_AGENT = f"bob-der/{__version__}"


class UpdateError(RuntimeError):
    pass


@dataclass(frozen=True)
class Release:
    version: str
    asset_name: str
    asset_url: str
    checksum_url: str
    page_url: str


@dataclass(frozen=True)
class UpdateOutcome:
    current_version: str
    latest_version: str
    page_url: str
    downloaded_path: Path | None = None

    @property
    def available(self) -> bool:
        return _version_key(self.latest_version) > _version_key(self.current_version)


Opener = Callable[..., Any]


def _version_key(version: str) -> tuple[int, ...]:
    cleaned = version.strip().lstrip("vV")
    parts = cleaned.split(".")
    if not parts or any(not part.isdigit() for part in parts):
        raise UpdateError(f"Unsupported release version: {version}")
    return tuple(int(part) for part in parts)


def _open(url: str, *, timeout: float, opener: Opener | None = None) -> Any:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": USER_AGENT,
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    return (opener or urllib.request.urlopen)(request, timeout=timeout)


def _read(url: str, *, timeout: float, opener: Opener | None = None) -> bytes:
    try:
        with _open(url, timeout=timeout, opener=opener) as response:
            return response.read()
    except (OSError, urllib.error.URLError) as exc:
        raise UpdateError(f"Could not fetch update information: {exc}") from exc


def latest_release(*, timeout: float = 15, opener: Opener | None = None) -> Release:
    try:
        payload = json.loads(
            _read(LATEST_RELEASE_URL, timeout=timeout, opener=opener).decode("utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UpdateError("GitHub returned invalid release information") from exc

    if not isinstance(payload, dict):
        raise UpdateError("GitHub returned invalid release information")
    version = str(payload.get("tag_name", "")).lstrip("vV")
    _version_key(version)
    assets = payload.get("assets", [])
    if not isinstance(assets, list):
        raise UpdateError("The GitHub release has no downloadable assets")

    deb_assets = [
        asset
        for asset in assets
        if isinstance(asset, dict)
        and str(asset.get("name", "")).startswith("bob-der_")
        and str(asset.get("name", "")).endswith("_all.deb")
    ]
    if not deb_assets:
        raise UpdateError("The latest GitHub release has no Ubuntu .deb package")
    asset = deb_assets[0]
    asset_name = str(asset["name"])
    asset_url = str(asset.get("browser_download_url", ""))
    checksum_name = f"{asset_name}.sha256"
    checksum_url = next(
        (
            str(item.get("browser_download_url", ""))
            for item in assets
            if isinstance(item, dict) and item.get("name") == checksum_name
        ),
        "",
    )
    if not asset_url or not checksum_url:
        raise UpdateError("The latest release is missing its package or checksum")
    return Release(
        version=version,
        asset_name=asset_name,
        asset_url=asset_url,
        checksum_url=checksum_url,
        page_url=str(payload.get("html_url", "")),
    )


def fetch_update(
    *,
    download: bool,
    destination: Path | None = None,
    timeout: float = 30,
    opener: Opener | None = None,
) -> UpdateOutcome:
    release = latest_release(timeout=timeout, opener=opener)
    outcome = UpdateOutcome(__version__, release.version, release.page_url)
    if not outcome.available or not download:
        return outcome

    target_dir = destination or (Path.home() / "Downloads")
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / release.asset_name
    if target.name != release.asset_name:
        raise UpdateError("The release contains an unsafe package filename")

    checksum_text = _read(
        release.checksum_url, timeout=timeout, opener=opener
    ).decode("ascii", errors="strict")
    fields = checksum_text.strip().split()
    if not fields or len(fields[0]) != 64:
        raise UpdateError("The release checksum is invalid")
    expected = fields[0].lower()
    if any(char not in "0123456789abcdef" for char in expected):
        raise UpdateError("The release checksum is invalid")

    package = _read(release.asset_url, timeout=timeout, opener=opener)
    actual = hashlib.sha256(package).hexdigest()
    if actual != expected:
        raise UpdateError(
            f"Downloaded package checksum mismatch (expected {expected}, got {actual})"
        )

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{target.name}.", dir=target_dir, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(package)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return UpdateOutcome(
        __version__, release.version, release.page_url, downloaded_path=target
    )


def format_outcome(outcome: UpdateOutcome) -> str:
    if not outcome.available:
        return f"bob-der {outcome.current_version} is up to date."
    if outcome.downloaded_path is None:
        return (
            f"bob-der {outcome.latest_version} is available "
            f"(installed: {outcome.current_version}).\n{outcome.page_url}"
        )
    quoted = str(outcome.downloaded_path).replace("'", "'\\''")
    return (
        f"Downloaded and verified bob-der {outcome.latest_version}:\n"
        f"{outcome.downloaded_path}\n\n"
        f"Install it with:\nsudo apt install '{quoted}'"
    )
