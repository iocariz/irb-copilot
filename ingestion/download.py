"""Download stage (SPEC §4, §5 step 1).

Fetches each registered PDF, verifies its sha256 (populating the registry hash
on first successful download), and is idempotent — an already-present file whose
hash matches is left untouched. A 404 (or any HTTP error) fails loudly with the
document id; nothing is silently skipped.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import httpx
import yaml

_TIMEOUT = httpx.Timeout(60.0)
_CHUNK = 1 << 16


@dataclass(frozen=True)
class Source:
    """One registered corpus document."""

    id: str
    title: str
    url: str
    sha256: str | None


def load_sources(path: Path) -> list[Source]:
    """Load and validate `data/sources.yaml`."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not raw or "documents" not in raw:
        raise ValueError(f"{path} must contain a top-level 'documents' list")
    return [
        Source(
            id=d["id"],
            title=d["title"],
            url=d["url"],
            sha256=d.get("sha256"),
        )
        for d in raw["documents"]
    ]


def download_all(
    sources: list[Source],
    raw_dir: Path,
    *,
    only_doc: str | None = None,
) -> dict[str, str]:
    """Download each source into `raw_dir`; return {doc_id: sha256}."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for source in sources:
        if only_doc and source.id != only_doc:
            continue
        hashes[source.id] = _download_one(source, raw_dir)
    return hashes


def _download_one(source: Source, raw_dir: Path) -> str:
    dest = raw_dir / f"{source.id}.pdf"

    if dest.exists():
        digest = _sha256_file(dest)
        if source.sha256 is None or digest == source.sha256:
            print(f"[download] {source.id}: present ({digest[:12]}…), skipping")
            return digest
        raise ValueError(
            f"[download] {source.id}: sha256 mismatch "
            f"(expected {source.sha256[:12]}…, got {digest[:12]}…); delete to re-fetch"
        )

    print(f"[download] {source.id}: fetching {source.url}")
    _fetch(source.url, dest, doc_id=source.id)
    digest = _sha256_file(dest)

    if source.sha256 and digest != source.sha256:
        dest.unlink(missing_ok=True)
        raise ValueError(
            f"[download] {source.id}: sha256 mismatch after download "
            f"(expected {source.sha256[:12]}…, got {digest[:12]}…)"
        )
    if source.sha256 is None:
        print(f"[download] {source.id}: record this sha256 in sources.yaml -> {digest}")
    return digest


def _fetch(url: str, dest: Path, *, doc_id: str) -> None:
    try:
        with httpx.stream("GET", url, timeout=_TIMEOUT, follow_redirects=True) as resp:
            resp.raise_for_status()
            with dest.open("wb") as fh:
                for block in resp.iter_bytes(_CHUNK):
                    fh.write(block)
    except httpx.HTTPError as exc:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"[download] {doc_id}: failed to fetch {url} — {exc}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()
