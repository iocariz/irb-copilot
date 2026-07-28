"""Download stage (SPEC §4, §5 step 1).

Fetches each registered PDF, verifies its sha256 (populating the registry hash
on first successful download), and is idempotent — an already-present file whose
hash matches is left untouched. A 404 (or any HTTP error) fails loudly with the
document id; nothing is silently skipped.

Downloads are written to a temp ``.part`` file and promoted atomically only
after they validate, so an interrupted transfer never leaves a partial file at
the canonical path. When no sha256 is pinned yet (content can't be verified), a
present or freshly downloaded file must still be a structurally complete PDF, so
a truncated/corrupt file is rejected rather than trusted.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import httpx
import yaml

_TIMEOUT = httpx.Timeout(60.0)
_CHUNK = 1 << 16

# Structural markers of a complete PDF: it starts with %PDF- and ends with %%EOF.
# A truncated/interrupted download (or an HTML error page served with HTTP 200)
# fails this — the cheapest way to reject a "valid" corrupt file when no sha256
# is pinned to verify content against.
_PDF_HEADER = b"%PDF-"
_PDF_TRAILER = b"%%EOF"
_MIN_PDF_BYTES = 1024


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
        if source.sha256 is not None:
            if digest == source.sha256:
                print(f"[download] {source.id}: present ({digest[:12]}…), skipping")
                return digest
            raise ValueError(
                f"[download] {source.id}: sha256 mismatch "
                f"(expected {source.sha256[:12]}…, got {digest[:12]}…); delete to re-fetch"
            )
        # No pinned hash to verify content: an interrupted earlier download could
        # have left a partial file here. Don't trust it unless it is a structurally
        # complete PDF; fail loudly rather than silently feeding corruption downstream.
        if not _looks_like_pdf(dest):
            raise ValueError(
                f"[download] {source.id}: existing file is not a complete PDF "
                f"(truncated or corrupt) and no sha256 is pinned to verify it; "
                f"delete {dest} to re-fetch"
            )
        print(f"[download] {source.id}: present ({digest[:12]}…, unverified), skipping")
        return digest

    print(f"[download] {source.id}: fetching {source.url}")
    # Download to a temp path and promote atomically, so an interrupted transfer
    # never leaves a partial file at the canonical path for the next run to trust.
    tmp = dest.with_name(f"{source.id}.pdf.part")
    _fetch(source.url, tmp, doc_id=source.id)

    if not _looks_like_pdf(tmp):
        tmp.unlink(missing_ok=True)
        raise ValueError(
            f"[download] {source.id}: downloaded file is not a valid PDF "
            f"(truncated response or an error page served as 200)"
        )
    digest = _sha256_file(tmp)
    if source.sha256 and digest != source.sha256:
        tmp.unlink(missing_ok=True)
        raise ValueError(
            f"[download] {source.id}: sha256 mismatch after download "
            f"(expected {source.sha256[:12]}…, got {digest[:12]}…)"
        )

    tmp.replace(dest)  # atomic promote — only reached once all checks pass
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


def _looks_like_pdf(path: Path) -> bool:
    """True only for a structurally complete PDF (``%PDF-`` header, ``%%EOF``
    trailer, and a plausible minimum size). Truncated or non-PDF files fail."""
    size = path.stat().st_size
    if size < _MIN_PDF_BYTES:
        return False
    with path.open("rb") as fh:
        if fh.read(len(_PDF_HEADER)) != _PDF_HEADER:
            return False
        fh.seek(-_MIN_PDF_BYTES, 2)
        tail = fh.read()
    return _PDF_TRAILER in tail


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()
