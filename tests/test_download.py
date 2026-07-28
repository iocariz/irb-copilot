"""Download-stage tests (SPEC §4).

Network is stubbed by monkeypatching ``_fetch``; these exercise the idempotency,
atomic-promote, and integrity-gate logic around it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ingestion import download
from ingestion.download import Source, _download_one, _looks_like_pdf


def _pdf_bytes(pad: int = 2048) -> bytes:
    """A structurally complete, comfortably-over-minimum PDF."""
    return b"%PDF-1.7\n" + b"x" * pad + b"\n%%EOF\n"


_TRUNCATED = b"%PDF-1.7\n" + b"x" * 4096  # header but no %%EOF trailer


def _source(sha256: str | None = None) -> Source:
    return Source(id="doc", title="T", url="http://example/doc.pdf", sha256=sha256)


def _stub_fetch(monkeypatch, payload: bytes) -> None:
    """Make ``_fetch`` write ``payload`` to its temp dest instead of hitting the network."""

    def fake(url: str, dest: Path, *, doc_id: str) -> None:
        dest.write_bytes(payload)

    monkeypatch.setattr(download, "_fetch", fake)


# --- _looks_like_pdf -------------------------------------------------------- #
def test_looks_like_pdf_accepts_complete_file(tmp_path: Path) -> None:
    p = tmp_path / "ok.pdf"
    p.write_bytes(_pdf_bytes())
    assert _looks_like_pdf(p)


def test_looks_like_pdf_rejects_truncated_and_non_pdf(tmp_path: Path) -> None:
    truncated = tmp_path / "trunc.pdf"
    truncated.write_bytes(_TRUNCATED)
    assert not _looks_like_pdf(truncated)

    tiny = tmp_path / "tiny.pdf"
    tiny.write_bytes(b"%PDF-1.7\n%%EOF\n")  # below the minimum size
    assert not _looks_like_pdf(tiny)

    html = tmp_path / "err.pdf"
    html.write_bytes(b"<html>error</html>" + b" " * 2048)  # wrong header
    assert not _looks_like_pdf(html)


# --- fresh download (atomic promote + validation) --------------------------- #
def test_download_promotes_atomically_on_success(tmp_path: Path, monkeypatch) -> None:
    _stub_fetch(monkeypatch, _pdf_bytes())
    digest = _download_one(_source(), tmp_path)
    assert (tmp_path / "doc.pdf").read_bytes() == _pdf_bytes()
    assert not (tmp_path / "doc.pdf.part").exists()  # temp cleaned up by rename
    assert digest == hashlib.sha256(_pdf_bytes()).hexdigest()


def test_download_truncated_response_raises_and_leaves_no_file(tmp_path: Path, monkeypatch) -> None:
    _stub_fetch(monkeypatch, _TRUNCATED)
    with pytest.raises(ValueError, match="not a valid PDF"):
        _download_one(_source(), tmp_path)
    assert not (tmp_path / "doc.pdf").exists()  # nothing promoted
    assert not (tmp_path / "doc.pdf.part").exists()  # temp removed


def test_download_sha256_mismatch_raises_and_leaves_no_file(tmp_path: Path, monkeypatch) -> None:
    _stub_fetch(monkeypatch, _pdf_bytes())
    with pytest.raises(ValueError, match="mismatch after download"):
        _download_one(_source(sha256="deadbeef"), tmp_path)
    assert not (tmp_path / "doc.pdf").exists()
    assert not (tmp_path / "doc.pdf.part").exists()


# --- pre-existing files ----------------------------------------------------- #
def test_existing_corrupt_file_not_trusted_without_sha256(tmp_path: Path) -> None:
    # The reported bug: an interrupted earlier download left a partial file and,
    # with sha256 unpinned, the next run must not skip it as "present".
    (tmp_path / "doc.pdf").write_bytes(_TRUNCATED)
    with pytest.raises(ValueError, match="not a complete PDF"):
        _download_one(_source(), tmp_path)


def test_existing_valid_file_skipped_without_sha256(tmp_path: Path) -> None:
    (tmp_path / "doc.pdf").write_bytes(_pdf_bytes())
    digest = _download_one(_source(), tmp_path)
    assert digest == hashlib.sha256(_pdf_bytes()).hexdigest()


def test_existing_file_with_matching_sha256_is_skipped(tmp_path: Path) -> None:
    (tmp_path / "doc.pdf").write_bytes(_pdf_bytes())
    expected = hashlib.sha256(_pdf_bytes()).hexdigest()
    assert _download_one(_source(sha256=expected), tmp_path) == expected


def test_existing_file_with_mismatched_sha256_raises(tmp_path: Path) -> None:
    (tmp_path / "doc.pdf").write_bytes(_pdf_bytes())
    with pytest.raises(ValueError, match="sha256 mismatch"):
        _download_one(_source(sha256="deadbeef"), tmp_path)
