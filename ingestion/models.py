"""Data models shared across ingestion stages.

These are the persisted, on-disk intermediate representations that make each
stage independently runnable and resumable (SPEC §17):

    download -> data/raw/<doc_id>.pdf
    parse    -> data/parsed/<doc_id>.json      (ParsedDoc)
    chunk    -> data/chunks/<doc_id>.<kind>.jsonl (Chunk per line)
    index    -> Qdrant collection + on-disk BM25 index
"""

from __future__ import annotations

import uuid

from pydantic import BaseModel, Field

# Fixed namespace so chunk ids are deterministic across runs and machines.
_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "irb-copilot")


class Paragraph(BaseModel):
    """A single numbered paragraph extracted from a regulatory document."""

    para_id: str
    text: str
    section_path: list[str] = Field(default_factory=list)
    page: int


class ParsedDoc(BaseModel):
    """A document after parsing: title + ordered numbered paragraphs."""

    doc_id: str
    doc_title: str
    paragraphs: list[Paragraph]


class Chunk(BaseModel):
    """An indexable unit with its citation metadata (SPEC §5)."""

    chunk_id: str
    doc_id: str
    doc_title: str
    section_path: list[str] = Field(default_factory=list)
    para_ids: list[str] = Field(default_factory=list)
    pages: list[int] = Field(default_factory=list)
    text: str


def chunk_id_for(doc_id: str, para_ids: list[str], ordinal: int) -> str:
    """Deterministic id = uuid5(doc_id + para_ids + ordinal).

    `ordinal` is the chunk's position within the document. It is required for
    uniqueness because regulatory documents restart paragraph numbering in every
    section (e.g. paragraph "1" recurs dozens of times), so `para_ids` alone is
    not unique within a document. The ordinal is stable across identical re-runs
    (parsing and chunking are deterministic), so re-indexing upserts rather than
    duplicates; a changed corpus should be re-indexed with `--recreate`.
    """
    seed = f"{doc_id}|{','.join(para_ids)}|{ordinal}"
    return str(uuid.uuid5(_NAMESPACE, seed))
