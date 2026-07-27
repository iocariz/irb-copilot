"""Tokenization helpers (tiktoken-backed) used by chunking and indexing.

The chunker takes the token counter as a parameter so its merge/split logic can
be unit-tested deterministically without depending on tiktoken's exact vocab.
"""

from __future__ import annotations

import re
from functools import lru_cache

import tiktoken

_ENCODING_NAME = "cl100k_base"

# Split on sentence-final punctuation followed by whitespace. Deliberately
# simple (no nltk): good enough for hard-splitting oversized chunks.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

# Stray separator/control characters that PDF extraction leaves in text. Notably
# U+0085/U+2028/U+2029 are NOT escaped by pydantic's UTF-8 JSON output yet are
# treated as line boundaries by str.splitlines(), which would corrupt JSONL. We
# normalise them to spaces so stored text is clean and one chunk == one line.
_STRAY_SEPARATORS = re.compile("[\x0b\x0c\x1c\x1d\x1e\x1f\x85  ]")


@lru_cache(maxsize=1)
def _encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding(_ENCODING_NAME)


def count_tokens(text: str) -> int:
    """Number of tokens in `text` under the default encoding."""
    return len(_encoder().encode(text))


def encode(text: str) -> list[int]:
    return _encoder().encode(text)


def decode(token_ids: list[int]) -> str:
    return _encoder().decode(token_ids)


def normalize_text(text: str) -> str:
    r"""Replace stray separator/control characters with spaces (keeps \n, \t)."""
    return _STRAY_SEPARATORS.sub(" ", text)


def split_sentences(text: str) -> list[str]:
    """Split into sentences; never returns empty strings."""
    parts = [s.strip() for s in _SENTENCE_RE.split(text)]
    return [s for s in parts if s]
