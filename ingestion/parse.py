"""PDF parsing (SPEC §5, step 2).

Primary parser is docling, which recovers real document structure: section
headings, numbered list items (whose markers are the citation anchors), tables,
and footnotes. We consume that structure directly rather than re-deriving it
from flat text. If docling is unavailable or fails on a document, we fall back
to a pymupdf text extraction plus a line-based numbered-paragraph heuristic.

Both paths converge on a `ParsedDoc` of numbered paragraphs — the paragraph
numbers are the citation anchors (e.g. `EBA/GL/2020/06, para. 26`).

docling is an optional dependency (`uv sync --extra parse`); it is imported
lazily so the light install works for everything that does not parse PDFs.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from pathlib import Path

from .models import Paragraph, ParsedDoc

# --- pymupdf fallback heuristics -------------------------------------------- #
# A numbered-paragraph marker at the start of a line, e.g.
#   "82. ...", "82 ...", "Article 178(1)(b) ...", "5.3.2 ..."
_PARA_MARKER_RE = re.compile(
    r"^\s*("
    r"Article\s+\d+[\dA-Za-z().]*"  # Article 178(1)(b)
    r"|\d+(?:\.\d+)*\.?"  # 82.  or  5.3.2
    r")\s+(?=\S)"
)
# Lines that are pure section headings (short, no trailing sentence).
_HEADING_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\s+([A-Z][^.]{3,80})$")

# --- docling structure heuristics ------------------------------------------- #
# A list-item marker that is a bare paragraph number: "26." / "26".
_NUMERIC_MARKER_RE = re.compile(r"^(\d+)\.?$")
# A numbered paragraph mislabeled by docling as a section header: "15. In ...".
_INLINE_NUM_RE = re.compile(r"^(\d+)\.\s+(?=\S)")
# Item labels that never contribute body paragraphs.
_SKIP_LABELS = frozenset(
    {"picture", "footnote", "caption", "document_index", "page_header", "page_footer"}
)
# Numbered section headers longer than this are treated as body paragraphs, not
# headings (docling occasionally mislabels a long numbered paragraph as a header).
_HEADING_MAX_LEN = 60


def parse_pdf(pdf_path: Path, doc_id: str, doc_title: str) -> ParsedDoc:
    """Parse a PDF into a ParsedDoc, docling first with pymupdf fallback."""
    try:
        paragraphs = _parse_with_docling(pdf_path)
    except Exception as exc:  # noqa: BLE001 — fall back, but say why.
        print(f"[parse] docling failed on {doc_id} ({exc}); falling back to pymupdf")
        paragraphs = split_into_paragraphs(_extract_pages_pymupdf(pdf_path))
    if not paragraphs:
        raise ValueError(f"no paragraphs extracted from {doc_id} ({pdf_path})")
    return ParsedDoc(doc_id=doc_id, doc_title=doc_title, paragraphs=paragraphs)


# --------------------------------------------------------------------------- #
# docling path
# --------------------------------------------------------------------------- #
def _parse_with_docling(pdf_path: Path) -> list[Paragraph]:
    """Convert with docling and build paragraphs from its item structure."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    # Born-digital regulatory PDFs: OCR is unnecessary and slow; keep tables.
    pipeline = PdfPipelineOptions(do_ocr=False, do_table_structure=True)
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline)}
    )
    doc = converter.convert(str(pdf_path)).document
    return build_paragraphs_from_items(
        doc.iterate_items(), table_md=lambda item: _table_markdown(item, doc)
    )


def _table_markdown(item: object, doc: object) -> str:
    """Render a docling table item to markdown, or '' if it cannot be rendered."""
    try:
        return item.export_to_markdown(doc).strip()  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 — a bad table shouldn't abort the parse.
        return ""


def build_paragraphs_from_items(
    items: Iterable[tuple[object, int]],
    *,
    table_md: Callable[[object], str],
) -> list[Paragraph]:
    """Build numbered paragraphs from docling (item, level) pairs.

    Rules:
    * section headers update the current section path;
    * list items with a numeric marker ("26.") start a new paragraph, keyed by
      that number (the citation anchor);
    * sub-points ("a.", "ii.") and stray text fold into the current paragraph;
    * tables render to markdown and attach to the current paragraph;
    * pictures, footnotes, captions and running headers are skipped.
    """
    paragraphs: list[Paragraph] = []
    section_path: list[str] = []
    current: Paragraph | None = None

    for item, _level in items:
        label = _label_of(item)
        text = (getattr(item, "text", "") or "").strip()
        page = _docling_page_no(item)

        if label in _SKIP_LABELS:
            continue

        if label == "table":
            md = table_md(item)
            if current is not None and md:
                current = _append_text(current, md, sep="\n\n")
            continue

        if label == "section_header":
            inline = _INLINE_NUM_RE.match(text)
            if inline and len(text) > _HEADING_MAX_LEN:
                paragraphs, current = _flush(paragraphs, current)
                current = Paragraph(
                    para_id=inline.group(1),
                    text=text[inline.end() :].strip(),
                    section_path=list(section_path),
                    page=page,
                )
            elif text:
                section_path = _apply_heading(section_path, text, getattr(item, "level", None))
            continue

        if label == "list_item":
            marker = (getattr(item, "marker", "") or "").strip()
            numeric = _NUMERIC_MARKER_RE.match(marker)
            if numeric:
                paragraphs, current = _flush(paragraphs, current)
                current = Paragraph(
                    para_id=numeric.group(1),
                    text=text,
                    section_path=list(section_path),
                    page=page,
                )
            elif current is not None:
                current = _append_text(current, f"{marker} {text}".strip())
            elif text:
                current = Paragraph(
                    para_id=marker.rstrip(".") or text[:8],
                    text=text,
                    section_path=list(section_path),
                    page=page,
                )
            continue

        # Plain text: a continuation of the current paragraph (skip pre-body text).
        if text and current is not None:
            current = _append_text(current, text)

    paragraphs, current = _flush(paragraphs, current)
    return paragraphs


def _label_of(item: object) -> str | None:
    label = getattr(item, "label", None)
    return getattr(label, "value", label)  # DocItemLabel enum -> its str value


def _append_text(paragraph: Paragraph, extra: str, *, sep: str = "\n") -> Paragraph:
    """Return a copy of `paragraph` with `extra` appended (immutable update)."""
    return paragraph.model_copy(update={"text": f"{paragraph.text}{sep}{extra}".strip()})


def _flush(
    paragraphs: list[Paragraph], current: Paragraph | None
) -> tuple[list[Paragraph], None]:
    """Append `current` (if any) to `paragraphs` and clear it."""
    if current is not None:
        paragraphs.append(current)
    return paragraphs, None


def _apply_heading(section_path: list[str], title: str, level: object) -> list[str]:
    """Return a new section path with `title` applied at its heading depth."""
    depth = level if isinstance(level, int) and level > 0 else len(section_path) + 1
    return [*section_path[: depth - 1], title]


def _docling_page_no(item: object) -> int:
    """Best-effort page number for a docling item (defaults to 1)."""
    prov = getattr(item, "prov", None)
    if prov:
        page = getattr(prov[0], "page_no", None)
        if isinstance(page, int):
            return page
    return 1


# --------------------------------------------------------------------------- #
# pymupdf fallback path
# --------------------------------------------------------------------------- #
def _extract_pages_pymupdf(pdf_path: Path) -> list[tuple[int, str]]:
    """Return [(page_number, page_text)] using pymupdf (fallback)."""
    import pymupdf  # bundled core dependency

    pages: list[tuple[int, str]] = []
    with pymupdf.open(pdf_path) as doc:
        for page_index in range(doc.page_count):
            text = doc[page_index].get_text("text")
            pages.append((page_index + 1, text))
    return pages


def split_into_paragraphs(pages: list[tuple[int, str]]) -> list[Paragraph]:
    """Segment page text into numbered paragraphs, tracking section headings.

    Pure function over (page_no, text) pairs so it can be unit-tested without a
    real PDF. Lines before the first numbered marker on a page are treated as
    continuations/headings; recognised headings update the current section path.
    """
    paragraphs: list[Paragraph] = []
    section_path: list[str] = []
    current: Paragraph | None = None

    for page_no, text in pages:
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            # Headings ("5 Margin of conservatism") are title-cased and carry no
            # trailing period, so they never collide with numbered paragraphs
            # ("81. ..."), whose digits are immediately followed by a period.
            heading = _HEADING_RE.match(line)
            if heading:
                section_path = _update_section_path(section_path, heading)
                continue

            marker = _PARA_MARKER_RE.match(line)
            if marker:
                if current is not None:
                    paragraphs.append(current)
                para_id = marker.group(1).rstrip(".")
                body = line[marker.end() :].strip()
                current = Paragraph(
                    para_id=para_id,
                    text=body,
                    section_path=list(section_path),
                    page=page_no,
                )
            elif current is not None:
                current = _append_text(current, line, sep=" ")

    if current is not None:
        paragraphs.append(current)
    return paragraphs


def _update_section_path(section_path: list[str], heading: re.Match[str]) -> list[str]:
    """Return a new section path with `heading` applied at its numbering depth."""
    number, title = heading.group(1), heading.group(2).strip()
    depth = number.count(".") + 1
    label = f"{number} {title}"
    return [*section_path[: depth - 1], label]
