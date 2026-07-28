"""Parser tests (SPEC §5 step 2).

Two layers are tested without invoking docling's slow model pipeline:

* the docling item builder `build_paragraphs_from_items`, driven by lightweight
  fake items that mimic docling's (item, level) structure;
* the pymupdf fallback: `split_into_paragraphs` (pure) and an end-to-end run
  over a generated fixture PDF.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pymupdf

from ingestion.parse import (
    _extract_pages_pymupdf,
    build_paragraphs_from_items,
    split_into_paragraphs,
)


# --------------------------------------------------------------------------- #
# docling item builder (fake items — fast, no models)
# --------------------------------------------------------------------------- #
def _item(label, text="", marker="", level=None, page=1, md=""):
    return SimpleNamespace(
        label=label,
        text=text,
        marker=marker,
        level=level,
        prov=[SimpleNamespace(page_no=page)],
        md=md,
    )


def _items(*items):
    return [(it, 1) for it in items]


def _no_tables(_item):
    return ""


def test_numeric_list_items_become_numbered_paragraphs() -> None:
    items = _items(
        _item("section_header", "4. Governance", level=1, page=3),
        _item("list_item", "Institutions should develop a risk culture.", marker="26.", page=3),
        _item("list_item", "The culture should include tone from the top.", marker="27.", page=3),
    )
    paras = build_paragraphs_from_items(items, table_md=_no_tables)
    assert [p.para_id for p in paras] == ["26", "27"]
    assert paras[0].section_path == ["4. Governance"]
    assert paras[0].page == 3


def test_subpoints_fold_into_parent_paragraph() -> None:
    items = _items(
        _item("list_item", "Policies should specify:", marker="38.", page=5),
        _item("list_item", "approval of credit.", marker="a.", page=5),
        _item("list_item", "credit-granting criteria.", marker="b.", page=5),
        _item("list_item", "Next paragraph.", marker="39.", page=6),
    )
    paras = build_paragraphs_from_items(items, table_md=_no_tables)
    assert [p.para_id for p in paras] == ["38", "39"]
    assert "a. approval of credit." in paras[0].text
    assert "b. credit-granting criteria." in paras[0].text


def test_footnotes_and_pictures_are_skipped() -> None:
    items = _items(
        _item("list_item", "Body paragraph.", marker="1.", page=1),
        _item("footnote", "11 Directive 2013/36/EU of the European Parliament.", page=1),
        _item("picture", "", page=1),
    )
    paras = build_paragraphs_from_items(items, table_md=_no_tables)
    assert len(paras) == 1
    assert "Directive" not in paras[0].text


def test_table_attaches_to_current_paragraph() -> None:
    items = _items(
        _item("list_item", "See the table below.", marker="5.", page=2),
        _item("table", "", page=2, md="| a | b |\n|---|---|"),
    )
    paras = build_paragraphs_from_items(items, table_md=lambda it: it.md)
    assert "| a | b |" in paras[0].text


def test_long_numbered_section_header_is_treated_as_paragraph() -> None:
    long_text = (
        "15. In addition, for the purposes of these guidelines, the following "
        "definitions apply."
    )
    items = _items(_item("section_header", long_text, level=1, page=4))
    paras = build_paragraphs_from_items(items, table_md=_no_tables)
    assert len(paras) == 1
    assert paras[0].para_id == "15"


def test_short_numbered_header_updates_section_path() -> None:
    items = _items(
        _item("section_header", "2. Subject matter", level=1, page=1),
        _item("list_item", "Body.", marker="1.", page=1),
    )
    paras = build_paragraphs_from_items(items, table_md=_no_tables)
    assert paras[0].section_path == ["2. Subject matter"]


def test_section_path_respects_heading_level() -> None:
    items = _items(
        _item("section_header", "Chapter 4", level=1, page=1),
        _item("section_header", "4.1 Policies", level=2, page=1),
        _item("list_item", "Body.", marker="1.", page=1),
        _item("section_header", "Chapter 5", level=1, page=2),
        _item("list_item", "Other.", marker="2.", page=2),
    )
    paras = build_paragraphs_from_items(items, table_md=_no_tables)
    assert paras[0].section_path == ["Chapter 4", "4.1 Policies"]
    assert paras[1].section_path == ["Chapter 5"]


# --------------------------------------------------------------------------- #
# pymupdf fallback path
# --------------------------------------------------------------------------- #
def test_split_captures_numbered_paragraphs_and_sections() -> None:
    pages = [
        (
            45,
            "5 Margin of conservatism\n"
            "81. The institution shall apply an appropriate margin of conservatism.\n"
            "82. Where data is scarce the margin shall be increased accordingly.\n",
        )
    ]
    paras = split_into_paragraphs(pages)
    assert [p.para_id for p in paras] == ["81", "82"]
    assert paras[0].section_path == ["5 Margin of conservatism"]
    assert paras[0].page == 45


def test_split_appends_continuation_lines() -> None:
    pages = [
        (
            1,
            "12. First line of the paragraph\n"
            "continues on the next visual line without a marker.\n",
        )
    ]
    paras = split_into_paragraphs(pages)
    assert len(paras) == 1
    assert "continues on the next visual line" in paras[0].text


def test_article_style_marker_is_recognised() -> None:
    pages = [(1, "Article 178(1)(b) A default is considered to have occurred.\n")]
    paras = split_into_paragraphs(pages)
    assert paras[0].para_id == "Article 178(1)(b)"


def _write_fixture_pdf(path: Path) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    body = (
        "5 Margin of conservatism\n"
        "81. The institution shall apply an appropriate margin of conservatism.\n"
        "82. Where data is scarce the margin shall be increased accordingly.\n"
    )
    page.insert_textbox(pymupdf.Rect(50, 50, 550, 750), body, fontsize=11)
    doc.save(str(path))
    doc.close()


def test_pymupdf_fallback_pipeline_on_fixture_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    _write_fixture_pdf(pdf_path)
    paras = split_into_paragraphs(_extract_pages_pymupdf(pdf_path))
    ids = [p.para_id for p in paras]
    assert "81" in ids and "82" in ids


def test_docling_converter_is_built_once(monkeypatch) -> None:
    """The DocumentConverter (multi-minute model init) is created once and reused
    across documents, not rebuilt per PDF."""
    import sys
    import types

    import ingestion.parse as parse_mod

    built = {"count": 0}

    class _FakeConverter:
        def __init__(self, **kwargs: object) -> None:
            built["count"] += 1

    def _mod(name: str, **attrs: object) -> types.ModuleType:
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        return module

    # Minimal fake docling package tree so the lazy imports resolve without the
    # real (slow) dependency.
    monkeypatch.setitem(sys.modules, "docling", _mod("docling"))
    monkeypatch.setitem(sys.modules, "docling.datamodel", _mod("docling.datamodel"))
    monkeypatch.setitem(
        sys.modules,
        "docling.datamodel.base_models",
        _mod("docling.datamodel.base_models", InputFormat=SimpleNamespace(PDF="pdf")),
    )
    monkeypatch.setitem(
        sys.modules,
        "docling.datamodel.pipeline_options",
        _mod("docling.datamodel.pipeline_options", PdfPipelineOptions=lambda **kw: object()),
    )
    monkeypatch.setitem(
        sys.modules,
        "docling.document_converter",
        _mod(
            "docling.document_converter",
            DocumentConverter=_FakeConverter,
            PdfFormatOption=lambda **kw: object(),
        ),
    )

    parse_mod._docling_converter.cache_clear()
    try:
        first = parse_mod._docling_converter()
        second = parse_mod._docling_converter()
        assert first is second  # same instance reused
        assert built["count"] == 1  # constructed only once across calls
    finally:
        parse_mod._docling_converter.cache_clear()  # don't leak the fake to other tests


def test_missing_docling_fails_loudly(tmp_path: Path, monkeypatch) -> None:
    """A missing docling install is an environment error, not a silent fallback."""
    import ingestion.parse as parse_mod

    def _raise_import_error(_path):
        raise ImportError("No module named 'docling'")

    monkeypatch.setattr(parse_mod, "_parse_with_docling", _raise_import_error)
    pdf_path = tmp_path / "sample.pdf"
    _write_fixture_pdf(pdf_path)
    try:
        parse_mod.parse_pdf(pdf_path, doc_id="x", doc_title="X")
    except RuntimeError as exc:
        assert "docling" in str(exc)
        return
    raise AssertionError("expected RuntimeError when docling is missing")
