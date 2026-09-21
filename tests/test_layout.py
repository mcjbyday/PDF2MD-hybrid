"""Reading order and font classification -- the pipeline's largest known risk.

These come first because a column-ordering regression is silent: a scrambled
page is still valid Markdown, still the right length, and still passes every
check that does not actually read it.
"""

from __future__ import annotations

import pdfplumber

import common
import layout


def blocks_for(pages_dir, stem, page=0):
    doc = common.load_pages(common.pages_path(pages_dir, stem))
    return doc["pages"][page]["blocks"]


# --- 1. column ordering ---------------------------------------------------

def test_single_column_passes_through_unchanged(fixtures_dir):
    """The regression that matters most. One column must never be reordered."""
    with pdfplumber.open(f"{fixtures_dir}/single-column.pdf") as pdf:
        page = pdf.pages[0]
        chars = [c for c in page.chars if (c.get("text") or "") != ""]
        rows = layout.group_rows(chars)
        ordered = layout.order_rows(chars, float(page.width))
    assert [layout.row_text(r) for r in ordered] == [layout.row_text(r) for r in rows]
    assert layout.find_column_spines(rows, layout.median_char_width(chars),
                                     float(page.width)) == [], "no columns to find"


def test_two_columns_do_not_interleave(extracted):
    """Prose beside code must produce two sequential blocks, not one merged one."""
    blocks = blocks_for(extracted, "two-column")
    prose = next(b for b in blocks if b["kind"] == "para")
    code = next(b for b in blocks if b["kind"] == "code")

    # Every prose line stays in the prose block and out of the code block.
    for phrase in ("left column carries prose", "belongs to this side",
                   "of the page and to no other"):
        assert phrase in prose["text"]
        assert phrase not in code["text"]
    # ...and the reverse. Interleaving fails both directions at once.
    for phrase in ("def configure", "return True", "sorted(options)"):
        assert phrase in code["text"]
        assert phrase not in prose["text"]


def test_column_content_stays_in_source_order(extracted):
    """Within a column, lines must still read top to bottom."""
    prose = next(b for b in blocks_for(extracted, "two-column") if b["kind"] == "para")
    first = prose["text"].index("left column carries prose")
    last = prose["text"].index("of the page and to no other")
    assert first < last


def test_full_width_title_precedes_both_columns(extracted):
    blocks = blocks_for(extracted, "two-column")
    assert blocks[0]["kind"] == "heading"
    assert blocks[0]["text"] == "Two Column Layout"


# --- 2. code detection ----------------------------------------------------

def test_monospace_font_becomes_a_code_block(extracted):
    code = next(b for b in blocks_for(extracted, "two-column") if b["kind"] == "code")
    assert code["lang"] is None, "guessing the language is out of scope"
    assert code["text"].startswith("def configure(options):")


def test_code_indentation_round_trips(extracted):
    code = next(b for b in blocks_for(extracted, "two-column") if b["kind"] == "code")
    lines = code["text"].split("\n")
    indents = [len(line) - len(line.lstrip(" ")) for line in lines]
    assert indents[0] == 0
    assert indents[1] > 0, "nested code lost its indentation"
    assert indents[2] > indents[1], "relative nesting depth was not preserved"


def test_monospace_detection_is_not_name_matching():
    """Advance-width uniformity is the primary signal; names are the fallback."""
    uniform = [(6.6, 11.0)] * 40
    varied = [(3.0 + (i % 7), 11.0) for i in range(40)]
    assert layout.font_is_monospace(uniform, "AAAAAA+SomeUnbrandedFace")
    assert not layout.font_is_monospace(varied, "AAAAAA+NotReallyMonoFace")


def test_name_fallback_applies_only_without_enough_samples():
    assert layout.font_is_monospace([(6.6, 11.0)] * 3, "Courier")
    assert not layout.font_is_monospace([(6.6, 11.0)] * 3, "Helvetica")


# --- 3. headings ----------------------------------------------------------

def test_heading_levels_descend_with_font_size(extracted):
    blocks = blocks_for(extracted, "headings")
    headings = [b for b in blocks if b["kind"] == "heading"]
    assert [h["level"] for h in headings] == [1, 2, 3]
    assert headings[0]["text"] == "Title Of The Document"


def test_body_text_is_not_promoted_to_a_heading(extracted):
    for block in blocks_for(extracted, "headings"):
        if block["kind"] == "para":
            assert "Body text" in block["text"]


def test_heading_sizes_are_clustered_not_hard_coded():
    """Levels must come from the document's own distribution of sizes."""
    counts = {40.0: 3, 28.0: 9, 9.0: 2000}
    assert layout.body_size(counts) == 9.0
    assert layout.heading_levels(counts) == {40.0: 1, 28.0: 2}
    # The same shape at different absolute sizes yields the same levels.
    assert layout.heading_levels({80.0: 3, 56.0: 9, 18.0: 2000}) == {80.0: 1, 56.0: 2}


# --- 4. defects found against a real corpus -------------------------------
# Each of these reproduces something that actually came out wrong on a real
# document, reduced to the smallest case that still shows it.

def test_subvisible_metadata_is_not_content():
    """Authoring tools leave sub-point text in the layer: slide ids, export tags.

    It is invisible to a reader, so it is not content. Counted as content it
    corrupts the character total that triage runs on -- which is worse than
    ugly, because it silently stops image-dominant pages being flagged.
    """
    body = 32.0
    chars = ([{"text": "x", "size": 1.0, "x0": 0, "x1": 1, "top": 0, "bottom": 1}] * 697
             + [{"text": "y", "size": 32.0, "x0": 0, "x1": 16, "top": 0, "bottom": 32}] * 40)
    kept = layout.visible_chars(chars, body)
    assert len(kept) == 40
    assert all(c["size"] == 32.0 for c in kept)


def test_body_size_survives_a_flood_of_invisible_text():
    """The junk can outnumber the prose on a page; it must not become 'body'."""
    raw = {1.0: 697, 32.0: 500, 86.0: 40}
    assert layout.body_size(raw) == 1.0, "the naive vote is won by the junk"
    assert layout.legible_body_size(raw) == 32.0, "measuring only legible text is not"

    chars = ([{"text": "x", "size": 1.0, "x0": 0, "x1": 1, "top": 0, "bottom": 1}] * 697
             + [{"text": "y", "size": 32.0, "x0": 0, "x1": 16, "top": 0, "bottom": 32}] * 500)
    assert len(layout.visible_chars(chars, layout.legible_body_size(raw))) == 500


def test_a_large_second_body_style_is_not_a_heading():
    """Size alone over-promotes. A face 12% larger that carries a third of the
    document's prose is a second body style, not a heading."""
    counts = {32.0: 5892, 36.0: 1966, 52.0: 553, 120.0: 83}
    levels = layout.heading_levels(counts)
    assert 36.0 not in levels, "a size carrying a third of the prose is body text"
    assert levels == {120.0: 1, 52.0: 2}


def test_volume_guard_stands_down_on_short_documents():
    """With one paragraph of body text, rarity means nothing; size decides."""
    counts = {11.0: 84, 14.0: 15, 18.0: 15, 24.0: 21}
    assert layout.heading_levels(counts) == {24.0: 1, 18.0: 2, 14.0: 3}


def test_ligatures_are_folded():
    """'Certification' encoded with a ligature glyph must still match a search
    for 'certification'."""
    assert layout.normalize_text("Certiﬁcation") == "Certification"
    assert layout.normalize_text("ﬂow ﬁnally") == "flow finally"


# --- 5. grids: grouping without grid reconstruction -----------------------

def test_narrow_gutters_are_found_by_recurrence(fixtures_dir):
    """A grid's columns are separated by less than a character of whitespace.

    Width alone cannot distinguish that from a word space. Recurrence can: the
    boundary falls at the same x on row after row.
    """
    with pdfplumber.open(f"{fixtures_dir}/grid.pdf") as pdf:
        page = pdf.pages[0]
        chars = [c for c in page.chars if (c.get("text") or "").strip()]
        spines = layout.find_column_spines(
            layout.group_rows(chars), layout.median_char_width(chars), float(page.width))
    assert len(spines) == 5, f"expected five columns, found {len(spines)}"


def test_grid_keeps_each_label_with_its_own_description(extracted):
    """The enhancement itself: terms and their descriptions stay together."""
    blocks = blocks_for(extracted, "grid")
    rendered = [(b["kind"], b["text"]) for b in blocks]

    # A cell label surfaces as a `term`, or as a `heading` where its size is
    # rare enough document-wide to be one. Which of the two is not the point;
    # staying attached to its own description is.
    labels = [t for k, t in rendered if k in ("term", "heading")]
    for name in ("ALPHA", "BETA", "GAMMA", "DELTA", "EPSILON"):
        assert name in labels

    for label, expected in (("ALPHA", "Cheap and simple"), ("BETA", "Faster but larger"),
                            ("GAMMA", "Needs a network"), ("EPSILON", "Deprecated now")):
        i = next(n for n, (k, t) in enumerate(rendered) if t == label)
        assert expected in rendered[i + 1][1], f"{label} lost its description"


def test_grid_columns_do_not_bleed_into_each_other(extracted):
    blocks = blocks_for(extracted, "grid")
    body = {b["text"] for b in blocks if b["kind"] == "para"}
    alpha = next(t for t in body if "Cheap and simple" in t)
    assert "Faster but larger" not in alpha
    assert "Deprecated now" not in alpha


def test_full_width_line_stays_out_of_the_columns(extracted):
    blocks = blocks_for(extracted, "grid")
    banner = next(b for b in blocks if "runs across every column" in b["text"])
    assert "ALPHA" not in banner["text"]
    assert blocks.index(banner) < next(
        i for i, b in enumerate(blocks) if b["text"] == "ALPHA")


def test_indented_prose_is_not_mistaken_for_columns(fixtures_dir, extracted):
    """The dangerous false positive: repeated left edges that are an indent.

    Wrongly splitting this page would reorder it silently, so the coexistence
    guard has to hold.
    """
    with pdfplumber.open(f"{fixtures_dir}/indented.pdf") as pdf:
        page = pdf.pages[0]
        chars = [c for c in page.chars if (c.get("text") or "").strip()]
        rows = layout.group_rows(chars)
        spines = layout.find_column_spines(
            rows, layout.median_char_width(chars), float(page.width))
        ordered = layout.order_rows(chars, float(page.width))
    assert spines == [], "an indent is not a column"
    assert [layout.row_text(r) for r in ordered] == [layout.row_text(r) for r in rows]

    text = " ".join(b["text"] for b in blocks_for(extracted, "indented"))
    assert text.index("indented from the margin") < text.index("returns to the margin")


def test_coexistence_guard_needs_columns_to_run_side_by_side():
    """Left edges that repeat but never overlap vertically are not columns."""
    def row(top, *spans):
        chars = [{"text": "x", "size": 10.0, "x0": x, "x1": x + 40,
                  "top": top, "bottom": top + 10} for x in spans]
        return {"chars": chars, "top": top, "bottom": top + 10,
                "x0": min(spans), "x1": max(spans) + 40}

    stacked = [row(i * 20, 0 if i % 2 else 300) for i in range(8)]
    assert layout.find_column_spines(stacked, 5.0, 600.0) == []

    side_by_side = [row(i * 20, 0, 300) for i in range(8)]
    assert len(layout.find_column_spines(side_by_side, 5.0, 600.0)) == 2
