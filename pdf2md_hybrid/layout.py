#!/usr/bin/env python3
"""Page geometry: reading order, font-size headings, and monospace detection.

This module is deliberately free of PDF-library calls. It takes plain character
dicts -- ``{text, x0, x1, top, bottom, size, fontname}`` -- and returns ordered
blocks. That separation exists because reading order is the pipeline's largest
known risk and the only way to test it honestly is to feed it synthetic pages
whose correct answer is known in advance.

The ordering strategy, and why:

1.  Characters group into rows by shared baseline.
2.  Each row splits into *segments* at its internal gaps. On a multi-column
    page a row is a horizontal slice holding one line of each column, and those
    are separate lines of reading.
3.  Segment left edges are clustered across the whole page into *column
    spines*. This is the key measurement, and it is deliberately not a search
    for vertical whitespace: on a dense grid some row's cell is always wide
    enough to bridge the gap between columns, so the whitespace closes up and
    the page looks single-column when it plainly is not.

    What separates a real column boundary from an ordinary word space is not
    width but *recurrence*. A word space falls at an arbitrary x once. A column
    boundary falls at the same x on row after row. So a spine has to be
    supported by several rows independently before it counts, which is what
    makes narrow boundaries safe to trust -- a 9pt gap is ambiguous on its own
    and unmistakable once twenty rows agree on it.
4.  Full-width elements (titles, banners) are single segments reaching past the
    next spine. They split the page into bands.
5.  Within a band, columns are read left-to-right and rows top-to-bottom.

A page with fewer than two supported spines is single-column and its rows pass
through in plain top-to-bottom order, untouched. That path is exact and it is
the common case, so two guards protect it: spines must be genuinely far apart,
and several rows must occupy two different spines *at once*. An indented block
of prose puts one segment on each row and so can never qualify -- which matters,
because wrongly splitting a single-column page corrupts it silently.

This module reconstructs reading order, not table structure. It keeps each
column's cells together and in sequence; it does not try to recover a grid of
rows and cells. Where the text wraps inside a cell, which line belongs to which
row genuinely is ambiguous, and guessing produces confident nonsense.
"""

from __future__ import annotations

import statistics
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple

Char = Dict[str, Any]
Row = Dict[str, Any]

# A gap this many median character widths wide ends a segment. Generous, so
# ordinary word spacing does not fragment a line.
SEGMENT_GAP_RATIO = 0.6

# Left edges within this many character widths of each other share a spine.
# Cells are often centred, so a column's left edges scatter rather than align
# exactly, and the tolerance has to absorb that scatter.
SPINE_TOLERANCE_RATIO = 3.0

# A spine must recur on at least this many rows to be believed. This is the
# recurrence test that makes narrow boundaries trustworthy.
SPINE_MIN_ROWS = 3

# Two spines must be at least this far apart, relative to the page, to be
# separate columns rather than an indent.
SPINE_MIN_SEPARATION = 0.06

# ...and this many rows must put segments on two different spines at once.
# Without it, a consistently indented single-column page reads as two columns
# and is silently reordered. All the derived numbers above are ratios, never
# points: absolute sizes vary per document and must not be hard-coded.
SPINE_MIN_COEXISTING_ROWS = 3

# A row counts as a heading when its dominant size exceeds body size by this
# much. Below it, the difference is leading or a superscript, not a heading.
HEADING_SIZE_RATIO = 1.08

# ...but size alone is not enough. A face carrying this much of the document's
# body volume is a second body style, not a heading, however large it is set.
# Headings are rare by nature; a "heading" holding a third of the prose is a
# misclassification, and on slide decks it is a common one.
HEADING_MAX_VOLUME = 0.20

# ...but that ratio is only meaningful once there is enough body text to measure.
# Below this many characters, a short document's single paragraph says nothing
# about how rare its headings are, so size alone decides.
HEADING_VOLUME_MIN_SAMPLE = 500

# And headings are short. Past this length a row is a sentence set large, not a
# title, whatever its size says.
HEADING_MAX_CHARS = 100

# Text below this fraction of body size is invisible to a reader at any sane
# zoom, so it is not content: it is the metadata some authoring tools leave
# behind (slide ids, export tags). Dropping it matters beyond tidiness -- left
# in, it inflates the character count that triage depends on.
MIN_VISIBLE_SIZE_RATIO = 0.35

# An absolute floor as well as a relative one, because the relative test alone
# is circular: where the junk outnumbers the prose it becomes "body size" and
# switches its own filter off. Nothing below this is legible at any zoom on any
# page, in any corpus -- that is a fact about rendering, not about a document.
MIN_VISIBLE_PT = 4.0

# A row this wide relative to the text measure is a banner, not a column line.
FULL_WIDTH_RATIO = 0.9

# Advance widths this consistent mean the face is fixed-pitch.
MONOSPACE_CV = 0.06
MONOSPACE_MIN_SAMPLES = 24

# Last-resort fallback only; see `font_is_monospace`.
MONO_NAME_HINTS = ("mono", "consol", "courier", "menlo", "code", "typewriter")


# --- primitives -----------------------------------------------------------

def normalize_text(text: str) -> str:
    """Fold compatibility forms so extracted text is searchable.

    PDFs routinely encode "fi" and "fl" as single ligature glyphs. They look
    right and read wrong: a search for "final" does not match "ﬁnal". NFKC
    is the standard fold for exactly this.
    """
    return unicodedata.normalize("NFKC", text)


def visible_chars(chars: Sequence[Char], body: float) -> List[Char]:
    """Drop characters too small to be content. See MIN_VISIBLE_SIZE_RATIO."""
    floor = MIN_VISIBLE_PT
    if body:
        floor = max(floor, body * MIN_VISIBLE_SIZE_RATIO)
    return [c for c in chars if float(c.get("size") or 0) >= floor]


def legible_body_size(size_counts: Dict[float, int]) -> float:
    """Body size, measured only over text large enough to be read.

    Measuring over everything is circular where sub-point metadata outnumbers
    the prose: the junk wins the vote, becomes "body", and disables the very
    filter meant to remove it.
    """
    return body_size({s: n for s, n in size_counts.items() if s >= MIN_VISIBLE_PT})


def size_histogram(chars: Sequence[Char]) -> Dict[float, int]:
    """Count characters by rounded font size. Rounding keeps clusters stable
    against the sub-point jitter that PDF text matrices introduce."""
    sizes: Dict[float, int] = {}
    for c in chars:
        if not (c.get("text") or "").strip():
            continue
        key = round(float(c.get("size") or 0), 1)
        sizes[key] = sizes.get(key, 0) + 1
    return sizes


def median_char_width(chars: Sequence[Char]) -> float:
    widths = [c["x1"] - c["x0"] for c in chars if (c.get("text") or "").strip()]
    return statistics.median(widths) if widths else 1.0


def median_char_height(chars: Sequence[Char]) -> float:
    heights = [c["bottom"] - c["top"] for c in chars if (c.get("text") or "").strip()]
    return statistics.median(heights) if heights else 1.0


def group_rows(chars: Sequence[Char], tolerance: Optional[float] = None) -> List[Row]:
    """Cluster characters into rows sharing a baseline, left-to-right within each."""
    printable = [c for c in chars if (c.get("text") or "") != ""]
    if not printable:
        return []
    if tolerance is None:
        tolerance = max(median_char_height(printable) * 0.5, 0.5)

    rows: List[Row] = []
    for c in sorted(printable, key=lambda c: (round(c["top"], 1), c["x0"])):
        placed = False
        for row in reversed(rows):
            if abs(row["top"] - c["top"]) <= tolerance:
                row["chars"].append(c)
                row["top"] = min(row["top"], c["top"])
                row["bottom"] = max(row["bottom"], c["bottom"])
                placed = True
                break
        if not placed:
            rows.append({"chars": [c], "top": c["top"], "bottom": c["bottom"]})

    for row in rows:
        row["chars"].sort(key=lambda c: c["x0"])
        row["x0"] = min(c["x0"] for c in row["chars"])
        row["x1"] = max(c["x1"] for c in row["chars"])
    rows.sort(key=lambda r: (r["top"], r["x0"]))
    return rows


def split_segments(row: Row, unit: float) -> List[Row]:
    """Cut a row at its internal gaps. On a grid, one segment per cell."""
    chars = row["chars"]
    if not chars:
        return []
    groups: List[List[Char]] = [[chars[0]]]
    for c in chars[1:]:
        if c["x0"] - groups[-1][-1]["x1"] > unit * SEGMENT_GAP_RATIO:
            groups.append([c])
        else:
            groups[-1].append(c)
    return [{
        "chars": g,
        "top": min(c["top"] for c in g),
        "bottom": max(c["bottom"] for c in g),
        "x0": min(c["x0"] for c in g),
        "x1": max(c["x1"] for c in g),
        "full_width": False,
    } for g in groups]


def find_column_spines(rows: Sequence[Row], unit: float, page_width: float) -> List[float]:
    """Cluster segment left edges into column spines. See the module docstring.

    Returns the spine x positions, or an empty list when the page is
    single-column -- which is the answer that has to be right by default.
    """
    observations: List[Tuple[float, int]] = []
    for i, row in enumerate(rows):
        for seg in split_segments(row, unit):
            observations.append((seg["x0"], i))
    if not observations:
        return []
    observations.sort()

    tolerance = unit * SPINE_TOLERANCE_RATIO
    clusters: List[List[Tuple[float, int]]] = [[observations[0]]]
    for x, i in observations[1:]:
        if x - clusters[-1][-1][0] > tolerance:
            clusters.append([(x, i)])
        else:
            clusters[-1].append((x, i))

    # A cluster becomes a spine only if several different rows put a segment
    # there. One row agreeing with itself proves nothing.
    spines = [min(x for x, _ in cluster) for cluster in clusters
              if len({i for _, i in cluster}) >= SPINE_MIN_ROWS]
    if len(spines) < 2:
        return []

    # Adjacent spines must be far enough apart to be columns, not an indent.
    separation = page_width * SPINE_MIN_SEPARATION
    merged = [spines[0]]
    for x in spines[1:]:
        if x - merged[-1] >= separation:
            merged.append(x)
    if len(merged) < 2:
        return []

    # Finally: do the columns actually coexist? Real columns run side by side,
    # so many rows carry segments on two spines at the same time. An indented
    # single-column page never does, and this is what protects it.
    coexisting = 0
    for row in rows:
        occupied = {column_of(seg["x0"], merged) for seg in split_segments(row, unit)}
        if len(occupied) > 1:
            coexisting += 1
    if coexisting < SPINE_MIN_COEXISTING_ROWS:
        return []
    return merged


def column_of(x: float, spines: Sequence[float]) -> int:
    """Index of the rightmost spine at or left of x."""
    col = 0
    for i, spine in enumerate(spines):
        if x >= spine - 1e-6:
            col = i
        else:
            break
    return col


def _is_full_width(segments: Sequence[Row], spines: Sequence[float]) -> bool:
    """A lone segment reaching past the next spine is a title or a banner."""
    if len(segments) != 1:
        return False
    seg = segments[0]
    col = column_of(seg["x0"], spines)
    return col + 1 < len(spines) and seg["x1"] > spines[col + 1]


def order_rows(chars: Sequence[Char], page_width: float) -> List[Row]:
    """Return rows in reading order. See the module docstring for the strategy."""
    rows = group_rows(chars)
    if not rows:
        return []

    unit = median_char_width(chars) or 1.0
    spines = find_column_spines(rows, unit, page_width)
    if not spines:
        return rows  # single column: exact, untouched, and the common case

    ordered: List[Row] = []
    band: List[Row] = []

    def flush() -> None:
        if not band:
            return
        band.sort(key=lambda s: (s["col"], s["top"], s["x0"]))
        ordered.extend(band)
        band.clear()

    for row in rows:
        segments = split_segments(row, unit)
        if _is_full_width(segments, spines):
            flush()
            segments[0]["full_width"] = True
            segments[0]["col"] = -1
            ordered.append(segments[0])
            continue
        for seg in segments:
            seg["col"] = column_of(seg["x0"], spines)
            band.append(seg)
    flush()
    return ordered


# --- font classification --------------------------------------------------

def font_is_monospace(widths_by_size: Sequence[Tuple[float, float]],
                      fontname: str = "") -> bool:
    """Decide monospace-ness from advance-width uniformity, per BUILD's ordering.

    The PDF font descriptor's fixed-pitch flag would be the first choice, but
    pdfplumber does not surface it, so the robust signal -- advance widths that
    do not vary with the glyph -- is the primary mechanism here. Name matching
    is a fallback used only when there are too few samples to measure.
    """
    ratios = [w / s for w, s in widths_by_size if s > 0 and w > 0]
    if len(ratios) >= MONOSPACE_MIN_SAMPLES:
        mean = statistics.fmean(ratios)
        if mean <= 0:
            return False
        return (statistics.pstdev(ratios) / mean) < MONOSPACE_CV
    lowered = (fontname or "").lower()
    return any(hint in lowered for hint in MONO_NAME_HINTS)


def monospace_fonts(chars: Sequence[Char]) -> set:
    """Names of the faces in `chars` that behave as fixed-pitch."""
    samples: Dict[str, List[Tuple[float, float]]] = {}
    for c in chars:
        text = c.get("text") or ""
        if not text.strip():
            continue  # spaces are the same width in every face; they prove nothing
        samples.setdefault(c.get("fontname") or "", []).append(
            (c["x1"] - c["x0"], float(c.get("size") or 0)))
    return {name for name, obs in samples.items() if font_is_monospace(obs, name)}


def body_size(size_counts: Dict[float, int]) -> float:
    """The most-used font size in the document: body text by definition."""
    if not size_counts:
        return 0.0
    return max(size_counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]


def heading_levels(size_counts: Dict[float, int]) -> Dict[float, int]:
    """Map font size -> Markdown heading level, largest size to ``h1``.

    Clustered across the whole document, never per page: a page whose only text
    is a title would otherwise report that title as body text. Absolute point
    sizes are never hard-coded -- only their ranking within this document.

    A size must clear two bars, not one. It has to be meaningfully larger than
    body text, *and* it has to be rare. A size set 12% larger that carries a
    third of the document's prose is a second body style, and promoting it turns
    whole paragraphs into headings.
    """
    body = body_size(size_counts)
    if not body:
        return {}
    volume = size_counts.get(body, 0) or 1
    cap = volume * HEADING_MAX_VOLUME if volume >= HEADING_VOLUME_MIN_SAMPLE else float("inf")
    bigger = sorted(
        (s for s, n in size_counts.items()
         if s >= body * HEADING_SIZE_RATIO and n <= cap),
        reverse=True)
    return {size: min(i + 1, 6) for i, size in enumerate(bigger)}


def row_size(row: Row) -> float:
    """The row's dominant character size, rounded to keep clusters stable."""
    sizes = size_histogram(row["chars"])
    if not sizes:
        return 0.0
    return max(sizes.items(), key=lambda kv: kv[1])[0]


def row_is_code(row: Row, mono: set) -> bool:
    """True when most of the row's ink is set in a fixed-pitch face."""
    ink = [c for c in row["chars"] if (c.get("text") or "").strip()]
    if not ink:
        return False
    hits = sum(1 for c in ink if (c.get("fontname") or "") in mono)
    return hits * 2 > len(ink)


def row_text(row: Row, preserve_indent: bool = False) -> str:
    """Reconstruct a row's text, inserting spaces where the geometry implies them."""
    chars = row["chars"]
    if not chars:
        return ""
    unit = median_char_width(chars) or 1.0
    out: List[str] = []
    if preserve_indent:
        # Indentation is meaningful in code, so rebuild it from the row's offset
        # against the block's left edge rather than trusting emitted spaces.
        lead = max(0, int(round((row["x0"] - row.get("block_x0", row["x0"])) / unit)))
        out.append(" " * lead)
    prev = None
    for c in chars:
        text = c.get("text") or ""
        if prev is not None:
            gap = c["x0"] - prev["x1"]
            if gap > unit * 0.28 and not text.isspace() and not (prev.get("text") or "").isspace():
                out.append(" " * max(1, int(round(gap / unit)) if preserve_indent else 1))
        out.append(text)
        prev = c
    return normalize_text("".join(out)).rstrip()
