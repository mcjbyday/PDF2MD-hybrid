#!/usr/bin/env python3
"""Stage 2 + 3: extract every page deterministically, then triage.

No model runs here, and none ever should. Extraction is exact -- the text in a
born-digital PDF is ground truth, and a model asked to re-read a picture of it
can only degrade it. Keeping this stage model-free is also what makes the
pipeline debuggable: when a page comes out wrong, the stage that produced it is
never in question.

Usage:
    python extract.py CORPUS_DIR --out out/ [--files a,b] [--force]

Writes one `out/<stem>.pages.json` per source PDF. This stage owns every field
in that file except `vision`, which only enrich.py writes.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional

try:
    import pdfplumber
except ImportError:  # pragma: no cover - environment guard
    sys.exit("pdfplumber is required: pip install -e .")

import common
import layout


def _scan_fonts(pdf) -> Dict[str, Any]:
    """First pass: font sizes and monospace faces, clustered across the document.

    Both signals are document-level on purpose. Absolute point sizes vary per
    document and must never be hard-coded, and a face's advance widths need more
    samples than one page usually provides.
    """
    all_chars: List[Dict[str, Any]] = []
    for page in pdf.pages:
        try:
            chars = page.chars
        except Exception:
            chars = []
        all_chars.extend(c for c in chars if (c.get("text") or "").strip())
        page.flush_cache()
    raw = layout.size_histogram(all_chars)

    # Establish body size, then throw away text too small to be readable and
    # measure again. Authoring tools leave sub-point metadata in the text layer
    # -- slide ids and export tags -- and it is often the most numerous "size"
    # on a page. Counted as content it corrupts both the heading clusters and
    # the character totals triage runs on.
    body = layout.legible_body_size(raw)
    visible = layout.visible_chars(all_chars, body)
    size_counts = layout.size_histogram(visible)

    return {
        "body_size": layout.body_size(size_counts),
        "size_counts": size_counts,
        "heading_levels": layout.heading_levels(size_counts),
        "monospace": layout.monospace_fonts(visible),
    }


def _column_groups(rows: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Split ordered rows into runs that belong to the same column.

    On a single-column page no row carries a column, every row lands in one
    group, and the result is identical to not grouping at all.
    """
    groups: List[List[Dict[str, Any]]] = []
    previous = object()
    for row in rows:
        key = row.get("col")
        if key != previous or not groups:
            groups.append([])
            previous = key
        groups[-1].append(row)
    return groups


def _lead_term(group: List[Dict[str, Any]], fonts: Dict[str, Any]) -> int:
    """How many leading rows of a column are its label rather than its content.

    In a grid, a cell is typically a term set slightly larger over the lines
    that explain it. Keeping the two together -- and marking which is which --
    is the difference between a retrievable definition and a run of words.

    Measured against the column's own modal size, never the document's: these
    labels are usually too small to be document headings and too common to
    survive the heading-volume test, so they have to be found locally.
    """
    if group[0].get("col") is None or group[0].get("col", -1) < 0 or len(group) < 2:
        return 0
    sizes: Dict[float, int] = {}
    for row in group:
        sizes[layout.row_size(row)] = sizes.get(layout.row_size(row), 0) + 1
    modal = layout.body_size(sizes)
    if not modal:
        return 0
    count = 0
    for row in group:
        if fonts["heading_levels"].get(layout.row_size(row)):
            break  # a real document heading outranks a local cell label
        if layout.row_size(row) >= modal * layout.HEADING_SIZE_RATIO \
                and not layout.row_is_code(row, fonts["monospace"]):
            count += 1
        else:
            break
    # All of it larger than modal means there is no contrast to read, and a
    # label with nothing under it is not a label.
    return count if count < len(group) else 0


def _blocks_for_group(group: List[Dict[str, Any]], fonts: Dict[str, Any],
                      lead: int = 0) -> List[Dict[str, Any]]:
    """Turn one column's ordered rows into heading / term / para / code blocks.

    Adjacent rows of the same kind merge, so a paragraph broken across five
    lines becomes one block and a code listing keeps its shape.
    """
    blocks: List[Dict[str, Any]] = []
    pending_code: List[Dict[str, Any]] = []
    pending_para: List[str] = []
    term: List[str] = []

    def flush_code() -> None:
        if not pending_code:
            return
        block_x0 = min(r["x0"] for r in pending_code)
        lines = []
        for row in pending_code:
            row["block_x0"] = block_x0
            lines.append(layout.row_text(row, preserve_indent=True))
        while lines and not lines[-1].strip():
            lines.pop()
        text = "\n".join(lines)
        if text.strip():
            blocks.append({"kind": "code", "text": text, "lang": None})
        pending_code.clear()

    def flush_para() -> None:
        if not pending_para:
            return
        text = " ".join(s for s in pending_para if s.strip()).strip()
        if text:
            blocks.append({"kind": "para", "text": text})
        pending_para.clear()

    for i, row in enumerate(group):
        text = layout.row_text(row)
        if not text.strip():
            continue
        if i < lead:
            term.append(text)
            continue
        if term:
            blocks.append({"kind": "term", "text": " ".join(term).strip()})
            term.clear()
        if layout.row_is_code(row, fonts["monospace"]):
            flush_para()
            pending_code.append(row)
            continue
        flush_code()
        level = fonts["heading_levels"].get(layout.row_size(row))
        if level and len(text) > layout.HEADING_MAX_CHARS:
            level = None  # a sentence set large is still a sentence
        if level:
            flush_para()
            blocks.append({"kind": "heading", "level": level, "text": text})
        else:
            pending_para.append(text)

    if term:
        blocks.append({"kind": "term", "text": " ".join(term).strip()})
    flush_code()
    flush_para()
    return blocks


def _blocks_for_rows(rows: List[Dict[str, Any]], fonts: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Turn ordered rows into blocks, one column at a time.

    Columns are kept apart here rather than merged into running prose. A grid
    that reads in the right order but arrives as one paragraph has recovered
    the reading order and thrown away the grouping, which for a reader or a
    retrieval index is most of what the structure was carrying.
    """
    blocks: List[Dict[str, Any]] = []
    for group in _column_groups(rows):
        blocks.extend(_blocks_for_group(group, fonts, _lead_term(group, fonts)))
    return blocks


def extract_page(page, fonts: Dict[str, Any], image_frac_threshold: float,
                 text_floor: int) -> Dict[str, Any]:
    chars = layout.visible_chars(
        [c for c in page.chars if (c.get("text") or "") != ""], fonts["body_size"])
    rows = layout.order_rows(chars, float(page.width))
    blocks = _blocks_for_rows(rows, fonts)
    n_chars = sum(len(b.get("text", "")) for b in blocks)
    frac = common.page_image_frac(page)
    return {
        "n": page.page_number,
        "blocks": blocks,
        "chars": n_chars,
        "image_frac": round(frac, 4),
        "needs_vision": common.needs_vision(n_chars, frac, image_frac_threshold, text_floor),
        "vision": None,
    }


def extract_file(path: str, image_frac_threshold: float, text_floor: int) -> Dict[str, Any]:
    with pdfplumber.open(path) as pdf:
        fonts = _scan_fonts(pdf)
        pages: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        for page in pdf.pages:
            try:
                pages.append(extract_page(page, fonts, image_frac_threshold, text_floor))
            except Exception as exc:
                # One unreadable page must not cost the other 1,987.
                errors.append({"page": page.page_number, "error": f"{type(exc).__name__}: {exc}"})
                pages.append({"n": page.page_number, "blocks": [], "chars": 0,
                              "image_frac": 0.0, "needs_vision": False, "vision": None})
            finally:
                page.flush_cache()
        return {
            "source": os.path.basename(path),
            "source_sha256": common.sha256_file(path),
            "extracted_at": common.utc_now(),
            "page_count": len(pages),
            "thresholds": {"image_frac": image_frac_threshold, "text_floor": text_floor},
            "errors": errors,
            "pages": pages,
        }


def _is_current(target: str, src: str) -> bool:
    """True when an existing artifact already describes this exact file."""
    try:
        return common.load_pages(target).get("source_sha256") == common.sha256_file(src)
    except Exception:
        return False


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="PDF files, or directories to search for them")
    ap.add_argument("--out", default="out", help="directory for .pages.json artifacts (default out/)")
    ap.add_argument("--files", help="comma-separated stems to process, instead of all")
    ap.add_argument("--image-frac", type=float, default=common.IMAGE_FRAC,
                    help=f"page-area fraction that counts as a dominant image (default {common.IMAGE_FRAC})")
    ap.add_argument("--text-floor", type=int, default=common.TEXT_FLOOR,
                    help=f"chars below which a page's text is 'thin' (default {common.TEXT_FLOOR})")
    ap.add_argument("--force", action="store_true", help="re-extract even if the artifact is current")
    args = ap.parse_args(argv)

    files = common.find_pdfs(args.paths, common.parse_files_flag(args.files))
    if not files:
        print("No PDFs found in the given paths.", file=sys.stderr)
        return 1

    os.makedirs(args.out, exist_ok=True)
    started = time.time()
    total_pages = total_vision = 0
    failed = 0

    for path in files:
        name = common.stem(path)
        target = common.pages_path(args.out, name)
        if not args.force and os.path.exists(target) and _is_current(target, path):
            doc = common.load_pages(target)
            print(f"  {name:<32} unchanged, skipped ({doc.get('page_count', 0)} pages)")
            total_pages += doc.get("page_count", 0)
            total_vision += sum(1 for p in doc.get("pages", []) if p.get("needs_vision"))
            continue
        t0 = time.time()
        try:
            doc = extract_file(path, args.image_frac, args.text_floor)
        except Exception as exc:
            # A corrupt file is recorded and skipped; the run continues.
            print(f"  {name:<32} !! {type(exc).__name__}: {exc}", file=sys.stderr)
            failed += 1
            continue
        common.save_pages(target, doc)
        vision = sum(1 for p in doc["pages"] if p["needs_vision"])
        total_pages += doc["page_count"]
        total_vision += vision
        chars = sum(p["chars"] for p in doc["pages"])
        print(f"  {name:<32} {doc['page_count']:>5} pages  {chars:>8,} chars  "
              f"{vision:>4} for vision  {time.time() - t0:>5.1f}s")

    elapsed = time.time() - started
    print(f"\n{len(files) - failed} files, {total_pages} pages in {elapsed:.1f}s "
          f"({total_pages / elapsed:.0f} pages/s)")
    if total_pages:
        print(f"{total_vision} pages ({total_vision / total_pages:.1%}) flagged for the vision model; "
              f"the rest are done.")
    if failed:
        print(f"{failed} file(s) could not be read.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
