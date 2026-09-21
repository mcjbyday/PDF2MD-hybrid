#!/usr/bin/env python3
"""Corpus triage: decide whether this pipeline fits your PDFs before you build anything.

Answers three questions, in order of how much they matter:

  1. Are these PDFs text-native or scanned?  A scanned corpus needs OCR or an
     all-vision tool; this pipeline would produce nothing from it.
  2. How much text is actually in there?  Bytes on disk are a poor proxy -- an
     image-heavy slide deck is mostly JPEG, not prose.
  3. What fraction of pages carry meaning only a model can read?  That fraction
     is the entire cost of the run, because every other page is close to free.

Usage:
    python probe.py CORPUS_DIR [options]
    python probe.py a.pdf b.pdf --json report.json

Exit codes: 0 = text-native, this pipeline fits. 2 = looks scanned, it does not.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys

try:
    import pdfplumber
except ImportError:
    sys.exit("pdfplumber is required: pip install -e .")

import common

# The triage thresholds live in common.py and are shared with every other stage,
# so that what the probe predicts is exactly what extraction will do. A page is
# a "vision candidate" when a single image covers a large share of it AND the
# text layer is thin -- the picture is carrying meaning the text does not.
IMAGE_FRAC = common.IMAGE_FRAC
TEXT_FLOOR = common.TEXT_FLOOR
SCANNED_CHARS_PER_PAGE = common.SCANNED_CHARS_PER_PAGE

# Measured on one corpus and one machine. Used only to turn page counts into a
# rough wall-clock estimate; your hardware will differ.
CHARS_PER_TOKEN = 3.64
SECONDS_PER_VISION_PAGE = 8.1


def page_stats(page) -> tuple[int, float, int]:
    """Return (chars, largest-image-area-as-fraction-of-page, image count)."""
    text = page.extract_text() or ""
    return len(text), common.page_image_frac(page), len(page.images)


def scan_file(path: str, image_frac: float, text_floor: int) -> dict:
    rec = {
        "file": os.path.basename(path),
        "size_kb": os.path.getsize(path) // 1024,
        "pages": 0,
        "chars": 0,
        "images": 0,
        "vision_pages": 0,
        "empty_pages": 0,
        "error": None,
    }
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                chars, frac, n_img = page_stats(page)
                rec["pages"] += 1
                rec["chars"] += chars
                rec["images"] += n_img
                if chars < 80:
                    rec["empty_pages"] += 1
                if common.needs_vision(chars, frac, image_frac, text_floor):
                    rec["vision_pages"] += 1
    except Exception as exc:  # a corrupt file should not abort the survey
        rec["error"] = f"{type(exc).__name__}: {exc}"
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="PDF files, or directories to search for them")
    ap.add_argument("--image-frac", type=float, default=IMAGE_FRAC,
                    help=f"page-area fraction that counts as a dominant image (default {IMAGE_FRAC})")
    ap.add_argument("--text-floor", type=int, default=TEXT_FLOOR,
                    help=f"chars below which a page's text is 'thin' (default {TEXT_FLOOR})")
    ap.add_argument("--json", metavar="PATH", help="also write the full report as JSON")
    args = ap.parse_args()

    files: list[str] = []
    for p in args.paths:
        files.extend(sorted(glob.glob(os.path.join(p, "*.pdf"))) if os.path.isdir(p) else [p])
    files = [f for f in files if f.lower().endswith(".pdf")]
    if not files:
        return ap.error("no PDFs found in the given paths")

    rows = [scan_file(f, args.image_frac, args.text_floor) for f in files]
    ok = [r for r in rows if not r["error"]]

    print(f"{'file':<28}{'KB':>8}{'pages':>7}{'chars':>9}{'ch/pg':>7}{'imgs':>6}{'vision':>8}")
    print("-" * 73)
    for r in rows:
        if r["error"]:
            print(f"{r['file']:<28}  !! {r['error'][:40]}")
            continue
        per = r["chars"] // max(r["pages"], 1)
        print(f"{r['file']:<28}{r['size_kb']:>8}{r['pages']:>7}{r['chars']:>9}{per:>7}"
              f"{r['images']:>6}{r['vision_pages']:>8}")

    if not ok:
        print("\nNo readable PDFs.")
        return 2

    pages = sum(r["pages"] for r in ok)
    chars = sum(r["chars"] for r in ok)
    vision = sum(r["vision_pages"] for r in ok)
    empty = sum(r["empty_pages"] for r in ok)
    per_page = chars / max(pages, 1)
    per_file = [r["chars"] / max(r["pages"], 1) for r in ok]

    print("-" * 73)
    print(f"{'TOTAL':<28}{sum(r['size_kb'] for r in ok):>8}{pages:>7}{chars:>9}{int(per_page):>7}"
          f"{sum(r['images'] for r in ok):>6}{vision:>8}")

    tokens = int(chars / CHARS_PER_TOKEN)
    print(f"\n{len(ok)} files, {pages} pages, {chars:,} chars of extractable text")
    print(f"  median chars/page      {int(statistics.median(per_file)) if per_file else 0}")
    print(f"  near-empty pages       {empty} ({empty / pages:.1%})")
    print(f"  vision candidates      {vision} ({vision / pages:.1%})")
    print(f"  est. tokens of text    ~{tokens:,}")

    scanned = per_page < SCANNED_CHARS_PER_PAGE
    print("\nVERDICT")
    if scanned:
        print(f"  Looks SCANNED ({per_page:.0f} chars/page). This pipeline extracts text and will")
        print("  produce almost nothing. Use OCR (ocrmypdf, tesseract) or an all-vision tool.")
        return 2

    mins = vision * SECONDS_PER_VISION_PAGE / 60
    print(f"  Text-native ({per_page:.0f} chars/page). This pipeline fits.")
    print(f"  Extraction is near-free; {vision} pages ({vision / pages:.1%}) need the vision model.")
    print(f"  Rough one-time vision cost: ~{mins:.0f} min at {SECONDS_PER_VISION_PAGE}s/page.")
    print(f"  An all-vision tool would instead process all {pages} pages "
          f"(~{pages * SECONDS_PER_VISION_PAGE / 3600:.1f} hr).")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"files": rows, "totals": {
                "files": len(ok), "pages": pages, "chars": chars, "images": sum(r["images"] for r in ok),
                "vision_pages": vision, "empty_pages": empty, "est_tokens": tokens,
                "chars_per_page": round(per_page, 1), "scanned": scanned,
            }, "thresholds": {"image_frac": args.image_frac, "text_floor": args.text_floor}}, fh, indent=2)
        print(f"\nJSON report written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
