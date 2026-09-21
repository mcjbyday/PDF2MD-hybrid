#!/usr/bin/env python3
"""Shared vocabulary for every stage: the triage rule, file discovery, and artifact I/O.

The triage rule lives here and nowhere else. It is the pipeline's whole thesis --
a page goes to a vision model only when a picture carries meaning the text layer
does not -- and duplicating its thresholds across stages is how they drift apart.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence

# Tuning knobs. Both are judgment calls, not laws, and both are corpus-dependent:
# they were calibrated against one corpus and are exposed as CLI flags on every
# stage precisely so you can re-tune them against yours. Raise IMAGE_FRAC or
# lower TEXT_FLOOR to send fewer pages to the model.
IMAGE_FRAC = 0.18
TEXT_FLOOR = 400

# Pages this sparse are almost certainly scanned rather than born-digital.
SCANNED_CHARS_PER_PAGE = 50


def needs_vision(chars: int, image_frac: float,
                 image_frac_threshold: float = IMAGE_FRAC,
                 text_floor: int = TEXT_FLOOR) -> bool:
    """The triage rule. A dominant image AND a thin text layer."""
    return image_frac > image_frac_threshold and chars < text_floor


def page_image_frac(page) -> float:
    """Largest embedded image as a fraction of page area, from a pdfplumber page."""
    page_area = abs(float(page.width) * float(page.height)) or 1.0
    frac = 0.0
    for im in page.images:
        area = abs((im["x1"] - im["x0"]) * (im["bottom"] - im["top"]))
        frac = max(frac, area / page_area)
    return frac


# --- file discovery -------------------------------------------------------

def find_pdfs(paths: Sequence[str], only: Optional[Iterable[str]] = None) -> List[str]:
    """Expand files and directories into a sorted list of PDF paths.

    `only` filters by stem, so `--files a,b` works the same way on every stage.
    """
    files: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, "*.pdf"))))
        else:
            files.append(p)
    files = [f for f in files if f.lower().endswith(".pdf")]
    if only is not None:
        wanted = {s.strip() for s in only if s.strip()}
        files = [f for f in files if stem(f) in wanted]
    return files


def parse_files_flag(value: Optional[str]) -> Optional[List[str]]:
    """`--files a,b` -> ['a', 'b']; absent -> None (meaning 'all')."""
    if not value:
        return None
    return [s for s in (part.strip() for part in value.split(",")) if s]


def stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- artifact I/O ---------------------------------------------------------

def pages_path(out_dir: str, name: str) -> str:
    return os.path.join(out_dir, name + ".pages.json")


def enrich_log_path(out_dir: str, name: str) -> str:
    return os.path.join(out_dir, name + ".enrich.jsonl")


def load_pages(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_pages(path: str, doc: Dict[str, Any]) -> None:
    """Write atomically. A half-written artifact is worse than no artifact --
    the next stage cannot tell the difference between the two."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def find_pages_files(out_dir: str, only: Optional[Iterable[str]] = None) -> List[str]:
    files = sorted(glob.glob(os.path.join(out_dir, "*.pages.json")))
    if only is not None:
        wanted = {s.strip() for s in only if s.strip()}
        files = [f for f in files if os.path.basename(f)[:-len(".pages.json")] in wanted]
    return files
