#!/usr/bin/env python3
"""A record of exactly what produced a given run, and a way to reproduce it.

Output from this pipeline depends on more than the flags you typed. It depends
on the heuristic ratios compiled into `layout.py`, on the prompt in
`enrich.py`, on which model answered, and on the two PDF libraries' own
versions -- extraction thresholds are calibrated against pdfplumber's view of a
page, and a different library reports a different one.

So when output changes between two runs, "what did I change?" is not usually
answerable from the command line alone. This module writes the whole answer
next to the artifacts, and reads it back to re-run under the same settings.

    python pipeline.py corpus/ --out out/ --model M      # writes the snapshot
    python pipeline.py corpus/ --out out2/ --config out/pipeline.snapshot.json
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from typing import Any, Dict, List, Optional

from . import common

SNAPSHOT_NAME = "pipeline.snapshot.json"
SCHEMA = 1


def _tool_version() -> str:
    try:
        from importlib.metadata import version
        return version("PDF2MD-hybrid")
    except Exception:
        return "unknown"


def _dependency_versions() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for name in ("pdfplumber", "pypdfium2"):
        try:
            from importlib.metadata import version
            out[name] = version(name)
        except Exception:
            out[name] = "unknown"
    return out


def _git_commit() -> Optional[str]:
    """The revision of the tool itself, when it is running from a checkout."""
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        result = subprocess.run(["git", "-C", here, "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=5)
        if result.returncode:
            return None
        commit = result.stdout.strip()
        dirty = subprocess.run(["git", "-C", here, "status", "--porcelain"],
                               capture_output=True, text=True, timeout=5)
        # A dirty tree means the commit does not fully describe the run, and
        # silently implying otherwise is worse than admitting it.
        return commit + ("+dirty" if dirty.stdout.strip() else "")
    except Exception:
        return None


def _heuristics() -> Dict[str, Any]:
    """The tuning constants that are not command-line flags.

    These are the ones that change output without changing the command, which
    is exactly why they belong in the record.
    """
    from . import layout
    return {name: getattr(layout, name) for name in (
        "SEGMENT_GAP_RATIO", "SPINE_TOLERANCE_RATIO", "SPINE_MIN_ROWS",
        "SPINE_MIN_SEPARATION", "SPINE_MIN_COEXISTING_ROWS",
        "HEADING_SIZE_RATIO", "HEADING_MAX_VOLUME", "HEADING_VOLUME_MIN_SAMPLE",
        "HEADING_MAX_CHARS", "MIN_VISIBLE_SIZE_RATIO", "MIN_VISIBLE_PT",
        "MONOSPACE_CV", "MONOSPACE_MIN_SAMPLES",
    )}


def capture(corpus: str, out_dir: str, settings: Dict[str, Any]) -> Dict[str, Any]:
    """Build the snapshot for a completed run, including what it produced."""
    from . import enrich

    files: List[Dict[str, Any]] = []
    for path in common.find_pages_files(out_dir):
        try:
            doc = common.load_pages(path)
        except Exception:
            continue
        pages = doc.get("pages", [])
        files.append({
            "source": doc.get("source"),
            "source_sha256": doc.get("source_sha256"),
            "pages": doc.get("page_count", 0),
            "chars": sum(p.get("chars", 0) for p in pages),
            "flagged": sum(1 for p in pages if p.get("needs_vision")),
            "enriched": sum(1 for p in pages if (p.get("vision") or {}).get("text")),
        })

    return {
        "schema": SCHEMA,
        "created": common.utc_now(),
        "tool": {
            "name": "PDF2MD-hybrid",
            "version": _tool_version(),
            "commit": _git_commit(),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "dependencies": _dependency_versions(),
        },
        "settings": dict(settings, corpus=os.path.abspath(corpus)),
        # The prompt is part of what produced the output, so it travels with it.
        "prompt": enrich.PROMPT,
        "heuristics": _heuristics(),
        "results": {
            "files": files,
            "pages": sum(f["pages"] for f in files),
            "chars": sum(f["chars"] for f in files),
            "flagged": sum(f["flagged"] for f in files),
            "enriched": sum(f["enriched"] for f in files),
        },
    }


def write(out_dir: str, snapshot: Dict[str, Any]) -> str:
    path = os.path.join(out_dir, SNAPSHOT_NAME)
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(snapshot, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return path


def load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        snapshot = json.load(fh)
    if snapshot.get("schema") != SCHEMA:
        raise ValueError(f"{path}: unsupported snapshot schema {snapshot.get('schema')!r}")
    return snapshot


def warn_on_drift(snapshot: Dict[str, Any], stream=sys.stderr) -> List[str]:
    """Report where the current environment differs from a snapshot's.

    Reusing a snapshot's settings does not guarantee its output. Say so
    explicitly rather than letting a changed result look inexplicable.
    """
    drift: List[str] = []
    now = _heuristics()
    for name, was in (snapshot.get("heuristics") or {}).items():
        if name in now and now[name] != was:
            drift.append(f"heuristic {name}: snapshot {was}, now {now[name]}")

    deps = (snapshot.get("environment") or {}).get("dependencies") or {}
    for name, was in deps.items():
        now_version = _dependency_versions().get(name)
        if now_version and now_version != was:
            drift.append(f"{name}: snapshot {was}, now {now_version}")

    was_commit = (snapshot.get("tool") or {}).get("commit")
    now_commit = _git_commit()
    if was_commit and now_commit and was_commit != now_commit:
        drift.append(f"tool commit: snapshot {was_commit[:12]}, now {now_commit[:12]}")

    from . import enrich
    if snapshot.get("prompt") and snapshot["prompt"] != enrich.PROMPT:
        drift.append("the enrichment prompt has changed")

    if drift:
        print("Note: this run will not match the snapshot exactly:", file=stream)
        for line in drift:
            print(f"  - {line}", file=stream)
    return drift
