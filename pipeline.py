#!/usr/bin/env python3
"""Stage 7: run the stages in order. No logic of its own.

Every stage remains independently runnable, and this orchestrator exists only so
the common case is one command. If something goes wrong, run the failing stage
on its own -- that is the whole reason the boundaries are kept clean.

Usage:
    python pipeline.py CORPUS_DIR --out out/ [--model MODEL] [--files a,b]

Without --model the vision stage is skipped, which is a legitimate way to run:
extraction alone already covers the large majority of pages.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

import assemble
import common
import extract
import index as index_stage
import snapshot as snapshot_mod


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus", help="directory of source PDFs")
    ap.add_argument("--out", default="out", help="working directory for artifacts (default out/)")
    ap.add_argument("--md", default=None, help="Markdown output directory (default OUT/md)")
    ap.add_argument("--model", default=None,
                    help="vision model; omit to skip enrichment entirely")
    ap.add_argument("--endpoint", default=None, help="local model endpoint")
    ap.add_argument("--files", help="comma-separated stems to process, instead of all")
    ap.add_argument("--image-frac", type=float, default=common.IMAGE_FRAC)
    ap.add_argument("--text-floor", type=int, default=common.TEXT_FLOOR)
    ap.add_argument("--dpi", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="redo work already recorded as done")
    ap.add_argument("--config", metavar="SNAPSHOT",
                    help="re-run with the settings recorded in a previous run's snapshot")
    ap.add_argument("--no-snapshot", action="store_true",
                    help="skip writing the run snapshot")
    args = ap.parse_args(argv)

    if args.config:
        # Snapshot settings are defaults, not overrides: anything typed on this
        # command line still wins, so a replay can be adjusted one flag at a time.
        try:
            recorded = snapshot_mod.load(args.config)
        except (OSError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        typed = set(argv if argv is not None else sys.argv[1:])
        for key, value in (recorded.get("settings") or {}).items():
            if key in ("corpus", "out", "md") or value is None:
                continue
            if not hasattr(args, key) or any(a.startswith("--" + key.replace("_", "-")) for a in typed):
                continue
            setattr(args, key, value)
        snapshot_mod.warn_on_drift(recorded)

    md_dir = args.md or os.path.join(args.out, "md")
    passthrough = (["--files", args.files] if args.files else [])

    print("== extract ==")
    rc = extract.main([args.corpus, "--out", args.out,
                       "--image-frac", str(args.image_frac),
                       "--text-floor", str(args.text_floor)]
                      + passthrough + (["--force"] if args.force else []))
    if rc:
        return rc

    if args.model:
        import enrich
        print("\n== enrich ==")
        argv_enrich = [args.out, "--model", args.model, "--corpus", args.corpus]
        if args.endpoint:
            argv_enrich += ["--endpoint", args.endpoint]
        if args.dpi:
            argv_enrich += ["--dpi", str(args.dpi)]
        rc = enrich.main(argv_enrich + passthrough + (["--force"] if args.force else []))
        if rc:
            return rc
    else:
        print("\n== enrich == skipped (no --model); flagged pages stay text-only")

    print("\n== assemble ==")
    rc = assemble.main([args.out, "--md", md_dir] + passthrough)
    if rc:
        return rc

    print("\n== index ==")
    rc = index_stage.main([md_dir, "--out", os.path.join(args.out, "index.json")] + passthrough)
    if rc:
        return rc

    if not args.no_snapshot:
        settings = {
            "image_frac": args.image_frac, "text_floor": args.text_floor,
            "model": args.model, "endpoint": args.endpoint, "dpi": args.dpi,
            "files": args.files,
        }
        path = snapshot_mod.write(args.out, snapshot_mod.capture(args.corpus, args.out, settings))
        print(f"\nRun snapshot -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
