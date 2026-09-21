#!/usr/bin/env python3
"""Stage 5: pages JSON -> one Markdown file per source PDF.

Two rules shape this stage. Anchors must be stable, because they are how a
retrieval hit cites the page it came from. And generated text must stay visibly
separate from parsed text -- one is ground truth and the other is a model's
reading of a picture, and a consumer debugging a wrong answer has to be able to
tell which is which.

Usage:
    python assemble.py out/ --md out/md/ [--files a,b]
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import date
from typing import Any, Dict, List, Optional

import common

# Characters that would otherwise be read as inline HTML outside a code fence.
_ESCAPE = re.compile(r"[<>]")


def escape_inline(text: str) -> str:
    return _ESCAPE.sub(lambda m: "&lt;" if m.group() == "<" else "&gt;", text)


def anchor(page_number: int) -> str:
    """Stable per-page anchor. Zero-padded so it sorts and never collides."""
    return "p%03d" % page_number


def _yaml_scalar(value: Any) -> str:
    text = str(value)
    return '"%s"' % text.replace('"', '\\"') if re.search(r'[:#"\n]', text) else text


def render_page(page: Dict[str, Any]) -> List[str]:
    """Markdown lines for one page, anchor attached to its first heading."""
    out: List[str] = []
    tag = anchor(page["n"])
    used_anchor = False

    for block in page.get("blocks", []):
        kind = block.get("kind")
        text = (block.get("text") or "").rstrip()
        if not text.strip():
            continue
        if kind == "heading":
            level = min(max(int(block.get("level") or 2), 1), 6)
            suffix = "" if used_anchor else " {#%s}" % tag
            used_anchor = True
            out.append("%s %s%s" % ("#" * level, escape_inline(text), suffix))
            out.append("")
        elif kind == "term":
            # A grid cell's label, kept with the lines it introduces. Bold
            # rather than a heading: it labels a cell, not a section, and
            # promoting it would corrupt the document outline.
            out.append("**%s**" % escape_inline(text))
            out.append("")
        elif kind == "code":
            # Long enough to survive a fence inside the code. Extracted text
            # can contain anything, and a short fence would end the block early
            # and spill the rest of the listing into the document as prose.
            longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
            fence = "`" * max(3, longest + 1) + (block.get("lang") or "")
            out.append(fence)
            out.extend(text.split("\n"))
            out.append("`" * max(3, longest + 1))
            out.append("")
        else:
            out.append(escape_inline(text))
            out.append("")

    vision = page.get("vision") or {}
    body = (vision.get("text") or "").strip()
    if body:
        # Blockquoted and labelled, always. Never blended into parsed prose:
        # a reader must be able to tell a parser's output from a model's.
        out.append("> **Figure (%s):** generated description of this page's imagery." % tag)
        out.append(">")
        for line in body.split("\n"):
            out.append(("> " + line).rstrip())
        out.append("")

    if not out:
        # An empty page still needs its anchor, or every later citation that
        # points past it is describing a page that is not there.
        return ['<a id="%s"></a>' % tag, ""]
    if not used_anchor:
        out = ['<a id="%s"></a>' % tag, ""] + out
    return out


def render_document(doc: Dict[str, Any]) -> str:
    lines = [
        "---",
        "source: %s" % _yaml_scalar(doc.get("source", "")),
        "pages: %d" % doc.get("page_count", 0),
        "generated: %s" % date.today().isoformat(),
        "---",
        "",
    ]
    for page in doc.get("pages", []):
        lines.extend(render_page(page))
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).rstrip() + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", help="directory holding .pages.json artifacts")
    ap.add_argument("--md", default=None, help="directory for Markdown output (default OUT_DIR/md)")
    ap.add_argument("--files", help="comma-separated stems to process, instead of all")
    args = ap.parse_args(argv)

    pages_files = common.find_pages_files(args.out_dir, common.parse_files_flag(args.files))
    if not pages_files:
        print(f"No .pages.json artifacts in {args.out_dir}. Run extract.py first.", file=sys.stderr)
        return 1

    md_dir = args.md or os.path.join(args.out_dir, "md")
    os.makedirs(md_dir, exist_ok=True)

    total_pages = total_figures = 0
    for pages_file in pages_files:
        doc = common.load_pages(pages_file)
        name = os.path.basename(pages_file)[:-len(".pages.json")]
        target = os.path.join(md_dir, name + ".md")
        text = render_document(doc)
        with open(target, "w", encoding="utf-8") as fh:
            fh.write(text)
        figures = sum(1 for p in doc.get("pages", []) if (p.get("vision") or {}).get("text"))
        total_pages += doc.get("page_count", 0)
        total_figures += figures
        print(f"  {name:<32} {doc.get('page_count', 0):>5} pages  {figures:>4} figures  "
              f"{len(text) / 1024:>7.1f} KB")

    print(f"\n{len(pages_files)} files, {total_pages} pages, "
          f"{total_figures} figure descriptions -> {md_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
