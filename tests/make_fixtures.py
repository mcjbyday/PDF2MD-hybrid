#!/usr/bin/env python3
"""Generate the synthetic PDFs the test suite runs against.

No corpus ships with this tool and none ever will -- a user's documents are
their business. So the fixtures are built here instead, as raw PDF bytes with no
authoring dependency. Writing the bytes directly is not the obvious choice, but
it buys the one thing these tests need: exact control over which glyph lands at
which coordinate in which face, so a test that says "these two columns must not
interleave" has a known-correct answer rather than a plausible one.

Base-14 fonts are used throughout (Helvetica for prose, Courier for code) so
that character metrics come from the reader's built-in tables and the fixtures
stay byte-identical across machines.

    python tests/make_fixtures.py [OUT_DIR]
"""

from __future__ import annotations

import os
import sys
import zlib
from typing import List, Optional, Tuple

PAGE_W, PAGE_H = 612.0, 792.0


def _esc(text: str) -> bytes:
    out = text.encode("latin-1", "replace")
    for a, b in ((b"\\", b"\\\\"), (b"(", b"\\("), (b")", b"\\)")):
        out = out.replace(a, b)
    return out


class Page:
    """A page under construction, in top-left-origin coordinates."""

    def __init__(self, width: float = PAGE_W, height: float = PAGE_H):
        self.width, self.height = width, height
        self.ops: List[bytes] = []
        self.fonts = set()
        self.image: Optional[Tuple[float, float, float, float]] = None

    def text(self, x: float, top: float, s: str, font: str = "F1", size: float = 11.0) -> "Page":
        """Place a string with its baseline `top` points below the page top."""
        self.fonts.add(font)
        y = self.height - top - size
        self.ops.append(b"BT /%s %g Tf 1 0 0 1 %g %g Tm (%s) Tj ET" %
                        (font.encode(), size, x, y, _esc(s)))
        return self

    def lines(self, x: float, top: float, rows: List[str], font: str = "F1",
              size: float = 11.0, leading: float = 14.0) -> "Page":
        for i, row in enumerate(rows):
            if row:
                self.text(x, top + i * leading, row, font, size)
        return self

    def picture(self, x: float, top: float, w: float, h: float) -> "Page":
        """Place the shared placeholder image, sized in points."""
        self.image = (x, self.height - top - h, w, h)
        return self

    def content(self) -> bytes:
        ops = list(self.ops)
        if self.image:
            x, y, w, h = self.image
            ops.append(b"q %g 0 0 %g %g %g cm /Im0 Do Q" % (w, h, x, y))
        return b"\n".join(ops)


def write_pdf(path: str, pages: List[Page]) -> str:
    """Serialize pages to a PDF with a valid cross-reference table."""
    objects: List[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    font_ids = {}
    for tag, base in (("F1", "Helvetica"), ("F2", "Helvetica-Bold"), ("F3", "Courier")):
        font_ids[tag] = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /%s "
                            b"/Encoding /WinAnsiEncoding >>" % base.encode())

    # One tiny grey image, reused by every page that needs one. Its content is
    # irrelevant -- only the area it covers matters to triage.
    raw = bytes([128, 128, 128]) * 64
    comp = zlib.compress(raw)
    image_id = add(b"<< /Type /XObject /Subtype /Image /Width 8 /Height 8 "
                   b"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode "
                   b"/Length %d >>\nstream\n" % len(comp) + comp + b"\nendstream")

    pages_id = add(b"")  # reserved; filled once the kids are known
    kids: List[int] = []
    for page in pages:
        stream = page.content()
        content_id = add(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
        fonts = b" ".join(b"/%s %d 0 R" % (t.encode(), font_ids[t]) for t in sorted(page.fonts)) \
            or b"/F1 %d 0 R" % font_ids["F1"]
        xobj = b" /XObject << /Im0 %d 0 R >>" % image_id if page.image else b""
        kids.append(add(
            b"<< /Type /Page /Parent %d 0 R /MediaBox [0 0 %g %g] "
            b"/Resources << /Font << %s >>%s >> /Contents %d 0 R >>"
            % (pages_id, page.width, page.height, fonts, xobj, content_id)))

    objects[pages_id - 1] = (b"<< /Type /Pages /Count %d /Kids [%s] >>"
                             % (len(kids), b" ".join(b"%d 0 R" % k for k in kids)))
    root_id = add(b"<< /Type /Catalog /Pages %d 0 R >>" % pages_id)

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n" % (len(objects) + 1)
    out += b"0000000000 65535 f \n"
    for off in offsets[1:]:
        out += b"%010d 00000 n \n" % off
    out += (b"trailer\n<< /Size %d /Root %d 0 R >>\nstartxref\n%d\n%%%%EOF\n"
            % (len(objects) + 1, root_id, xref))

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(bytes(out))
    return path


# --- the fixtures ---------------------------------------------------------

LEFT_COLUMN = [
    "The left column carries prose that",
    "runs for several lines so the column",
    "has real vertical extent to cluster.",
    "Each sentence belongs to this side",
    "of the page and to no other.",
]

RIGHT_COLUMN = [
    "def configure(options):",
    "    for key in sorted(options):",
    "        apply(key, options[key])",
    "    return True",
]


def build(out_dir: str) -> List[str]:
    made = []

    # 1. Single column. The regression that matters most: this must pass
    #    through in plain top-to-bottom order, untouched by column logic.
    p = Page()
    p.text(72, 72, "Single Column Document", "F2", 22)
    p.text(72, 110, "Introduction", "F2", 15)
    p.lines(72, 140, [
        "This page has exactly one column of text.",
        "Every line runs the full measure of the page body.",
        "Reading order here is unambiguous and must not change.",
    ])
    made.append(write_pdf(os.path.join(out_dir, "single-column.pdf"), [p]))

    # 2. Two columns: prose beside code, under a full-width title. Naive
    #    top-to-bottom ordering interleaves these into nonsense.
    p = Page()
    p.text(72, 60, "Two Column Layout", "F2", 22)
    p.lines(72, 120, LEFT_COLUMN, "F1", 11)
    p.lines(340, 120, RIGHT_COLUMN, "F3", 11)
    made.append(write_pdf(os.path.join(out_dir, "two-column.pdf"), [p]))

    # 3. Heading hierarchy, sizes descending. Levels must come from clustering
    #    across the document, never from absolute point sizes.
    p = Page()
    p.text(72, 60, "Title Of The Document", "F2", 24)
    p.text(72, 110, "A Major Section", "F2", 18)
    p.lines(72, 145, ["Body text under the major section heading."])
    p.text(72, 185, "A Minor Section", "F2", 14)
    p.lines(72, 215, ["Body text under the minor section heading."])
    made.append(write_pdf(os.path.join(out_dir, "headings.pdf"), [p]))

    # 4. Image-dominant with a thin caption: the page triage must flag.
    p = Page()
    p.text(72, 60, "Figure 1", "F2", 16)
    p.picture(72, 100, 460, 560)
    p.text(72, 680, "A short caption.", "F1", 11)
    # ...and a companion page with the same big image but a full explanation,
    # which must NOT be flagged. A picture beside its own explanation is
    # already served by extraction.
    q = Page()
    q.text(72, 50, "Figure 2, Explained", "F2", 16)
    q.picture(72, 80, 460, 400)
    q.lines(72, 500, [
        "This page carries the same dominant image as the previous one, but the",
        "text layer explains it in full, and so the picture is illustration of",
        "text that is already present rather than the only place meaning lives.",
        "Triage must leave this page alone. The threshold is a character count,",
        "so these lines exist to carry it comfortably past the floor of four",
        "hundred characters without relying on any single long sentence to do",
        "the work. That is the entire point of the second half of the rule.",
    ])
    made.append(write_pdf(os.path.join(out_dir, "figures.pdf"), [p, q]))

    # 5. A dense grid: five labelled cells whose gutters are narrower than a
    #    character. Whitespace projection cannot see these columns at all; only
    #    the recurrence of their left edges gives them away. The reading order
    #    must keep each label with its own description.
    p = Page()
    p.text(60, 50, "Comparison Of Five Things", "F2", 20)
    p.lines(60, 95, ["A full width sentence that runs across every column below it."], "F1", 10)
    cells = [
        (60, "ALPHA", ["First option here.", "Cheap and simple."]),
        (170, "BETA", ["Second option.", "Faster but larger."]),
        (280, "GAMMA", ["Third option.", "Needs a network."]),
        (390, "DELTA", ["Fourth option.", "Rarely the right", "choice for this."]),
        (500, "EPSILON", ["Fifth option.", "Deprecated now."]),
    ]
    for x, label, body in cells:
        p.text(x, 140, label, "F2", 12)
        p.lines(x, 165, body, "F1", 9, leading=12)
    made.append(write_pdf(os.path.join(out_dir, "grid.pdf"), [p]))

    # 6. Single column with a consistently indented block. Structurally this
    #    looks like two columns -- repeated left edges at two x positions -- and
    #    it must NOT be reordered. This is the false positive that the column
    #    logic is most likely to produce and least likely to be noticed.
    p = Page()
    p.text(72, 60, "Indented Prose", "F2", 18)
    p.lines(72, 100, ["This paragraph sits at the left margin of the page body."], "F1", 11)
    p.lines(140, 130, [
        "This block is indented from the margin.",
        "It stays indented for several lines.",
        "It is still one column of reading.",
        "Nothing here runs beside anything else.",
    ], "F1", 11)
    p.lines(72, 200, ["And the text returns to the margin afterwards."], "F1", 11)
    made.append(write_pdf(os.path.join(out_dir, "indented.pdf"), [p]))

    # 7. A page with no text at all, and no image either: neither stage should
    #    claim it, and nothing downstream should crash on it.
    made.append(write_pdf(os.path.join(out_dir, "empty.pdf"), [Page()]))

    return made


def main() -> int:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "fixtures")
    for path in build(out_dir):
        print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
