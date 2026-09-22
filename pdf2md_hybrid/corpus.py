#!/usr/bin/env python3
"""The read side: open a finished run and query it.

Everything else in this package is about *producing* Markdown. This module is
for the program that consumes it, and it exists because the alternative -- a
downstream project reaching into `out/` and parsing the artifacts itself --
makes every internal format a public contract that can never change.

The shape it encourages is the one the output is designed for: hand a model a
small always-loaded outline, then retrieve the few pages a question actually
needs. Not the whole corpus in a context window.

    from pdf2md_hybrid import Corpus

    corpus = Corpus.open("out/")
    print(corpus.outline_markdown())          # a map of what exists
    for hit in corpus.search("exception handling", k=3):
        print(hit.cite, hit.excerpt)
        print(hit.text())                     # the full page
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterator, List, Optional

from . import common, index as index_stage


class Hit:
    """One retrieved page, and the means to read it or cite it."""

    __slots__ = ("file", "page", "anchor", "score", "excerpt", "_corpus")

    def __init__(self, corpus: "Corpus", record: Dict[str, Any]):
        self._corpus = corpus
        self.file: str = record["file"]
        self.page: int = record["page"]
        self.anchor: str = record["anchor"]
        self.score: float = record["score"]
        self.excerpt: str = record["excerpt"]

    @property
    def cite(self) -> str:
        """A stable reference a model can quote back, e.g. ``manual-a#p031``.

        Anchors are stable across re-runs, so a citation stays valid as long as
        the source PDF does.
        """
        return f"{self.file}#{self.anchor}"

    def text(self) -> str:
        """The full Markdown of this page, not just the excerpt."""
        return self._corpus.page(self.file, self.anchor)

    def __repr__(self) -> str:
        return f"<Hit {self.cite} score={self.score:.3f}>"


class Corpus:
    """A finished pipeline run, opened for reading.

    Construct with :meth:`open`. Nothing here writes, and nothing here needs a
    model or the source PDFs -- only the `out/` directory the pipeline left
    behind, which is what makes it safe to ship to a downstream service.
    """

    def __init__(self, out_dir: str, index: Dict[str, Any], md_dir: str):
        self.out_dir = out_dir
        self.md_dir = md_dir
        self._index = index
        self._pages: Dict[str, Dict[str, str]] = {}

    # --- opening ----------------------------------------------------------

    @classmethod
    def open(cls, out_dir: str, md_dir: Optional[str] = None) -> "Corpus":
        """Open a run directory, building the index if it was never written."""
        md = md_dir or os.path.join(out_dir, "md")
        if not os.path.isdir(md):
            raise FileNotFoundError(
                f"no assembled Markdown in {md}. Run the pipeline, or pass md_dir.")

        index_path = os.path.join(out_dir, "index.json")
        if os.path.exists(index_path):
            index = index_stage.load(index_path)
        else:
            # Building takes milliseconds at this corpus size, so a missing
            # index is an inconvenience rather than an error.
            index = index_stage.build(md)
        return cls(out_dir, index, md)

    # --- what exists ------------------------------------------------------

    def sources(self) -> List[str]:
        """Document stems, one per source PDF."""
        return sorted({entry["file"] for entry in self._index["outline"]}
                      | {doc["file"] for doc in self._index["docs"]})

    def outline(self) -> List[Dict[str, Any]]:
        """Every heading, with its file, anchor and level."""
        return list(self._index["outline"])

    def outline_markdown(self, max_level: int = 3) -> str:
        """The outline as an indented list, sized to sit in a system prompt.

        This is the "map of what exists" half of the retrieval shape: small
        enough to keep loaded, specific enough for a model to know what it can
        ask for.
        """
        lines: List[str] = []
        current: Optional[str] = None
        for entry in self._index["outline"]:
            if entry["level"] > max_level:
                continue
            if entry["file"] != current:
                current = entry["file"]
                lines.append(f"\n## {current}")
            indent = "  " * (entry["level"] - 1)
            lines.append(f"{indent}- {entry['heading']} [{current}#{entry['anchor']}]")
        return "\n".join(lines).strip()

    # --- retrieval --------------------------------------------------------

    def search(self, query: str, k: int = 5) -> List[Hit]:
        """Rank page-sized chunks against the query, best first."""
        return [Hit(self, r) for r in index_stage.search(self._index, query, k)]

    def context(self, query: str, k: int = 3, separator: str = "\n\n---\n\n") -> str:
        """Retrieved pages joined and labelled, ready to paste into a prompt.

        Each page keeps its citation, so an answer built from this can point at
        where it came from -- which is the whole reason the anchors are stable.
        """
        parts = []
        for hit in self.search(query, k):
            parts.append(f"[{hit.cite}]\n\n{hit.text().strip()}")
        return separator.join(parts)

    # --- reading ----------------------------------------------------------

    def markdown(self, source: str) -> str:
        """The full Markdown of one document."""
        path = os.path.join(self.md_dir, source + ".md")
        if not os.path.exists(path):
            raise FileNotFoundError(f"no such document: {source}")
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()

    def page(self, source: str, anchor: str) -> str:
        """The Markdown of a single page, addressed by its stable anchor."""
        if source not in self._pages:
            body = index_stage.strip_front_matter(self.markdown(source))
            self._pages[source] = dict(index_stage.split_pages(body))
        try:
            return self._pages[source][anchor]
        except KeyError:
            raise KeyError(f"no page {anchor} in {source}") from None

    def pages(self, source: str) -> Iterator[Dict[str, Any]]:
        """Every page of a document, in order, as ``{anchor, text}``."""
        body = index_stage.strip_front_matter(self.markdown(source))
        for anchor, text in index_stage.split_pages(body):
            yield {"anchor": anchor, "text": text}

    # --- provenance -------------------------------------------------------

    def snapshot(self) -> Optional[Dict[str, Any]]:
        """The run record, when the pipeline wrote one.

        Worth surfacing downstream: it names the model, the thresholds and the
        library versions that produced this text, which is the difference
        between a changed answer being explicable and being a mystery.
        """
        from . import snapshot as snapshot_mod
        path = os.path.join(self.out_dir, snapshot_mod.SNAPSHOT_NAME)
        return snapshot_mod.load(path) if os.path.exists(path) else None

    def stats(self) -> Dict[str, Any]:
        """Counts a downstream service can log or assert against at startup."""
        return {
            "sources": len(self.sources()),
            "pages": self._index["n_docs"],
            "headings": len(self._index["outline"]),
            "terms": len(self._index["df"]),
        }

    def __repr__(self) -> str:
        s = self.stats()
        return f"<Corpus {self.out_dir} {s['sources']} docs, {s['pages']} pages>"
