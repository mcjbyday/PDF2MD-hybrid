#!/usr/bin/env python3
"""Stage 6: make the corpus searchable without a vector store.

Two artifacts. An *outline* -- every heading with its file and anchor -- small
enough to hand a model whole as a map of what exists. And a *BM25* index over
page-sized chunks, in pure standard library.

That last part is a deliberate refusal. At roughly a megabyte of Markdown,
lexical search answers in milliseconds and adds no dependency; embeddings and a
vector database are a later optimization, to be reached for when retrieval
quality demonstrably demands it rather than because it is the expected answer.

Usage:
    python index.py out/md/ --out out/index.json
    python index.py out/md/ --out out/index.json --query "some phrase"
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys

from . import common
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Okapi BM25 defaults. k1 controls term-frequency saturation, b the strength of
# the length normalization; these are the standard values and there is no
# corpus-specific reason to move them.
K1 = 1.5
B = 0.75

_WORD = re.compile(r"[A-Za-z0-9_]+")

# The schema of a written index. Bumped when a change makes a stored index
# incompatible with the code that reads it -- stopword filtering does, because
# an index built without it holds document frequencies for terms the query path
# no longer produces. A stale index is rejected rather than served: retrieval
# fails silently, so the failure has to be made loud somewhere.
SCHEMA = 2

# English function words, removed from both the index and the query.
#
# This is a curated list, not a frequency cutoff, and that is the whole point.
# BM25's IDF term already damps words that appear in nearly every document, and
# it handles "the" and "a" correctly on its own. It cannot handle these, because
# in a technical corpus the worst offenders are *rare*: formal documentation
# seldom says "I", so BM25 reads "i" as a highly discriminative term and scores
# it above a genuine content word. The offenders sit in exactly the frequency
# band BM25 is designed to reward, so no threshold can separate them and the
# list has to be semantic.
#
# Interrogatives matter most here. The read path is documented as taking a
# question, so "how do I ..." is the expected shape of a query, not an edge case.
STOPWORDS = frozenset("""
a an the this that these those
i you he she it we they me my your his her its our their
is are was were be been being am
do does did done doing
have has had having
can could should would will shall may might must
how what why when where which who whom whose
and or but nor so yet if then than because
of in on at to for from by with without about into over under
as not no nor too very just only also
""".split())
_ANCHOR = re.compile(r"\{#(p\d+)\}|<a id=\"(p\d+)\">")
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)(?:\s*\{#p\d+\})?\s*$")


def tokenize(text: str, stopwords: Optional[Iterable[str]] = None) -> List[str]:
    """Split text into scored terms, dropping function words.

    One definition, used by both the index and the query path. Filtering in only
    one of them would leave the two disagreeing about what a term is, which is
    the kind of asymmetry that surfaces much later as an inexplicable ranking.

    Single characters are kept unless they are stopwords: `c`, `r` and `k` are
    real terms in a technical corpus, so length is never the test.

    Pass `stopwords` to supply another language's list, or an empty collection
    to disable filtering entirely.
    """
    stops = STOPWORDS if stopwords is None else frozenset(stopwords)
    return [w for w in (m.lower() for m in _WORD.findall(text)) if w not in stops]


def split_pages(markdown: str) -> List[Tuple[str, str]]:
    """Split a document into (anchor, text) chunks at each page anchor."""
    lines = markdown.split("\n")
    chunks: List[Tuple[str, List[str]]] = []
    current = "p000"
    buffer: List[str] = []
    for line in lines:
        match = _ANCHOR.search(line)
        if match:
            if buffer:
                chunks.append((current, buffer))
            current = match.group(1) or match.group(2)
            buffer = [line]
        else:
            buffer.append(line)
    if buffer:
        chunks.append((current, buffer))
    return [(a, "\n".join(b)) for a, b in chunks]


def strip_front_matter(text: str) -> str:
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            return text[end + 5:]
    return text


def build(md_dir: str, only: Optional[Iterable[str]] = None,
          stopwords: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    outline: List[Dict[str, Any]] = []
    docs: List[Dict[str, Any]] = []
    df: Dict[str, int] = {}

    wanted = {s.strip() for s in only if s.strip()} if only is not None else None
    for path in sorted(glob.glob(os.path.join(md_dir, "*.md"))):
        name = os.path.splitext(os.path.basename(path))[0]
        if wanted is not None and name not in wanted:
            continue
        with open(path, "r", encoding="utf-8") as fh:
            raw = fh.read()
        body = strip_front_matter(raw)

        for anchor, chunk in split_pages(body):
            for line in chunk.split("\n"):
                heading = _HEADING.match(line)
                if heading:
                    outline.append({"file": name, "anchor": anchor,
                                    "level": len(heading.group(1)),
                                    "heading": heading.group(2).strip()})
            tokens = tokenize(chunk, stopwords)
            if not tokens:
                continue
            freqs: Dict[str, int] = {}
            for token in tokens:
                freqs[token] = freqs.get(token, 0) + 1
            for token in freqs:
                df[token] = df.get(token, 0) + 1
            docs.append({
                "file": name,
                "page": int(anchor[1:]) if anchor[1:].isdigit() else 0,
                "anchor": anchor,
                "len": len(tokens),
                "tf": freqs,
                # A short excerpt, so a search result can be read without
                # opening the file it points at.
                "excerpt": " ".join(chunk.split())[:280],
            })

    avgdl = sum(d["len"] for d in docs) / len(docs) if docs else 0.0
    return {"schema": SCHEMA, "md_dir": os.path.abspath(md_dir), "outline": outline,
            "docs": docs, "df": df, "avgdl": avgdl, "n_docs": len(docs),
            # Recorded so a reader can tell a filtered index from an unfiltered
            # one, and so a non-English list travels with the index it built.
            "stopwords": sorted(STOPWORDS if stopwords is None else frozenset(stopwords))}


def search(index: Dict[str, Any], query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Rank page chunks against the query. IDF is computed here, not stored.

    A query that is entirely function words returns nothing. It must not fall
    back to the unfiltered tokens: a confident hit on an irrelevant page is
    worse than no hit at all, because the caller feeds it to a model as
    reference material with no way to tell that it is unrelated.
    """
    terms = tokenize(query, index.get("stopwords"))
    if not terms or not index.get("docs"):
        return []
    n = index["n_docs"]
    avgdl = index["avgdl"] or 1.0
    df = index["df"]

    idf = {}
    for term in set(terms):
        freq = df.get(term, 0)
        # Standard BM25 idf, floored: a term in nearly every document should
        # contribute nothing rather than a negative score.
        idf[term] = max(math.log(1 + (n - freq + 0.5) / (freq + 0.5)), 0.0)

    results = []
    for doc in index["docs"]:
        score = 0.0
        for term in terms:
            tf = doc["tf"].get(term)
            if not tf:
                continue
            norm = tf * (K1 + 1) / (tf + K1 * (1 - B + B * doc["len"] / avgdl))
            score += idf[term] * norm
        if score > 0:
            results.append({"file": doc["file"], "page": doc["page"],
                            "anchor": doc["anchor"], "score": round(score, 4),
                            "excerpt": doc["excerpt"]})
    results.sort(key=lambda r: (-r["score"], r["file"], r["page"]))
    return results[:k]


def load(path: str) -> Dict[str, Any]:
    """Read a written index, refusing one this code cannot score correctly."""
    with open(path, "r", encoding="utf-8") as fh:
        index = json.load(fh)
    found = index.get("schema", 1)
    if found != SCHEMA:
        raise ValueError(
            f"{path}: index schema {found}, this build reads {SCHEMA}. "
            "Rebuild it -- indexing a corpus takes seconds.")
    return index


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("md_dir", help="directory of assembled Markdown")
    ap.add_argument("--out", default=None, help="index path (default MD_DIR/../index.json)")
    ap.add_argument("--files", help="comma-separated stems to index, instead of all")
    ap.add_argument("--no-stopwords", action="store_true",
                    help="index every word, including function words "
                         "(for a corpus this list's language does not fit)")
    ap.add_argument("--query", help="build, then run this query and print the hits")
    ap.add_argument("-k", type=int, default=5, help="results to show with --query")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.md_dir):
        print(f"Not a directory: {args.md_dir}. Run assemble.py first.", file=sys.stderr)
        return 1

    index = build(args.md_dir, common.parse_files_flag(args.files),
                  stopwords=frozenset() if args.no_stopwords else None)
    if not index["docs"]:
        print(f"No Markdown found in {args.md_dir}.", file=sys.stderr)
        return 1

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.md_dir)), "index.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(index, fh, ensure_ascii=False)

    print(f"{index['n_docs']} page chunks, {len(index['outline'])} headings, "
          f"{len(index['df']):,} terms -> {out}")

    if args.query:
        hits = search(index, args.query, args.k)
        print(f"\n{len(hits)} hit(s) for {args.query!r}:")
        for hit in hits:
            print(f"  {hit['score']:>7.3f}  {hit['file']}#{hit['anchor']}")
            print(f"           {hit['excerpt'][:140]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
