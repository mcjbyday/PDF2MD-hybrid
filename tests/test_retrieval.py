"""Retrieval quality: function words must not outrank content words.

The read path is documented as taking a question, so "how do I ..." is the
expected shape of a query rather than an edge case. These tests exist because
the failure is silent: a wrong page is returned confidently, and a caller
feeding it to a model has no way to notice.
"""

from __future__ import annotations

import json
import os

import pytest

from pdf2md_hybrid import index as index_stage

TOPICS = [
    "caching", "sharding", "indexing", "batching", "streaming", "queueing",
    "routing", "throttling", "hashing", "compaction", "replication",
    "snapshotting", "tracing", "profiling", "buffering", "scheduling",
    "partitioning", "checkpointing", "serialization", "validation",
    "migration", "provisioning", "monitoring", "alerting", "rotation",
    "archival", "compression", "encryption",
]

RETRIES_PAGE = (
    "# Configuring Retries {#p040}\n\n"
    "To configure retries, set the retry policy on the client. A bounded retry "
    "policy prevents a failing dependency from exhausting the connection pool. "
    "Retries should use exponential backoff with jitter so that simultaneous "
    "failures never synchronize across the fleet. Configure the maximum attempt "
    "count explicitly, because the default is generous and masks upstream "
    "faults. A retry budget caps total retry volume, and a circuit breaker stops "
    "retries entirely once the dependency is clearly unhealthy. Tune the c and r "
    "parameters for the workload.\n"
)

# Ten tokens of navigation furniture. Short pages are the amplifier: BM25's
# length normalisation lets a single stopword hit on a stub outscore several
# content hits on a full page.
NAV_STUB = "# Learn More {#p068}\n\nWhich Tool Do I Use? Getting Started\n"


@pytest.fixture(scope="session")
def guide_md(tmp_path_factory) -> str:
    """A corpus where one page genuinely answers the question and one is a stub."""
    md_dir = tmp_path_factory.mktemp("swrepro")
    body = []
    for n, topic in enumerate(TOPICS, start=1):
        body.append(
            f"# {topic.title()} Overview {{#p{n:03d}}}\n\n"
            f"The {topic} subsystem manages {topic} across the cluster. Operators "
            f"tune {topic} through the configuration file. Correct {topic} "
            f"behaviour depends on the workload shape and on how much memory the "
            f"node has available for {topic}.\n")
    body.append(RETRIES_PAGE)
    body.append(NAV_STUB)
    (md_dir / "guide.md").write_text(
        "---\nsource: guide.pdf\npages: 30\n---\n\n" + "\n".join(body), encoding="utf-8")
    return str(md_dir)


@pytest.fixture(scope="session")
def guide_index(guide_md):
    return index_stage.build(guide_md)


def top(index, query):
    hits = index_stage.search(index, query, k=1)
    return hits[0]["anchor"] if hits else None


# --- the reported regression ---------------------------------------------

def test_question_phrasing_matches_keyword_phrasing(guide_index):
    """The core regression. Asking a question must retrieve what the keywords do."""
    assert top(guide_index, "how do I configure retries") == "p040"
    assert top(guide_index, "configure retries") == "p040"
    assert (top(guide_index, "how do I configure retries")
            == top(guide_index, "configure retries"))


def test_offtopic_question_returns_no_hits(guide_index):
    """Abstaining is what protects downstream grounding. A confidently wrong
    page is worse than nothing, because the model cannot tell it is unrelated."""
    assert index_stage.search(guide_index, "how do I bake sourdough bread", k=1) == []
    assert index_stage.search(guide_index, "bake sourdough bread", k=1) == []


@pytest.mark.parametrize("query", ["how do I", "what is this", "the a of and"])
def test_stopwords_only_query_returns_no_hits(guide_index, query):
    """No unfiltered fallback: a query of pure function words matches nothing."""
    assert index_stage.search(guide_index, query, k=5) == []


def test_short_page_does_not_outrank_content_page(guide_index):
    """The stub must not win on length normalisation alone."""
    ranked = [h["anchor"] for h in index_stage.search(
        guide_index, "how do I configure retries", k=10)]
    assert "p040" in ranked
    assert ranked.index("p040") == 0
    if "p068" in ranked:
        assert ranked.index("p040") < ranked.index("p068")


# --- the guards on the fix -----------------------------------------------

def test_single_character_terms_survive(guide_index):
    """Length is never the test: c, r and k are real terms in technical prose."""
    assert index_stage.tokenize("tune the c and r values") == ["tune", "c", "r", "values"]
    assert top(guide_index, "c r parameters") == "p040"


def test_index_and_query_tokenization_agree():
    """One definition for both paths. Filtering only one leaves them disagreeing
    about what a term is, which surfaces later as an inexplicable ranking."""
    text = "How do I configure the retry policy for a c compiler"
    assert index_stage.tokenize(text) == index_stage.tokenize(text)
    assert "i" not in index_stage.tokenize(text)
    assert "c" in index_stage.tokenize(text)


def test_stopword_list_is_overridable(guide_md):
    """A non-English corpus must be able to supply its own list, or none."""
    assert index_stage.tokenize("how do I", stopwords=[]) == ["how", "do", "i"]
    unfiltered = index_stage.build(guide_md, stopwords=frozenset())
    assert index_stage.search(unfiltered, "how do I", k=1), "override did not apply"
    assert index_stage.tokenize("der die das", stopwords={"der", "die", "das"}) == []


def test_index_records_the_list_it_was_built_with(guide_index):
    """So a query is scored against the same vocabulary the index used."""
    assert "i" in guide_index["stopwords"]
    assert "retries" not in guide_index["stopwords"]


# --- stale artifacts ------------------------------------------------------

def test_stale_index_is_rejected(guide_index, tmp_path):
    """A pre-fix index holds frequencies for terms the query path no longer
    produces. Serving it returns wrong rankings and says nothing."""
    path = tmp_path / "old.json"
    stale = dict(guide_index)
    stale.pop("schema", None)
    path.write_text(json.dumps(stale), encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        index_stage.load(str(path))
    assert "schema" in str(excinfo.value).lower()
    assert "rebuild" in str(excinfo.value).lower(), "the error must say what to do"


def test_corpus_rebuilds_a_stale_index(fixtures_dir, tmp_path):
    """Corpus.open has the Markdown to hand, so it rebuilds rather than failing."""
    from pdf2md_hybrid import Corpus, convert

    out = tmp_path / "out"
    convert(fixtures_dir, str(out))
    index_path = out / "index.json"
    stale = json.loads(index_path.read_text())
    stale["schema"] = 1
    index_path.write_text(json.dumps(stale), encoding="utf-8")

    corpus = Corpus.open(str(out))
    assert corpus.stats()["pages"] > 0
    assert corpus.search("configure options", k=1)[0].file == "two-column"


def test_written_index_carries_its_schema(guide_md, tmp_path):
    out = tmp_path / "index.json"
    assert index_stage.main([guide_md, "--out", str(out)]) == 0
    assert json.loads(out.read_text())["schema"] == index_stage.SCHEMA


def test_no_stopwords_flag_round_trips(guide_md, tmp_path):
    """The escape hatch has to survive being written and read back."""
    out = tmp_path / "raw.json"
    assert index_stage.main([guide_md, "--out", str(out), "--no-stopwords"]) == 0
    loaded = index_stage.load(str(out))
    assert loaded["stopwords"] == []
    assert index_stage.search(loaded, "how do I", k=1), "filtering was not disabled"
