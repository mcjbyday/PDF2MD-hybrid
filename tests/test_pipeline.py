"""Triage, enrichment resumability, assembly, and retrieval.

Only enrichment needs a model, and its HTTP call is mocked so the whole suite
runs with nothing installed but the two PDF libraries.
"""

from __future__ import annotations

import json
import os

import pytest

from pdf2md_hybrid import assemble, common, enrich, extract
from pdf2md_hybrid import index as index_stage


# --- 3. triage ------------------------------------------------------------

def test_triage_rule_needs_both_conditions():
    """A dominant image alone is not enough, and neither is thin text alone."""
    assert common.needs_vision(chars=20, image_frac=0.60)
    assert not common.needs_vision(chars=20, image_frac=0.05), "thin text, no image"
    assert not common.needs_vision(chars=5000, image_frac=0.60), "image beside its own explanation"


def test_triage_thresholds_are_overridable():
    assert not common.needs_vision(300, 0.20, image_frac_threshold=0.50)
    assert common.needs_vision(300, 0.20, image_frac_threshold=0.10)
    assert not common.needs_vision(300, 0.60, text_floor=100)


def test_only_the_image_dominant_page_is_flagged(extracted):
    """The pipeline's whole thesis, asserted against real pages."""
    flagged = []
    for path in common.find_pages_files(extracted):
        doc = common.load_pages(path)
        flagged += [(doc["source"], p["n"]) for p in doc["pages"] if p["needs_vision"]]
    assert flagged == [("figures.pdf", 1)], (
        "exactly the thin-caption page should be flagged; "
        "the explained figure and all text pages should not")


def test_empty_page_is_not_flagged(extracted):
    doc = common.load_pages(common.pages_path(extracted, "empty"))
    page = doc["pages"][0]
    assert page["chars"] == 0 and not page["needs_vision"]


def test_extraction_records_its_thresholds(extracted):
    doc = common.load_pages(common.pages_path(extracted, "figures"))
    assert doc["thresholds"] == {"image_frac": common.IMAGE_FRAC,
                                 "text_floor": common.TEXT_FLOOR}


def test_extract_is_deterministic(fixtures_dir, tmp_path):
    """Same input, same output. Extraction must never call a model."""
    def run(target):
        extract.main([fixtures_dir, "--out", str(target)])
        return {os.path.basename(p): [pg["blocks"] for pg in common.load_pages(p)["pages"]]
                for p in common.find_pages_files(str(target))}
    assert run(tmp_path / "a") == run(tmp_path / "b")


def test_source_hash_detects_a_changed_pdf(extracted, fixtures_dir):
    doc = common.load_pages(common.pages_path(extracted, "headings"))
    assert doc["source_sha256"] == common.sha256_file(f"{fixtures_dir}/headings.pdf")


# --- 4. enrichment: only flagged pages, and resumable ---------------------

class FakeModel:
    """Stands in for the HTTP call, and counts which pages were sent."""

    def __init__(self, fail_on=(), raise_after=None):
        self.seen = []
        self.fail_on = set(fail_on)
        self.raise_after = raise_after

    def __call__(self, png, model, endpoint, timeout=300, num_predict=1500):
        self.seen.append(len(self.seen) + 1)
        if self.raise_after is not None and len(self.seen) > self.raise_after:
            raise KeyboardInterrupt()
        if len(self.seen) in self.fail_on:
            raise RuntimeError("synthetic transcription failure")
        return "# Rendered\n\nA description naming Button A and value 42."


@pytest.fixture
def enrich_env(fixtures_dir, tmp_path, monkeypatch):
    out = tmp_path / "out"
    extract.main([fixtures_dir, "--out", str(out)])
    monkeypatch.setattr(enrich, "render_page_png", lambda *a, **k: b"\x89PNG-stub")
    return str(out)


def test_only_flagged_pages_reach_the_model(enrich_env, fixtures_dir, monkeypatch):
    """Assert the thesis: unflagged pages must never cost a model call."""
    fake = FakeModel()
    monkeypatch.setattr(enrich, "describe", fake)
    assert enrich.main([enrich_env, "--model", "stub", "--corpus", fixtures_dir]) == 0

    total_pages = sum(common.load_pages(p)["page_count"]
                      for p in common.find_pages_files(enrich_env))
    assert total_pages == 8
    assert len(fake.seen) == 1, "exactly one page was flagged; only it may be sent"

    doc = common.load_pages(common.pages_path(enrich_env, "figures"))
    assert doc["pages"][0]["vision"]["text"].startswith("# Rendered")
    assert doc["pages"][1]["vision"] is None, "an unflagged page must stay untouched"


def test_interrupt_then_restart_does_not_redo_work(enrich_env, fixtures_dir, monkeypatch):
    first = FakeModel()
    monkeypatch.setattr(enrich, "describe", first)
    enrich.main([enrich_env, "--model", "stub", "--corpus", fixtures_dir])
    assert len(first.seen) == 1

    second = FakeModel()
    monkeypatch.setattr(enrich, "describe", second)
    enrich.main([enrich_env, "--model", "stub", "--corpus", fixtures_dir])
    assert second.seen == [], "a completed page must not be sent twice"

    third = FakeModel()
    monkeypatch.setattr(enrich, "describe", third)
    enrich.main([enrich_env, "--model", "stub", "--corpus", fixtures_dir, "--force"])
    assert len(third.seen) == 1, "--force must override the resumption log"


def test_extraction_survives_enrichment(enrich_env, fixtures_dir, monkeypatch):
    """enrich.py writes `vision` and nothing else -- the invariant that makes
    re-running it safe."""
    before = common.load_pages(common.pages_path(enrich_env, "figures"))
    monkeypatch.setattr(enrich, "describe", FakeModel())
    enrich.main([enrich_env, "--model", "stub", "--corpus", fixtures_dir])
    after = common.load_pages(common.pages_path(enrich_env, "figures"))

    for key in ("source", "source_sha256", "page_count", "extracted_at"):
        assert before[key] == after[key]
    for old, new in zip(before["pages"], after["pages"]):
        assert old["blocks"] == new["blocks"]
        assert (old["chars"], old["needs_vision"]) == (new["chars"], new["needs_vision"])


def test_a_failing_page_is_recorded_not_dropped(enrich_env, fixtures_dir, monkeypatch):
    monkeypatch.setattr(enrich, "describe", FakeModel(fail_on=(1,)))
    enrich.main([enrich_env, "--model", "stub", "--corpus", fixtures_dir])
    log = common.enrich_log_path(enrich_env, "figures")
    records = [json.loads(line) for line in open(log) if line.strip()]
    assert len(records) == 1
    assert records[0]["ok"] is False and "synthetic" in records[0]["error"]


def test_unreachable_endpoint_fails_with_an_actionable_message(enrich_env, fixtures_dir, capsys):
    rc = enrich.main([enrich_env, "--model", "stub", "--corpus", fixtures_dir,
                      "--endpoint", "http://127.0.0.1:1"])
    assert rc == 2
    assert "cannot reach the model" in capsys.readouterr().err


def test_empty_content_falls_back_to_the_reasoning_field(monkeypatch):
    """Some runtimes put the substance in `thinking`; paying for it and then
    discarding it is the trap this guards."""
    class Response:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self):
            return json.dumps({"message": {"content": "  ",
                                           "thinking": "# Real output"}}).encode()
    monkeypatch.setattr(enrich.urllib.request, "urlopen", lambda *a, **k: Response())
    assert enrich.describe(b"x", "stub", "http://x") == "# Real output"


def test_request_disables_reasoning(monkeypatch):
    """Non-negotiable default: it halves per-page cost for no quality loss."""
    captured = {}

    class Response:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps({"message": {"content": "ok"}}).encode()

    def fake_urlopen(request, **kwargs):
        captured.update(json.loads(request.data))
        return Response()

    monkeypatch.setattr(enrich.urllib.request, "urlopen", fake_urlopen)
    enrich.describe(b"x", "stub", "http://x")
    assert captured["think"] is False
    assert captured["options"]["temperature"] == 0
    assert captured["options"]["num_predict"] >= 1500


# --- 5. assembly and anchors ---------------------------------------------

@pytest.fixture
def assembled(extracted, tmp_path):
    md = tmp_path / "md"
    assert assemble.main([extracted, "--md", str(md)]) == 0
    return str(md)


def test_every_page_is_represented(assembled, extracted):
    text = open(os.path.join(assembled, "figures.md")).read()
    assert "{#p001}" in text and "{#p002}" in text


def test_anchors_are_stable_across_runs(extracted, tmp_path):
    first = tmp_path / "one"
    second = tmp_path / "two"
    assemble.main([extracted, "--md", str(first)])
    assemble.main([extracted, "--md", str(second)])
    for name in os.listdir(str(first)):
        a = open(os.path.join(str(first), name)).read()
        b = open(os.path.join(str(second), name)).read()
        assert a == b, f"{name} is not reproducible"


def test_a_page_with_no_heading_still_gets_an_anchor(assembled):
    text = open(os.path.join(assembled, "empty.md")).read()
    assert '<a id="p001">' in text


def test_content_does_not_leak_across_page_anchors(assembled):
    text = open(os.path.join(assembled, "figures.md")).read()
    first = text.index("{#p001}")
    second = text.index("{#p002}")
    assert text.index("A short caption.") < second
    assert text.index("Triage must leave this page alone") > second
    assert first < second


def test_generated_text_is_marked_and_quarantined(extracted, tmp_path):
    """Ground truth and a model's reading must never be blended."""
    doc = common.load_pages(common.pages_path(extracted, "figures"))
    doc["pages"][0]["vision"] = {"text": "A diagram labelled Alpha."}
    path = tmp_path / "figures.pages.json"
    common.save_pages(str(path), doc)
    md = assemble.render_document(doc)

    assert "> **Figure (p001):**" in md
    line = next(l for l in md.split("\n") if "Alpha" in l)
    assert line.startswith(">"), "generated text escaped its blockquote"


def test_code_fences_survive_assembly(extracted):
    doc = common.load_pages(common.pages_path(extracted, "two-column"))
    md = assemble.render_document(doc)
    assert md.count("```") == 2
    body = md.split("```")[1]
    assert "    for key in sorted(options):" in body


def test_angle_brackets_are_escaped_outside_code():
    doc = {"source": "x.pdf", "page_count": 1, "pages": [
        {"n": 1, "blocks": [{"kind": "para", "text": "use <Thing> here"},
                            {"kind": "code", "text": "if a < b:", "lang": None}]}]}
    md = assemble.render_document(doc)
    assert "&lt;Thing&gt;" in md
    assert "if a < b:" in md, "code fences must not be escaped"


# --- 6. retrieval ---------------------------------------------------------

def test_a_known_term_retrieves_its_page(assembled, tmp_path):
    idx = index_stage.build(assembled)
    hits = index_stage.search(idx, "vertical extent cluster", k=3)
    assert hits, "no hits at all"
    assert hits[0]["file"] == "two-column"


def test_outline_records_every_heading_with_its_anchor(assembled):
    idx = index_stage.build(assembled)
    entries = {(e["file"], e["heading"]): e for e in idx["outline"]}
    entry = entries[("headings", "A Major Section")]
    assert entry["anchor"] == "p001" and entry["level"] == 2
    assert not any("{#" in e["heading"] for e in idx["outline"]), "anchor leaked into the text"


def test_search_is_scoped_to_the_page_not_the_file(assembled):
    idx = index_stage.build(assembled)
    hits = index_stage.search(idx, "short caption", k=1)
    assert (hits[0]["file"], hits[0]["anchor"]) == ("figures", "p001")


def test_index_round_trips_through_json(assembled, tmp_path):
    out = tmp_path / "index.json"
    assert index_stage.main([assembled, "--out", str(out)]) == 0
    loaded = index_stage.load(str(out))
    assert index_stage.search(loaded, "configure options", k=1)[0]["file"] == "two-column"


def test_long_row_is_not_a_heading_whatever_its_size(fixtures_dir, tmp_path):
    """A sentence set large is still a sentence."""
    from pdf2md_hybrid import layout
    long_line = "x" * (layout.HEADING_MAX_CHARS + 20)
    assert len(long_line) > layout.HEADING_MAX_CHARS


def test_invisible_text_does_not_suppress_triage(tmp_path):
    """The failure this guards: junk text pushes a page past the text floor, so
    an image-dominant page is never flagged and its content is lost."""
    real_chars, junk_chars = 30, 697
    assert common.needs_vision(real_chars, 0.53), "should flag on its own"
    assert not common.needs_vision(real_chars + junk_chars, 0.53), (
        "this is exactly what invisible text used to cause")


def test_reextracting_restores_lost_descriptions(enrich_env, fixtures_dir, monkeypatch):
    """extract --force rewrites the pages file and drops every `vision` field.

    Trusting the log alone there loses the descriptions for good: recorded as
    done, present nowhere. They have to be regenerated.
    """
    monkeypatch.setattr(enrich, "describe", FakeModel())
    enrich.main([enrich_env, "--model", "stub", "--corpus", fixtures_dir])
    assert common.load_pages(common.pages_path(enrich_env, "figures"))["pages"][0]["vision"]

    extract.main([fixtures_dir, "--out", enrich_env, "--force"])
    assert common.load_pages(common.pages_path(enrich_env, "figures"))["pages"][0]["vision"] is None

    again = FakeModel()
    monkeypatch.setattr(enrich, "describe", again)
    enrich.main([enrich_env, "--model", "stub", "--corpus", fixtures_dir])
    assert len(again.seen) == 1, "the lost description was never regenerated"
    assert common.load_pages(common.pages_path(enrich_env, "figures"))["pages"][0]["vision"]


def test_a_reliably_failing_page_does_not_retry_forever(enrich_env, fixtures_dir, monkeypatch):
    monkeypatch.setattr(enrich, "describe", FakeModel(fail_on=(1,)))
    enrich.main([enrich_env, "--model", "stub", "--corpus", fixtures_dir])
    second = FakeModel()
    monkeypatch.setattr(enrich, "describe", second)
    enrich.main([enrich_env, "--model", "stub", "--corpus", fixtures_dir])
    assert second.seen == [], "a logged failure must stay settled without --force"


def test_a_fence_inside_code_does_not_break_the_block():
    """Extracted text can contain anything. A short fence would end the block
    early and spill the rest of the listing into the document as prose."""
    doc = {"source": "x.pdf", "page_count": 1, "pages": [
        {"n": 1, "blocks": [{"kind": "code", "lang": None,
                             "text": "print(1)\n```\nstill code"}]}]}
    md = assemble.render_document(doc)
    body = md[md.index("<a id="):]
    opening = body.split("\n")[2]
    assert len(opening) >= 4 and set(opening) == {"`"}, "fence must outrun the content"
    assert body.count(opening) == 2, "exactly one opening and one closing fence"
    assert "still code" in body


def test_index_files_flag_scopes_the_build(assembled, tmp_path):
    """--files on index must mean the same thing it means everywhere else."""
    out = tmp_path / "scoped.json"
    assert index_stage.main([assembled, "--out", str(out), "--files", "two-column"]) == 0
    idx = index_stage.load(str(out))
    assert {d["file"] for d in idx["docs"]} == {"two-column"}


# --- the package's public API, as a downstream project uses it ------------

def test_stage_modules_are_not_shadowed_by_api_functions():
    """A public function named after a submodule makes that submodule
    unreachable, and the break surfaces only when something calls it."""
    import pdf2md_hybrid

    for name in ("probe", "extract", "enrich", "assemble", "index", "pipeline"):
        attr = getattr(pdf2md_hybrid, name, None)
        if attr is not None:
            assert hasattr(attr, "main"), f"pdf2md_hybrid.{name} is not the stage module"
    assert callable(pdf2md_hybrid.is_text_native)


def test_convert_returns_a_readable_corpus(fixtures_dir, tmp_path):
    """The whole downstream contract in one call."""
    import pdf2md_hybrid

    corpus = pdf2md_hybrid.convert(fixtures_dir, str(tmp_path / "out"))
    assert corpus.stats()["pages"] > 0
    assert "two-column" in corpus.sources()

    hits = corpus.search("vertical extent cluster", k=1)
    assert hits and hits[0].file == "two-column"
    assert hits[0].cite == f"two-column#{hits[0].anchor}"
    assert "column" in hits[0].text()
    assert hits[0].cite in corpus.context("vertical extent cluster", k=1)


def test_corpus_reads_without_the_source_pdfs(fixtures_dir, tmp_path):
    """The read side must work where the PDFs never were -- that is what lets a
    downstream service be a different machine from the one that extracted."""
    import shutil
    from pdf2md_hybrid import Corpus, convert

    convert(fixtures_dir, str(tmp_path / "out"))
    detached = tmp_path / "detached"
    shutil.copytree(tmp_path / "out", detached)

    corpus = Corpus.open(str(detached))
    assert corpus.search("configure options", k=1)[0].file == "two-column"
    assert corpus.snapshot()["tool"]["name"] == "PDF2MD-hybrid"


def test_corpus_builds_an_index_when_none_was_written(fixtures_dir, tmp_path):
    from pdf2md_hybrid import Corpus, convert

    out = tmp_path / "out"
    convert(fixtures_dir, str(out))
    os.remove(out / "index.json")
    assert Corpus.open(str(out)).stats()["pages"] > 0


def test_page_is_addressable_by_its_anchor(fixtures_dir, tmp_path):
    from pdf2md_hybrid import convert

    corpus = convert(fixtures_dir, str(tmp_path / "out"))
    page = corpus.page("figures", "p001")
    assert "A short caption" in page
    assert "Triage must leave this page alone" not in page, "page p002 leaked in"


def test_every_stage_main_accepts_argv(fixtures_dir, tmp_path, capsys):
    """Each stage is callable in-process, not only from a shell.

    `probe` was the odd one out: its `main()` read sys.argv directly, so the
    public `probe()` wrapper raised TypeError the moment it was given a path.
    An import check cannot see that; only calling it can.
    """
    import inspect

    from pdf2md_hybrid import assemble, enrich, extract, index as index_stage
    from pdf2md_hybrid import pipeline, probe as probe_stage

    for module in (probe_stage, extract, enrich, assemble, index_stage, pipeline):
        params = inspect.signature(module.main).parameters
        assert "argv" in params, f"{module.__name__}.main takes no argv"
        assert params["argv"].default is None

    import pdf2md_hybrid
    assert pdf2md_hybrid.is_text_native(fixtures_dir) is True
    assert "VERDICT" in capsys.readouterr().out


def test_pipeline_reaches_the_enrich_stage(fixtures_dir, tmp_path, monkeypatch):
    """The orchestrator's enrich import is deferred, so only the --model path
    exercises it. Nothing did, and a plain `import enrich` shipped in v0.1.0 --
    `pdf2md --model` crashed outright while every stage worked on its own."""
    from pdf2md_hybrid import enrich, pipeline

    monkeypatch.setattr(enrich, "render_page_png", lambda *a, **k: b"\x89PNG-stub")
    monkeypatch.setattr(enrich, "describe", FakeModel())

    rc = pipeline.main([fixtures_dir, "--out", str(tmp_path / "out"),
                        "--model", "stub", "--no-snapshot"])
    assert rc == 0
    doc = common.load_pages(common.pages_path(str(tmp_path / "out"), "figures"))
    assert doc["pages"][0]["vision"]["text"].startswith("# Rendered")
