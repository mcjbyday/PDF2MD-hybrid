# PDF2MD-hybrid

**Turn born-digital PDFs into grounded Markdown with a local vision model — without sending every page through it.**

Most "PDF to Markdown with a local LLM" tools rasterize every page and ask a vision model to transcribe it. That is the right design for *scanned* documents. For born-digital PDFs — slide decks, reports, manuals exported from design tools — this doesn't work due to issues with processing speed and hallucinations.

This pipeline triages first. It parses every page deterministically, then sends a model only the minority of pages where a picture carries meaning the text layer does not.

---

## Is this for you?

**Yes, if:**

- Your PDFs are **born-digital** (exported, not scanned) but **visually rich** — slides, diagrams, UI screenshots, annotated figures.
- You want **Markdown for retrieval or grounding** with some headings, fenced code, stable per-page anchors.
- You want it **fully local**. Without any cloud API, per-token cost, and without any documents leaving the machine.
- Your corpus is big enough that transcribing every page is an overnight job you would rather not run.

**No, if:**

| Situation | Use instead |
|---|---|
| Your PDFs are **scanned images** | `ocrmypdf`, `tesseract`, or an all-vision tool |
| You need **tables as tables** — real cells, spans and headers | [Docling](https://github.com/DS4SD/docling) or [Marker](https://github.com/VikParuchuri/marker) |
| You just need **raw text**, no structure | `pdftotext -layout` command |
| You have **a handful of documents** | The triage only pays for itself at scale |

`probe.py` tells you which bucket you are in and this can be executed first; `--json report.json` writes a full per-file breakdown for scripting. An exit code of `0` constitues a fit and `2` suggests a corpus that appears as scanned.

---

## Install

Python 3.9 or newer.

```bash
pip install -e .
```

This project has two dependencies: [`pdfplumber`](https://github.com/jsvine/pdfplumber) (MIT) for text, fonts and positions, and [`pypdfium2`](https://github.com/pypdfium2-team/pypdfium2) (Apache-2.0/BSD-3) for rasterization. Other functions are included in the standard library. This project does not require a system binary be installed.

Enrichment additionally requires a local model runtime — [Ollama](https://ollama.com) by default — along with a vision model.

---

## Quickstart

```bash
# 1. Does this pipeline have relevance to your corpus?
python probe.py /path/to/pdfs

# 2. Run everything.
python pipeline.py /path/to/pdfs --out out/ --model <your-vision-model>

# 3. Search what came out.
python index.py out/md --out out/index.json --query "your question"
```

### `out/`

| File | What it is |
|---|---|
| `<stem>.pages.json` | The details extracted from a PDF representing the stage-to-stage contract. |
| `<stem>.enrich.jsonl` | One line per page sent to the model. This is what makes a run resumable. |
| `md/<stem>.md` | The result. |
| `index.json` | An outline plus BM25 ranking function, for retrieval. |
| `pipeline.snapshot.json` | Exactly what produced this run. See [Tuning](#tuning). |

BM25 (Best Matching 25) is a ranking function used to score documents based on query term frequency and rarity.

Omit `--model` and the vision stage is skipped. Extraction alone can cover a majority of pages.

The probe's exit code is scriptable — `0` means text-native and worth proceeding, `2` means it looks scanned and you should reach for OCR instead. `--json report.json` writes a full per-file breakdown.

---

## The core idea

A page needs a vision model only when **a picture is carrying meaning the text layer does not**.

```
vision candidate  =  largest image covers > 18% of the page
                     AND the page's text layer holds < 400 characters
```

A page with a big diagram *and* a full text explanation is already well served by extraction. A page that is mostly screenshot with a two-line caption is not.

**Both thresholds are configurable judgment calls.** They are corpus-dependent, they are exposed as flags on every stage (`--image-frac`, `--text-floor`), and you should expect to re-tune them. Treat the defaults as a starting point and check what `probe.py` reports against your own files.

**Recurrence, rather than width defines a column.** A word space falls at an arbitrary position once; a column boundary falls at the same position on row after row. Detecting columns by clustering line left edges — and requiring several rows to agree before believing one — reads layouts whose gutters are narrower than a single character, which searching for vertical whitespace cannot do at all.

That matters most on the dense "concept grid" slides that reference material is full of: five labelled cells across a page, each a term over the lines explaining it. Measured by whitespace those pages look single-column, and they come out as three columns of prose interleaved into gibberish. Measured by recurrence they come out grouped.

**Fonts often suggest code.** Monospace-ness is measured from advance-width uniformity — whether a face's glyphs all occupy the same horizontal space — rather than by matching font names. That separation is deterministic, does not require a model, and takes a source wall of text into a fenced code block.

---

## How it works

```
  PDFs
   │
   ├─ 1. PROBE      is this corpus a fit?                     (seconds)
   │
   ├─ 2. EXTRACT    all pages: text, fonts, layout                 (fast, no model)
   │                  column-aware reading order
   │                  font-size clustering → headings
   │                  monospace detection → fenced code
   │
   ├─ 3. TRIAGE     flag pages where an image carries meaning      (instant)
   │
   ├─ 4. ENRICH     ONLY flagged pages → local vision model        (the slow stage)
   │                  resumable at page granularity
   │
   ├─ 5. ASSEMBLE   one .md per source PDF, stable page anchors    (instant)
   │
   └─ 6. INDEX      outline + BM25 over page chunks                (seconds)
```


Each stage is independently runnable and writes its own artifact, so a defective page can always be traced back to the stage that produced it. Extraction does not call a model.

---

## Result

One Markdown file for one source PDF is generated with stable anchors:

```markdown
---
source: manual-a.pdf
pages: 196
generated: 2026-09-21
---

## Configuration Options {#p030}

Several standard components are available...

​```
public class Example {
    // recovered as code via monospace detection
}
​```

> **Figure (p031):** generated description of this page's imagery.
>
> A screenshot of the deployment settings panel, naming the exact
> labels and values visible.
```

Where a page is laid out as a grid of labelled cells, each label stays attached to its own description rather than being merged into running prose:

```markdown
**TRY BLOCK**

Code for business logic is run within a try block.

**FINALLY BLOCK**

Code in the finally block is run whether an exception has been thrown or not.
```

This is column *grouping*, not grid reconstruction — see [Known limitations](#known-limitations) for the distinction and the rationale.

**Generated text is always blockquoted and labelled.** One is derived from the source while the other is an interpretation by the  model of a picture. A consumer must be able to tell which is which.

---

## Resumability

Enrichment is a slow stage, and it is resumable at page granularity. Each completed page appends a line to `out/<stem>.enrich.jsonl` before the pages file is updated, so an interrupt costs one page rather than the run. Re-running skips completed work; `--force` redoes it.

Failures are recorded as lines too, with an error. 

Files are independent, so you can work through a corpus in batches:

```bash
python extract.py corpus/ --files doc-a,doc-b --out out/
python enrich.py out/ --files doc-a,doc-b --model <your-vision-model> --corpus corpus/
```

---

## Sizing your output

**Do not try to preload a corpus into a context window.** Markdown is small in bytes and large in tokens.

Long contexts also get *slower*: quadratic attention cost emerges prior to context window saturation. Likewise, a large preload will cost minutes before the first token of an answer.

The practical shape is a small always-loaded outline plus per-query retrieval of a few pages, which is what stage 6 produces. At around a megabyte of Markdown, BM25 over page-sized chunks answers in milliseconds and adds no dependency. 
---

## Troubleshooting

**`cannot reach the model at http://localhost:11434`** — the runtime is not running, or is on another port. Start it, or pass `--endpoint`. Extraction does not need it. Enrichment does.

**The model name is rejected** — `--model` must name a model the runtime has already pulled, spelled exactly as it lists it.

**A page came out scrambled** — extraction and enrichment are separate stages writing separate fields, so run `extract.py` alone on that one file and look at its `blocks` to see which stage produced the problem. Multi-column ordering can produce this; see [Known limitations](#known-limitations).

**Almost nothing was extracted** — run `probe.py`. An exit code of `2` means the corpus is scanned, not born-digital, and this is the wrong tool.

**Too many or too few pages flagged** — the thresholds are corpus-specific. Re-tune `--image-frac` and `--text-floor`, checking the counts `probe.py` reports.

## Known limitations

- **Tables are grouped, not reconstructed.** A grid's columns are kept apart and each label stays with its own description, which preserves relevance for a reader or a retrieval index. But no `| a | b |` table is emitted and no cell grid is recovered. Where text wraps inside a cell, which line belongs to which row remains ambiguous. 
- **Column ordering is silent and is not enforced.** A scrambled page is still valid Markdown of roughly the right length. Single-column pages are passed through untouched — that path is exact and it is the common case — and a page is only split when several rows independently agree on the same boundaries *and* occupy two columns at once. Spot-check multi-column output on your own corpus before trusting it at scale.
- **Grouping does not change retrieval ranking.** BM25 (Best Matching 25) is a probabilistic relevance ranking function used to estimate how well documents match a given query. It scores a whole page chunk, so a page ranks the same whether or not its columns were untangled. 
- **Code language is not labeled** for code blocks. Fences are emitted bare.
- **Figure descriptions are an output from the model.** They can be wrong, and occasionally a model will invent a detail. They ought to be informative for retrieval, but they are not presumed to be derived from the source.

---

## Tuning

| Flag | Stage | What it does |
|---|---|---|
| `--image-frac` | probe, extract | Page-area fraction that counts as a dominant image |
| `--text-floor` | probe, extract | Character count below which a page's text is "thin" |
| `--dpi` | enrich | Rasterization resolution; the default is adequate for legible UI text |
| `--endpoint` | enrich | Local model endpoint |
| `--files` | all | Process a named subset |
| `--force` | extract, enrich | Redo work already recorded as done |

Every run writes `out/pipeline.snapshot.json`: the flags, the model, the prompt, the dependency versions, the tool's own git revision, and the heuristic ratios of the run. 

```bash
python pipeline.py corpus/ --out out2/ --config out/pipeline.snapshot.json
```

replays a previous run's settings. Anything you type still wins, so a replay can be adjusted one flag at a time, and any drift from the snapshot — a changed prompt, a bumped library, an edited constant — is reported before the run starts rather than discovered afterwards.

Reasoning ("thinking") is **disabled** for transcription. It doubles per-page cost for output and did not demonstrate improvement.

---

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The suite generates its own synthetic PDFs, including single-column text, two columns of prose beside code, heading hierarchies, an image-dominant page and an empty one. The suite executes offline, and in the vision stage, its HTTP call is mocked.

---

## License

MIT — see [LICENSE](LICENSE).
