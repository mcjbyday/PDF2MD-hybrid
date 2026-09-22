"""PDF2MD-hybrid: born-digital PDFs to grounded Markdown, mostly without a model.

Two things a downstream project needs, and nothing else:

    from pdf2md_hybrid import convert, Corpus

    convert("pdfs/", "out/", model="your-vision-model")   # run the pipeline
    corpus = Corpus.open("out/")                          # consume the output

`is_text_native` answers the go/no-go question before you spend a run on it.

`convert` is the whole pipeline in one call. `Corpus` is the read side, and it
needs neither a model nor the source PDFs -- only the directory the run left
behind, so the service that answers questions can be separate from the machine
that did the extraction.

Every stage is also a command. `pdf2md` runs all of them; `pdf2md-probe`,
`pdf2md-extract`, `pdf2md-enrich`, `pdf2md-assemble` and `pdf2md-index` run one
each, and `python -m pdf2md_hybrid.<stage> --help` documents any of them.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .corpus import Corpus, Hit

__all__ = ["convert", "is_text_native", "Corpus", "Hit", "__version__"]

try:  # pragma: no cover - depends on install method
    from importlib.metadata import version as _version
    __version__ = _version("PDF2MD-hybrid")
except Exception:  # pragma: no cover
    __version__ = "unknown"


def convert(corpus_dir: str, out_dir: str = "out", model: Optional[str] = None,
            files: Optional[List[str]] = None, endpoint: Optional[str] = None,
            image_frac: Optional[float] = None, text_floor: Optional[int] = None,
            dpi: Optional[int] = None, force: bool = False) -> "Corpus":
    """Run the whole pipeline and return the result, opened for reading.

    Omitting `model` skips the vision stage, which is a legitimate way to run:
    extraction alone already covers the large majority of pages, and it needs
    no model runtime installed.

    Raises `RuntimeError` if a stage fails, so a caller is never handed a
    half-built corpus that looks complete.
    """
    from .pipeline import main as pipeline_main

    argv: List[str] = [corpus_dir, "--out", out_dir]
    if model:
        argv += ["--model", model]
    if files:
        argv += ["--files", ",".join(files)]
    if endpoint:
        argv += ["--endpoint", endpoint]
    if image_frac is not None:
        argv += ["--image-frac", str(image_frac)]
    if text_floor is not None:
        argv += ["--text-floor", str(text_floor)]
    if dpi is not None:
        argv += ["--dpi", str(dpi)]
    if force:
        argv += ["--force"]

    code = pipeline_main(argv)
    if code:
        raise RuntimeError(f"pipeline failed with exit code {code}")
    return Corpus.open(out_dir)


def is_text_native(*paths: str, image_frac: Optional[float] = None,
                   text_floor: Optional[int] = None) -> bool:
    """Is this corpus a fit? True for text-native, False if it looks scanned.

    Named for what the boolean means rather than for the stage that answers it,
    which also leaves `pdf2md_hybrid.probe` free to be the stage module, as it
    is for every other stage.

    Worth calling before `convert` in an automated flow: a scanned corpus
    produces almost nothing here, and failing loudly beats shipping empty
    Markdown downstream.
    """
    from .probe import main as probe_main

    argv = list(paths)
    if image_frac is not None:
        argv += ["--image-frac", str(image_frac)]
    if text_floor is not None:
        argv += ["--text-floor", str(text_floor)]
    return probe_main(argv) == 0
