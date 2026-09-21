#!/usr/bin/env python3
"""Stage 4: describe only the pages triage flagged. The one slow stage.

Everything else in this pipeline is close to free; this is where the time goes,
so the only thing that matters here is that it runs on as few pages as possible
and that an interrupt costs one page rather than the whole run.

The model is reached over a local HTTP endpoint (Ollama by default). Nothing is
sent anywhere else, and no hosted API is ever contacted.

Usage:
    python enrich.py out/ --model MODEL [--files a,b] [--endpoint URL] [--force]
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

import common

DEFAULT_ENDPOINT = "http://localhost:11434"

# Rasterization resolution. 110 DPI was adequate for legible UI text on the one
# corpus this was measured against; higher costs time for little gain. Pages
# with denser small text may need more -- that curve has not been measured.
DEFAULT_DPI = 110

# Edit this freely. "Naming the exact labels and values visible" is the clause
# that turns a useless gloss ("a screenshot of a settings page") into a fact a
# retrieval query can actually match. Keep it.
PROMPT = """Transcribe this page into clean Markdown. Use a # heading for the title.
Describe any screenshot, diagram, or figure in one short paragraph, naming
the exact labels and values visible. Output only Markdown."""


class ModelError(RuntimeError):
    pass


def render_page_png(pdf_path: str, page_number: int, dpi: int = DEFAULT_DPI) -> bytes:
    """Rasterize one 1-indexed page to PNG bytes."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(pdf_path)
    try:
        page = pdf[page_number - 1]
        image = page.render(scale=dpi / 72.0).to_pil()
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()
    finally:
        pdf.close()


def describe(png: bytes, model: str, endpoint: str, timeout: int = 300,
             num_predict: int = 1500) -> str:
    """POST one image to the chat endpoint and return the Markdown it produces."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT,
                      "images": [base64.b64encode(png).decode("ascii")]}],
        "stream": False,
        # Transcription is a perception task, not a reasoning one. Reasoning
        # roughly doubled per-page cost for output that was not better.
        "think": False,
        "options": {"temperature": 0, "num_predict": num_predict},
    }
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise ModelError(
            f"cannot reach the model at {endpoint}: {exc}. "
            "Is the runtime running, and is --model a model it has pulled?") from exc
    except json.JSONDecodeError as exc:
        raise ModelError(f"the endpoint returned something that is not JSON: {exc}") from exc

    if isinstance(body, dict) and body.get("error"):
        raise ModelError(str(body["error"]))
    message = (body or {}).get("message") or {}
    text = (message.get("content") or "").strip()
    if not text:
        # Some runtimes put the substance in a separate reasoning field. Falling
        # back beats recording an empty result for tokens already paid for.
        text = (message.get("thinking") or message.get("reasoning") or "").strip()
    return text


def _completed(log_path: str) -> Dict[int, Dict[str, Any]]:
    """Pages already attempted, from the resumability log."""
    done: Dict[int, Dict[str, Any]] = {}
    if not os.path.exists(log_path):
        return done
    with open(log_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn final line from a hard kill
            if isinstance(rec.get("page"), int):
                done[rec["page"]] = rec
    return done


def _append_log(log_path: str, record: Dict[str, Any]) -> None:
    """Record the attempt *before* the pages file is updated, and flush it.

    The log is the resumption point, so it has to hit the disk first. If the
    process dies between the two writes, the page is merely redone.
    """
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _eta(seconds_each: List[float], remaining: int) -> str:
    if not seconds_each or not remaining:
        return ""
    avg = sum(seconds_each) / len(seconds_each)
    total = int(avg * remaining)
    return f"  eta {total // 60}m{total % 60:02d}s"


def enrich_file(pages_file: str, corpus_dir: str, model: str, endpoint: str,
                dpi: int, force: bool, timeout: int, num_predict: int) -> Dict[str, int]:
    doc = common.load_pages(pages_file)
    out_dir = os.path.dirname(pages_file) or "."
    name = os.path.basename(pages_file)[:-len(".pages.json")]
    log_path = common.enrich_log_path(out_dir, name)

    pdf_path = os.path.join(corpus_dir, doc["source"])
    if not os.path.exists(pdf_path):
        raise ModelError(f"source PDF not found: {pdf_path} (pass --corpus)")

    done = {} if force else _completed(log_path)
    if force and os.path.exists(log_path):
        os.remove(log_path)

    targets = [p for p in doc["pages"] if p.get("needs_vision")]

    def settled(page: Dict[str, Any]) -> bool:
        """Has this page already been dealt with?

        The log records that a page was attempted; the pages file holds what
        came back. Both have to agree. Re-extracting rewrites the pages file and
        drops every `vision` field, and trusting the log alone there would leave
        the descriptions permanently gone -- recorded as done, present nowhere.

        A page logged as failed stays settled, so a page that reliably fails
        does not retry forever. --force overrides both.
        """
        record = done.get(page["n"])
        if record is None:
            return False
        if not record.get("ok"):
            return True
        return bool((page.get("vision") or {}).get("text"))

    todo = [p for p in targets if not settled(p)]
    stats = {"flagged": len(targets), "done": len(targets) - len(todo), "ok": 0, "failed": 0}
    if not todo:
        print(f"  {name:<28} {len(targets)} flagged, all already enriched")
        return stats

    print(f"  {name:<28} {len(todo)} of {len(targets)} flagged pages to enrich")
    timings: List[float] = []
    for i, page in enumerate(todo, start=1):
        t0 = time.time()
        try:
            png = render_page_png(pdf_path, page["n"], dpi)
            text = describe(png, model, endpoint, timeout, num_predict)
            elapsed = time.time() - t0
            ok = bool(text)
            record = {"page": page["n"], "ok": ok, "chars": len(text),
                      "seconds": round(elapsed, 1)}
            if not ok:
                record["error"] = "model returned empty content"
        except ModelError:
            raise  # an endpoint that is down is not a per-page problem
        except Exception as exc:
            elapsed = time.time() - t0
            text, ok = "", False
            # A page that reliably fails is diagnostic information, not
            # something to drop silently.
            record = {"page": page["n"], "ok": False, "chars": 0,
                      "seconds": round(elapsed, 1),
                      "error": f"{type(exc).__name__}: {exc}"}

        _append_log(log_path, record)
        if ok:
            page["vision"] = {"text": text, "model": model, "dpi": dpi,
                              "at": common.utc_now()}
            stats["ok"] += 1
        else:
            stats["failed"] += 1
        common.save_pages(pages_file, doc)

        timings.append(elapsed)
        flag = "" if ok else "  FAILED"
        print(f"    p{page['n']:<5} {elapsed:>5.1f}s  {len(text):>5} chars  "
              f"[{i}/{len(todo)}]{_eta(timings, len(todo) - i)}{flag}")
    return stats


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", help="directory holding .pages.json artifacts")
    ap.add_argument("--model", required=True, help="vision model name as the runtime knows it")
    ap.add_argument("--corpus", help="where the source PDFs live (default: alongside out_dir's parent)")
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT,
                    help=f"local model endpoint (default {DEFAULT_ENDPOINT})")
    ap.add_argument("--files", help="comma-separated stems to process, instead of all")
    ap.add_argument("--dpi", type=int, default=DEFAULT_DPI,
                    help=f"rasterization resolution (default {DEFAULT_DPI})")
    ap.add_argument("--timeout", type=int, default=300, help="per-request timeout in seconds")
    ap.add_argument("--num-predict", type=int, default=1500,
                    help="output token ceiling; too low truncates mid-description")
    ap.add_argument("--force", action="store_true", help="re-enrich pages already recorded as done")
    args = ap.parse_args(argv)

    pages_files = common.find_pages_files(args.out_dir, common.parse_files_flag(args.files))
    if not pages_files:
        print(f"No .pages.json artifacts in {args.out_dir}. Run extract.py first.", file=sys.stderr)
        return 1

    corpus = args.corpus or os.path.dirname(os.path.abspath(args.out_dir)) or "."
    started = time.time()
    totals = {"flagged": 0, "done": 0, "ok": 0, "failed": 0}
    try:
        for pages_file in pages_files:
            stats = enrich_file(pages_file, corpus, args.model, args.endpoint,
                                args.dpi, args.force, args.timeout, args.num_predict)
            for key in totals:
                totals[key] += stats[key]
    except ModelError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted. Re-run the same command to resume; "
              "completed pages will be skipped.", file=sys.stderr)
        return 130

    elapsed = time.time() - started
    print(f"\n{totals['ok']} pages enriched, {totals['done']} already done, "
          f"{totals['failed']} failed, in {elapsed / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
