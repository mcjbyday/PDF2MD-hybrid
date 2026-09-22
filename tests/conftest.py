"""Fixture generation and import path setup for the suite."""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_fixtures  # noqa: E402


@pytest.fixture(scope="session")
def fixtures_dir(tmp_path_factory) -> str:
    """Build the synthetic corpus once per run, outside the repo."""
    out = tmp_path_factory.mktemp("fixtures")
    make_fixtures.build(str(out))
    return str(out)


@pytest.fixture(scope="session")
def extracted(fixtures_dir, tmp_path_factory) -> str:
    """Run extraction over the fixtures and return the artifact directory."""
    from pdf2md_hybrid import extract

    out = str(tmp_path_factory.mktemp("out"))
    assert extract.main([fixtures_dir, "--out", out]) == 0
    return out
