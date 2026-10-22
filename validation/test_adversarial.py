"""Pytest entry point for the adversarial validation cases (see cases.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cases import build_cases  # noqa: E402

CASES = build_cases()


def test_case_ids_are_unique():
    ids = [c.case_id for c in CASES]
    assert len(ids) == len(set(ids))


def test_every_case_documents_its_expectation():
    for case in CASES:
        assert case.kind in {"positive", "negative", "boundary", "diagnostic"}
        assert case.reason and case.description and case.expect
        if case.kind in {"positive", "negative"}:
            assert case.designated, case.case_id


@pytest.mark.parametrize("case", CASES, ids=[c.case_id for c in CASES])
def test_case(case):
    passed, details, _ = case.evaluate()
    failures = [msg for ok, msg in details if not ok]
    assert passed, f"{case.case_id} ({case.kind}): {case.description} -> {failures}"
