"""Tests for the Streamlit-free UI theme helpers (escaping, theme config, meanings)."""

from __future__ import annotations

import tomllib
from pathlib import Path

from explain_fakes import INJECTION, injection_result, leaky_result
from splitguard import ui_theme as theme
from splitguard.ui_launcher import streamlit_args

EVIL = '<img src=x onerror="alert(1)"><script>alert(2)</script>'


def test_builders_escape_untrusted_text():
    class F:  # minimal finding-like object with a hostile column name
        affected_columns = [EVIL]
        rule_id = "TGT-001"
        finding_id = f"TGT-001:{EVIL}"

        class severity:  # noqa: N801
            value = "critical"

    outputs = [
        theme.columns_grid_html([F()]),
        theme.chips_html([(EVIL, "")], tooltips={EVIL: EVIL}),
        theme.split_bar_html([(EVIL, 0.5, "train"), ("test", 0.5, "test")], caption=EVIL),
        theme.terminal_header_html(EVIL, EVIL),
        theme.ref_chip_html(EVIL),
        theme.anchor_html(EVIL),
    ]
    for html in outputs:
        assert "<script" not in html and "<img" not in html
        assert 'onerror="' not in html


def test_anchor_ids_are_safe_and_stable():
    assert theme.anchor_id("TGT-001:churn_score_final") == "f-tgt-001-churn-score-final"
    assert theme.anchor_id(EVIL).startswith("f-")
    assert set(theme.anchor_id(EVIL)) <= set("abcdefghijklmnopqrstuvwxyz0123456789-f")


def test_columns_grid_links_to_most_severe_finding_and_escapes_injection():
    result = injection_result()
    html = theme.columns_grid_html(result.findings)
    assert INJECTION not in html  # escaped, not raw
    assert theme.esc(INJECTION) in html
    leaky = leaky_result()
    grid = theme.columns_grid_html(leaky.findings)
    target = next(f for f in leaky.findings if "leak" in f.affected_columns)
    assert f'href="#{theme.anchor_id(target.finding_id)}"' in grid


def test_overview_ring_shows_coverage_not_a_safety_score():
    html = theme.overview_html({"critical": 2, "low": 1}, checks_run=21, checks_not_run=5)
    assert "81%" in html and "CHECKS RAN" in html
    assert "most severe: critical" in html
    assert "safe" not in html.lower() and "score" not in html.lower()
    empty = theme.overview_html({}, checks_run=10, checks_not_run=0)
    assert "no findings" in empty and "100%" in empty


def test_confidence_glyphs_are_distinct_from_severity():
    assert [theme.confidence_html(c) for c in ("high", "medium", "low")] == ["▮▮▮", "▮▮▯", "▮▯▯"]


def test_launcher_and_repo_config_use_the_same_theme():
    args = streamlit_args("localhost", 8501, headless=True)
    pairs = dict(zip(args[::2], args[1::2], strict=False)) | dict(
        zip(args[1::2], args[2::2], strict=False)
    )
    for name, value in theme.THEME_OPTIONS.items():
        assert pairs[f"--theme.{name}"] == value
    config = Path(__file__).resolve().parents[1] / ".streamlit" / "config.toml"
    if config.exists():  # absent in an installed wheel; present in the repository
        data = tomllib.loads(config.read_text(encoding="utf-8"))
        assert data["theme"] == theme.THEME_OPTIONS
        assert data["server"]["address"] == "localhost"
        assert data["browser"]["gatherUsageStats"] is False


def test_css_fetches_nothing_from_the_network():
    assert "http://" not in theme.CSS and "https://" not in theme.CSS
    assert "@import" not in theme.CSS and "url(" not in theme.CSS
