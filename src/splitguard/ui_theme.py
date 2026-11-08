"""Visual theme for the SplitGuard UI: palette, CSS and small HTML builders.

Streamlit-free so it can be unit-tested. Every builder escapes the text it is
given: column names, titles and finding IDs come from user data and must never
be interpreted as HTML. No fonts or assets are fetched from the network; the
font stacks below only use fonts already installed on the machine.
"""

from __future__ import annotations

import html
import re
from collections.abc import Iterable, Mapping, Sequence

# ------------------------------------------------------------------------ palette

BG = "#0B0F19"
SURFACE = "#111827"
SURFACE_2 = "#151D2E"
BORDER = "#222C40"
TEXT = "#E6EDF7"
MUTED = "#8B98AE"
ACCENT = "#22D3EE"  # neon cyan: brand and interactive elements
ACCENT_2 = "#A78BFA"  # violet: the optional local-model panel

SEVERITY_COLORS: dict[str, str] = {
    "critical": "#FF4D6D",
    "high": "#FF9F43",
    "medium": "#FACC15",
    "low": "#818CF8",
    "info": "#94A3B8",
}
SEVERITY_ORDER = ("critical", "high", "medium", "low", "info")
CONFIDENCE_BARS = {"high": 3, "medium": 2, "low": 1}

# Passed by the splitguard-ui launcher and mirrored in .streamlit/config.toml.
THEME_OPTIONS: dict[str, str] = {
    "base": "dark",
    "primaryColor": ACCENT,
    "backgroundColor": BG,
    "secondaryBackgroundColor": SURFACE,
    "textColor": TEXT,
    "borderColor": BORDER,
}

SANS = "Inter, 'Segoe UI Variable Text', 'Segoe UI', system-ui, -apple-system, sans-serif"
MONO = "'JetBrains Mono', 'Cascadia Code', 'Cascadia Mono', Consolas, ui-monospace, monospace"


def esc(value: object) -> str:
    """HTML-escape untrusted text (quotes included)."""
    return html.escape(str(value), quote=True)


def anchor_id(finding_id: str) -> str:
    """A stable, HTML-safe anchor for a finding card."""
    slug = re.sub(r"[^a-z0-9]+", "-", finding_id.lower()).strip("-")[:60]
    return f"f-{slug or 'finding'}"


# --------------------------------------------------------------------------- CSS

# Built separately: backslashes inside f-string expressions need Python 3.12+.
_SEVERITY_RAILS = "".join(
    f'[class*="st-key-sev-{s}-"] {{ border-left: 3px solid {c} !important; }}\n'
    f'[class*="st-key-sev-{s}-"]:hover {{ border-color: {c}; '
    f"box-shadow: 0 0 0 1px {c}55, 0 0 26px -10px {c}; }}\n"
    for s, c in SEVERITY_COLORS.items()
)

CSS = f"""
<style>
:root {{
  --sg-bg: {BG}; --sg-surface: {SURFACE}; --sg-surface-2: {SURFACE_2};
  --sg-border: {BORDER}; --sg-text: {TEXT}; --sg-muted: {MUTED};
  --sg-accent: {ACCENT}; --sg-accent-2: {ACCENT_2};
  --sg-mono: {MONO}; --sg-sans: {SANS};
}}
html, body, [data-testid="stAppViewContainer"] {{ font-family: var(--sg-sans); }}
[data-testid="stAppViewContainer"] {{
  background:
    radial-gradient(1200px 500px at 85% -10%, rgba(34,211,238,0.07), transparent 60%),
    radial-gradient(900px 400px at -10% 10%, rgba(167,139,250,0.06), transparent 60%);
}}
code, pre, [data-testid="stMetricValue"], .sg-mono {{ font-family: var(--sg-mono) !important; }}
[data-testid="stMetricValue"] {{ font-size: 1.6rem; letter-spacing: -0.02em; }}

/* Step headers: monospace step index, quiet divider */
h2 {{ letter-spacing: -0.02em; }}

/* Cards and containers */
[data-testid="stVerticalBlockBorderWrapper"] {{
  transition: border-color .18s ease, box-shadow .18s ease, transform .18s ease;
}}
[data-testid="stVerticalBlockBorderWrapper"]:hover {{ border-color: rgba(34,211,238,0.35); }}
[data-testid="stExpander"] details {{ border-radius: 10px; }}
[data-testid="stExpander"] summary:hover {{ color: var(--sg-accent); }}

/* Metric tiles */
[data-testid="stMetric"] {{
  background: linear-gradient(180deg, rgba(255,255,255,0.025), rgba(255,255,255,0));
  border: 1px solid var(--sg-border); border-radius: 12px; padding: 12px 14px;
}}

/* Upload dropzone */
[data-testid="stFileUploaderDropzone"] {{
  border: 1.5px dashed rgba(34,211,238,0.45) !important; border-radius: 14px !important;
  background: repeating-linear-gradient(135deg, rgba(34,211,238,0.035) 0 12px, transparent 12px 24px)
              !important;
  transition: box-shadow .2s ease, border-color .2s ease, transform .2s ease;
}}
[data-testid="stFileUploaderDropzone"]:hover {{
  border-color: var(--sg-accent) !important; transform: translateY(-1px);
  box-shadow: 0 0 0 1px rgba(34,211,238,0.35), 0 0 28px rgba(34,211,238,0.18);
}}

/* Buttons: crisp, with a soft glow on the primary action */
.stButton > button, .stDownloadButton > button {{
  border-radius: 10px; transition: box-shadow .18s ease, transform .12s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{ transform: translateY(-1px); }}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {{
  color: #04121A; font-weight: 600;
  box-shadow: 0 0 0 1px rgba(34,211,238,0.5), 0 0 22px rgba(34,211,238,0.25);
}}

/* Finding cards: severity rail on the left edge */
[class*="st-key-sev-"] {{
  border: 1px solid var(--sg-border); border-radius: 12px; padding: 12px 14px 6px 16px;
  background: linear-gradient(90deg, rgba(255,255,255,0.025), rgba(255,255,255,0) 40%);
  transition: box-shadow .18s ease, border-color .18s ease;
}}
{_SEVERITY_RAILS}

/* Filter tags: quiet outline instead of solid blocks */
[data-baseweb="tag"], [data-testid="stMultiSelectTagsContainer"] [data-tag] {{
  background: rgba(34,211,238,0.10) !important; border: 1px solid rgba(34,211,238,0.35);
  color: {ACCENT} !important; font-family: var(--sg-mono); font-size: 11.5px;
}}
[data-baseweb="tag"] span, [data-testid="stMultiSelectTagsContainer"] [data-tag] * {{
  color: {ACCENT} !important;
}}

/* Local-model insight panel */
.st-key-insight {{
  border: 1px solid rgba(167,139,250,0.35); border-radius: 14px; padding: 0 14px 12px;
  background: linear-gradient(180deg, rgba(167,139,250,0.06), rgba(11,15,25,0) 40%);
}}

/* Hero */
.sg-hero {{ display: flex; align-items: center; gap: 16px; margin: 4px 0 6px; }}
.sg-logo {{
  width: 46px; height: 46px; border-radius: 12px; flex: none; display: grid; place-items: center;
  background: conic-gradient(from 200deg, {ACCENT}, {ACCENT_2}, {ACCENT});
  box-shadow: 0 0 26px rgba(34,211,238,0.35);
}}
.sg-logo span {{
  width: 40px; height: 40px; border-radius: 10px; background: {BG}; display: grid;
  place-items: center; font: 700 15px var(--sg-mono); color: {ACCENT};
}}
.sg-hero h1 {{ margin: 0; padding: 0; font-size: 2.1rem; letter-spacing: -0.03em; }}
.sg-tagline {{ color: var(--sg-muted); margin: 2px 0 0; font-size: .95rem; }}
.sg-chips {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0 4px; }}
.sg-chip {{
  font: 500 11.5px var(--sg-mono); letter-spacing: .02em; padding: 3px 9px; border-radius: 999px;
  border: 1px solid var(--sg-border); color: var(--sg-muted); background: rgba(255,255,255,0.02);
  white-space: nowrap;
}}
.sg-chip.on {{ color: {
    ACCENT
}; border-color: rgba(34,211,238,0.45); background: rgba(34,211,238,0.07); }}
.sg-chip.rec {{
  color: #04121A; background: {ACCENT}; border-color: {ACCENT};
  box-shadow: 0 0 16px rgba(34,211,238,0.35);
}}
.sg-chip[title] {{ cursor: help; }}

/* Panels built from HTML */
.sg-panel {{
  border: 1px solid var(--sg-border); border-radius: 14px; padding: 16px 18px;
  background: linear-gradient(180deg, {SURFACE_2}, {SURFACE}); color: {TEXT};
}}
.sg-label {{
  font: 600 11px var(--sg-mono); letter-spacing: .12em; text-transform: uppercase;
  color: var(--sg-muted); margin: 0 0 10px;
}}
.sg-overview {{ display: grid; grid-template-columns: 170px 1fr; gap: 22px; align-items: center; }}
@media (max-width: 640px) {{ .sg-overview {{ grid-template-columns: 1fr; justify-items: center; }} }}
.sg-ring {{
  --c: {
    ACCENT
}; width: 150px; height: 150px; border-radius: 50%; display: grid; place-items: center;
  background: conic-gradient(var(--c) calc(var(--p) * 1%), rgba(255,255,255,0.06) 0);
  box-shadow: 0 0 34px -6px var(--c);
}}
.sg-ring > div {{
  width: 122px; height: 122px; border-radius: 50%; background: {BG}; display: grid;
  place-items: center; text-align: center;
}}
.sg-ring b {{ display: block; font: 700 30px var(--sg-mono); color: {TEXT}; line-height: 1.1; }}
.sg-ring small {{ font: 500 10px var(--sg-mono); color: var(--sg-muted); letter-spacing: .1em; }}
.sg-ring-wrap {{ display: flex; flex-direction: column; align-items: center; gap: 12px; }}
.sg-worst {{
  font: 600 11.5px var(--sg-mono); color: var(--c); border: 1px solid var(--c);
  border-radius: 999px; padding: 3px 10px; box-shadow: 0 0 16px -4px var(--c);
  text-transform: uppercase; letter-spacing: .06em; white-space: nowrap;
}}
.sg-meter {{ display: grid; grid-template-columns: 74px 1fr 34px; align-items: center; gap: 10px;
  margin: 7px 0; font: 500 12px var(--sg-mono); color: var(--sg-muted); }}
.sg-track {{ height: 8px; border-radius: 999px; background: rgba(255,255,255,0.06); overflow: hidden; }}
.sg-fill {{ height: 100%; border-radius: 999px; background: var(--c); box-shadow: 0 0 12px var(--c);
  transform-origin: left; animation: sg-grow .7s cubic-bezier(.2,.8,.2,1) both; }}
.sg-meter .n {{ text-align: right; color: {TEXT}; }}
.sg-meter.zero .n {{ color: var(--sg-muted); opacity: .6; }}
.sg-note {{ font-size: 12px; color: var(--sg-muted); margin-top: 10px; }}

/* Affected-column grid */
.sg-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); gap: 8px; }}
.sg-col {{
  --c: {ACCENT}; display: block; text-decoration: none !important; color: {TEXT} !important;
  border: 1px solid var(--sg-border); border-radius: 10px; padding: 9px 11px;
  background: rgba(255,255,255,0.015); transition: border-color .15s, box-shadow .15s, transform .15s;
}}
.sg-col:hover {{ border-color: var(--c); box-shadow: 0 0 18px -6px var(--c); transform: translateY(-1px); }}
.sg-col .name {{ font: 600 12.5px var(--sg-mono); overflow: hidden; text-overflow: ellipsis;
  white-space: nowrap; display: block; }}
.sg-col .meta {{ font: 500 11px var(--sg-mono); color: var(--sg-muted); margin-top: 4px;
  display: flex; align-items: center; gap: 6px; }}
.sg-dot {{ width: 8px; height: 8px; border-radius: 50%; background: var(--c); box-shadow: 0 0 8px var(--c);
  flex: none; }}

/* Split bar */
.sg-split {{ display: flex; height: 34px; border-radius: 10px; overflow: hidden;
  border: 1px solid var(--sg-border); font: 600 12px var(--sg-mono); }}
.sg-split > div {{ display: flex; align-items: center; justify-content: center; white-space: nowrap;
  overflow: hidden; padding: 0 8px; animation: sg-grow .7s cubic-bezier(.2,.8,.2,1) both;
  transform-origin: left; }}
.sg-split .train {{ background: linear-gradient(90deg, rgba(34,211,238,0.28), rgba(34,211,238,0.14));
  color: {ACCENT}; }}
.sg-split .test {{ background: linear-gradient(90deg, rgba(167,139,250,0.30), rgba(167,139,250,0.16));
  color: {ACCENT_2}; border-left: 1px solid {BG}; }}
.sg-split .other {{ background: rgba(255,255,255,0.05); color: var(--sg-muted); }}

/* Terminal header of the insight panel */
.sg-term {{ display: flex; align-items: center; gap: 8px; padding: 10px 0 8px;
  border-bottom: 1px solid rgba(167,139,250,0.25); margin-bottom: 8px; font: 500 12px var(--sg-mono);
  color: var(--sg-muted); }}
.sg-term i {{ width: 9px; height: 9px; border-radius: 50%; display: inline-block; }}
.sg-term .title {{ color: {ACCENT_2}; margin-left: 6px; }}
.sg-term .model {{ margin-left: auto; color: {TEXT}; }}
.sg-ref {{
  font: 600 11.5px var(--sg-mono); color: {ACCENT_2} !important; text-decoration: none !important;
  border: 1px solid rgba(167,139,250,0.4); border-radius: 6px; padding: 1px 6px;
  background: rgba(167,139,250,0.08); transition: background .15s, box-shadow .15s;
}}
.sg-ref:hover {{ background: rgba(167,139,250,0.2); box-shadow: 0 0 12px rgba(167,139,250,0.35); }}

/* Anchor targets glow briefly when jumped to */
.sg-anchor {{ position: relative; top: -70px; display: block; height: 0; }}

@keyframes sg-grow {{ from {{ transform: scaleX(0); }} to {{ transform: scaleX(1); }} }}
@media (prefers-reduced-motion: reduce) {{
  *, *::before, *::after {{ animation: none !important; transition: none !important; }}
}}
</style>
"""


# ----------------------------------------------------------------- HTML builders


def chips_html(items: Iterable[tuple[str, str]], tooltips: Mapping[str, str] | None = None) -> str:
    """``items`` are ``(text, state)`` with state ``''``, ``'on'`` or ``'rec'``."""
    tooltips = tooltips or {}
    parts = []
    for text, state in items:
        tip = tooltips.get(text)
        title = f' title="{esc(tip)}"' if tip else ""
        parts.append(f'<span class="sg-chip {esc(state)}"{title}>{esc(text)}</span>')
    return f'<div class="sg-chips">{"".join(parts)}</div>'


def hero_html(version: str) -> str:
    return (
        '<div class="sg-hero"><div class="sg-logo"><span>SG</span></div><div>'
        "<h1>SplitGuard</h1>"
        '<p class="sg-tagline">Evidence-driven checks for ML data leakage and train/test '
        "contamination.</p></div></div>"
        + chips_html(
            [
                (f"v{version}", "on"),
                ("deterministic rules", ""),
                ("runs locally", ""),
                ("no telemetry", ""),
                ("no raw rows in reports", ""),
            ]
        )
    )


def overview_html(by_severity: Mapping[str, int], checks_run: int, checks_not_run: int) -> str:
    """Ring for the worst severity present plus one meter per severity level.

    The ring shows *coverage* (share of checks that ran), coloured by the most severe
    finding; it is never a "safety score".
    """
    worst = next((s for s in SEVERITY_ORDER if by_severity.get(s, 0)), None)
    n_checks = checks_run + checks_not_run
    coverage = round(100 * checks_run / n_checks) if n_checks else 0
    worst_color = SEVERITY_COLORS[worst] if worst else MUTED
    worst_text = f"most severe: {worst}" if worst else "no findings"
    ring = (
        '<div class="sg-ring-wrap">'
        f'<div class="sg-ring" style="--c:{ACCENT};--p:{coverage}" '
        f'title="{checks_run} of {n_checks} checks ran">'
        f"<div><div><b>{coverage}%</b><small>CHECKS RAN</small></div></div></div>"
        f'<span class="sg-worst" style="--c:{worst_color}">{esc(worst_text)}</span></div>'
    )
    peak = max([by_severity.get(s, 0) for s in SEVERITY_ORDER] + [1])
    meters = []
    for sev in SEVERITY_ORDER:
        n = by_severity.get(sev, 0)
        width = 0 if not n else max(6, round(100 * n / peak))
        meters.append(
            f'<div class="sg-meter{" zero" if not n else ""}" style="--c:{SEVERITY_COLORS[sev]}">'
            f"<span>{sev}</span><div class='sg-track'><div class='sg-fill' "
            f"style='width:{width}%'></div></div><span class='n'>{n}</span></div>"
        )
    note = (
        f"Ring: {checks_run} of {n_checks} checks ran; the rest were skipped or failed and their "
        "areas were not assessed. Bars: findings by severity (potential impact if real). "
        "Confidence is shown on each finding."
    )
    return (
        '<div class="sg-panel"><p class="sg-label">Risk overview</p>'
        f'<div class="sg-overview">{ring}<div>{"".join(meters)}'
        f'<div class="sg-note">{esc(note)}</div></div></div></div>'
    )


def columns_grid_html(findings: Sequence[object], limit: int = 24) -> str:
    """One tile per affected column: most severe finding, rule IDs, link to its card."""
    by_col: dict[str, dict[str, object]] = {}
    for f in findings:  # findings arrive most severe first
        for col in getattr(f, "affected_columns", []) or []:
            entry = by_col.setdefault(
                col, {"sev": f.severity.value, "rules": [], "fid": f.finding_id}
            )
            if f.rule_id not in entry["rules"]:
                entry["rules"].append(f.rule_id)
    if not by_col:
        return ""
    tiles = []
    for col, entry in list(by_col.items())[:limit]:
        sev = str(entry["sev"])
        rules = ", ".join(entry["rules"])
        tiles.append(
            f'<a class="sg-col" href="#{anchor_id(str(entry["fid"]))}" target="_self" '
            f'style="--c:{SEVERITY_COLORS[sev]}" title="{esc(col)}">'
            f'<span class="name">{esc(col)}</span>'
            f'<span class="meta"><span class="sg-dot"></span>{esc(sev)} · {esc(rules)}</span></a>'
        )
    more = len(by_col) - limit
    extra = (
        f'<div class="sg-note">+{more} more column(s) in the findings below.</div>'
        if more > 0
        else ""
    )
    return (
        '<div class="sg-panel"><p class="sg-label">Columns named in findings</p>'
        f'<div class="sg-grid">{"".join(tiles)}</div>{extra}'
        '<div class="sg-note">Click a column to jump to its most severe finding. A column listed '
        "here is not necessarily leaking; read the finding's evidence and limitations.</div></div>"
    )


def split_bar_html(segments: Sequence[tuple[str, float, str]], caption: str = "") -> str:
    """``segments`` are ``(label, share 0-1, css class)``; shares are normalised."""
    total = sum(max(0.0, s) for _, s, _ in segments) or 1.0
    parts = []
    for label, share, css in segments:
        pct = 100 * max(0.0, share) / total
        if pct <= 0:
            continue
        parts.append(
            f'<div class="{esc(css)}" style="width:{pct:.2f}%" title="{esc(label)}">{esc(label)}</div>'
        )
    note = f'<div class="sg-note">{esc(caption)}</div>' if caption else ""
    return f'<div class="sg-split">{"".join(parts)}</div>{note}'


def terminal_header_html(model: str, endpoint: str) -> str:
    return (
        '<div class="sg-term"><i style="background:#FF5F57"></i><i style="background:#FEBC2E">'
        '</i><i style="background:#28C840"></i><span class="title">insight</span>'
        f'<span class="model" title="{esc(endpoint)}">{esc(model)}</span></div>'
    )


def ref_chip_html(finding_id: str) -> str:
    """A small chip that jumps to the finding's card."""
    return (
        f'<a class="sg-ref" href="#{anchor_id(finding_id)}" target="_self" '
        f'title="Jump to the original finding">{esc(finding_id)}</a>'
    )


def anchor_html(finding_id: str) -> str:
    return f'<span class="sg-anchor" id="{anchor_id(finding_id)}"></span>'


def confidence_html(confidence: str) -> str:
    n = CONFIDENCE_BARS.get(confidence, 0)
    return "▮" * n + "▯" * (3 - n)
