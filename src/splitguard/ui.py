"""SplitGuard Streamlit interface.

Launch with ``splitguard-ui`` (recommended: binds to localhost and disables
Streamlit usage statistics) or ``streamlit run`` on ``splitguard/ui_app.py``.

All detection happens in the existing engine, reached through
:mod:`splitguard.ui_adapter`. This module only lays out widgets and renders the
engine's own result objects.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

import streamlit as st

from splitguard import __version__
from splitguard import ui_theme as theme
from splitguard.config import Limits
from splitguard.explain import (
    DISCLAIMER,
    ExplainerError,
    ExplainerSettings,
    ExplanationInputError,
    ExplanationProvider,
    OllamaProvider,
    build_explanation_input,
    explain_report,
)
from splitguard.schemas import AuditResult, Finding
from splitguard.ui_adapter import (
    CATEGORIES,
    CONFIDENCES,
    MODE_ROLES,
    MODES,
    NO_PROOF_NOTE,
    OVERLAP_NOTE,
    PRIVACY_NOTE,
    SCENARIO_HELP,
    SCENARIO_LABELS,
    SEVERITIES,
    STRATEGY_HELP,
    WARNING_GROUP_TITLES,
    FormState,
    LoadedFile,
    UIError,
    category_label,
    create_split_files,
    default_split_label,
    describe_error,
    evidence_label,
    filter_findings,
    finding_view,
    group_warnings,
    headline,
    json_report_bytes,
    load_file,
    markdown_report,
    md_escape,
    missing_inputs,
    partial_note_count,
    run_audit,
    safe_preview,
    skipped_by_reason,
    split_labels,
)
from splitguard.ui_samples import Sample, samples

NONE = "(not set)"
SOURCE_UPLOAD = "Upload my own CSV files"
SOURCE_SAMPLE = "Try a sample dataset"
SEVERITY_COLORS = {
    "critical": "red",
    "high": "orange",
    "medium": "yellow",
    "low": "blue",
    "info": "gray",
}
CONFIDENCE_COLORS = {"high": "violet", "medium": "blue", "low": "gray"}
ROLE_LABELS = {"train": "Training CSV", "test": "Testing CSV", "data": "CSV file"}


def _msg(text: str) -> str:
    """Escape engine/user text for st.markdown-based elements."""
    return md_escape(text)


def md_strip(text: str) -> str:
    """Plain text from the small amount of Markdown in help strings (for tooltips)."""
    return text.replace("*", "").replace("`", "")


def _code(text: str) -> str:
    """Inline code span. Escapes are not processed inside code, so strip backticks instead."""
    return "`" + " ".join(str(text).replace("`", "'").split()) + "`"


def _show_error(err: UIError) -> None:
    body = "\n".join(f"- {_msg(m)}" for m in err.messages)
    if err.hints:
        body += "\n\n" + "\n".join(f"*Hint:* {_msg(h)}" for h in err.hints)
    st.error(f"**{_msg(err.title)}**\n\n{body}", icon=":material/error:")


# ----------------------------------------------------------------------- sidebar


def _sidebar() -> None:
    limits = Limits()
    with st.sidebar:
        st.markdown(f"### SplitGuard {__version__}")
        st.caption(
            "Audits ML datasets for data leakage and train/test contamination using "
            "transparent, rule-based checks."
        )
        st.markdown("**Severity vs confidence**")
        st.caption(
            "*Severity* is the potential impact if the issue is real. *Confidence* is how "
            "strongly the data supports it. A finding can be high severity but low confidence."
        )
        st.markdown("**Privacy**")
        st.caption(PRIVACY_NOTE)
        st.markdown("**Limits**")
        st.caption(
            f"Up to {limits.max_file_bytes // (1024 * 1024)} MB, {limits.max_rows:,} rows and "
            f"{limits.max_columns:,} columns per file. UTF-8 CSV only."
        )


# ------------------------------------------------------------------ step 1: data


def _landing() -> None:
    st.markdown(
        "SplitGuard checks a dataset for patterns that commonly make ML evaluation look "
        "better than it really is: features that copy the target, rows shared between train "
        "and test, entities crossing the split, and values recorded after prediction time."
    )
    cols = st.columns(3)
    blurbs = [
        ("Separate train and test files", "Audit an existing split for contamination."),
        ("One file with a split column", "Same audit, when train/test are labelled in one file."),
        ("One unsplit file", "Within-dataset checks and advice on how to split it."),
    ]
    for col, (title, text) in zip(cols, blurbs, strict=True):
        with col.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(text)
    st.info(
        "Upload CSV files below, or choose **Try a sample dataset** to see SplitGuard on "
        "synthetic data with planted problems.",
        icon=":material/info:",
    )


def _load_cached(role: str, name: str, data: bytes) -> LoadedFile | UIError:
    """Parse each distinct upload once per session."""
    cache: dict[str, Any] = st.session_state.setdefault("_loaded", {})
    key = f"{role}:{hashlib.sha256(data).hexdigest()}:{name}"
    if key not in cache:
        # Keep only the latest file per role so replaced uploads are released from memory.
        for old in [k for k in cache if k.startswith(f"{role}:")]:
            del cache[old]
        try:
            cache[key] = load_file(name, data)
        except Exception as exc:  # noqa: BLE001 - shown via describe_error
            cache[key] = describe_error(exc)
    return cache[key]


def _step_data() -> tuple[str, dict[str, LoadedFile], Sample | None, bool]:
    st.header("1. Data", divider="gray")
    source = st.radio("Data source", [SOURCE_UPLOAD, SOURCE_SAMPLE], horizontal=True, key="source")
    sample: Sample | None = None
    raw: dict[str, tuple[str, bytes]] = {}
    if source == SOURCE_SAMPLE:
        options = samples()
        key = st.selectbox(
            "Sample dataset",
            list(options),
            format_func=lambda k: options[k].title,
            key="sample_key",
        )
        sample = options[key]
        mode = sample.mode
        st.caption(sample.description)
        raw = {role: (sample.names[role], sample.files[role]) for role in sample.files}
    else:
        mode = st.radio(
            "What are you auditing?",
            list(MODES),
            format_func=MODES.get,
            key="mode",
            help="Pick the shape of your data. SplitGuard never splits data during an audit.",
        )
        cols = st.columns(len(MODE_ROLES[mode]))
        for col, role in zip(cols, MODE_ROLES[mode], strict=True):
            upload = col.file_uploader(ROLE_LABELS[role], type=["csv"], key=f"upload_{mode}_{role}")
            if upload is not None:
                raw[role] = (upload.name, upload.getvalue())

    files: dict[str, LoadedFile] = {}
    had_error = False
    if raw:
        cols = st.columns(len(raw))
        for col, (role, (name, data)) in zip(cols, raw.items(), strict=True):
            with col:
                loaded = _load_cached(role, name, data)
                if isinstance(loaded, UIError):
                    _show_error(loaded)
                    had_error = True
                    continue
                files[role] = loaded
                with st.container(border=True):
                    st.markdown(f"**{_msg(loaded.name)}** ({role})")
                    st.caption(f"{len(loaded.frame):,} rows x {len(loaded.columns):,} columns")
                    with st.expander("Preview first 5 rows"):
                        st.caption("Values are truncated. The preview is shown only to you.")
                        st.dataframe(safe_preview(loaded.frame), hide_index=True)
    return mode, files, sample, had_error


# ------------------------------------------------------------- step 2: configure


def _select(
    label: str, options: list[str], key: str, default: str | None = None, **kw: Any
) -> str | None:
    choices = [NONE, *options]
    index = choices.index(default) if default in choices else 0
    value = st.selectbox(label, choices, index=index, key=key, **kw)
    return None if value == NONE else value


def _step_configure(mode: str, files: dict[str, LoadedFile], sample: Sample | None) -> FormState:
    st.header("2. Configure", divider="gray")
    suggested = dict(sample.suggested) if sample else {}
    scope = f"{mode}:{sample.key if sample else 'upload'}"  # reset widgets when context changes
    k = lambda name: f"cfg:{scope}:{name}"  # noqa: E731
    primary = files.get("train") or files.get("data")
    columns = primary.columns if primary else []
    state = FormState(mode=mode)
    if sample:
        st.caption("Suggested settings for this sample are pre-filled. Change them freely.")

    if mode == "split_column":
        state.split_column = _select(
            "Split column",
            columns,
            k("split_column"),
            suggested.get("split_column"),
            help="The column whose values say whether each row is train or test.",
        )
        if state.split_column:
            labels = split_labels(primary.frame, state.split_column)
            if labels is None:
                st.error(
                    "This column has more than 20 distinct values; it does not look like a split column."
                )
            else:
                c1, c2 = st.columns(2)
                with c1:
                    state.train_value = _select(
                        "Label meaning 'train'",
                        labels,
                        k("train_value"),
                        default_split_label(labels, "train"),
                    )
                with c2:
                    state.test_value = _select(
                        "Label meaning 'test'",
                        labels,
                        k("test_value"),
                        default_split_label(labels, "test"),
                    )
        columns = [c for c in columns if c != state.split_column]

    left, right = st.columns(2)
    with left:
        st.subheader("Target and use case")
        state.target = _select(
            "Target column",
            columns,
            k("target"),
            suggested.get("target"),
            help="The column the model predicts. Needed for target-leakage checks.",
        )
        state.task_type = st.selectbox(
            "Task type",
            ["auto", "classification", "regression"],
            key=k("task_type"),
            help="'auto' treats text or two-valued targets as classification and other numbers as "
            "regression. Override for numeric multi-class labels.",
        )
        scenarios = list(SCENARIO_HELP)
        default_scenario = suggested.get("scenario", "unknown")
        state.scenario = st.selectbox(
            "How will the model be used?",
            scenarios,
            index=scenarios.index(default_scenario),
            key=k("scenario"),
            format_func=SCENARIO_LABELS.get,
            help="SplitGuard does not guess this. It decides whether shared entities or "
            "overlapping time periods matter.",
        )
        st.caption(SCENARIO_HELP[state.scenario])
        state.entity_id = _select(
            "Entity ID column (optional)",
            columns,
            k("entity_id"),
            suggested.get("entity_id"),
            help="Identifies a customer, patient, device... Used to check whether entities appear "
            "in both partitions and for group-based split advice.",
        )
    with right:
        st.subheader("Timing (optional)")
        st.caption(
            "Without timestamps, temporal leakage is reported as *not assessed* - never as safe."
        )
        state.prediction_timestamp = _select(
            "Prediction time column",
            columns,
            k("prediction_timestamp"),
            suggested.get("prediction_timestamp"),
            help="When each prediction would be made. ISO-8601 text; naive times are read as UTC.",
        )
        state.outcome_timestamp = _select(
            "Outcome time column",
            columns,
            k("outcome_timestamp"),
            suggested.get("outcome_timestamp"),
            help="When the target outcome became known.",
        )
        taken = {state.target, state.entity_id, state.prediction_timestamp, state.outcome_timestamp}
        candidates = [c for c in columns if c not in taken]
        suggested_av: dict[str, str] = suggested.get("feature_availability", {})
        chosen = st.multiselect(
            "Features with a 'recorded at' timestamp",
            candidates,
            key=k("availability_features"),
            default=[f for f in suggested_av if f in candidates],
            help="For each selected feature, pick the column that records when its value became "
            "known. Only explicit metadata is used; names are never trusted.",
        )
        for feature in chosen:
            column = _select(
                f"'{feature}' recorded-at column",
                [c for c in candidates if c != feature],
                k(f"availability:{feature}"),
                suggested_av.get(feature),
            )
            if column:
                state.feature_availability[feature] = column

    with st.expander("More options"):
        state.exclude_columns = st.multiselect(
            "Columns that are not model features",
            columns,
            key=k("exclude"),
            help="For example row numbers or audit columns. They are excluded from feature checks.",
        )
        state.min_lead_time = (
            st.text_input(
                "Minimum lead time between prediction and outcome",
                key=k("lead"),
                placeholder="e.g. 7D or 12h (leave empty if unknown)",
            )
            or None
        )
        if mode != "unsplit":
            state.expect_target_in_test = st.checkbox(
                "Test data should contain the target",
                value=True,
                key=k("expect_target"),
            )
            state.test_target_diagnostics = st.checkbox(
                "Also scan test labels for target copies (diagnostic only)",
                key=k("diagnostics"),
                help="Findings are marked as diagnostics. Never use them to choose features or tune models.",
            )
        else:
            c1, c2 = st.columns(2)
            state.test_size = c1.slider(
                "Test share for split advice", 0.05, 0.5, 0.2, 0.05, key=k("test_size")
            )
            state.seed = int(c2.number_input("Random seed", 0, 2**31 - 1, 42, key=k("seed")))
    return state


# ------------------------------------------------------------------- step 3: run


def _fingerprint(state: FormState, files: dict[str, LoadedFile]) -> str:
    parts = [state.fingerprint()] + [
        f"{r}:{f.table.metadata['sha256']}" for r, f in sorted(files.items())
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def _step_run(state: FormState, files: dict[str, LoadedFile]) -> None:
    st.header("3. Run", divider="gray")
    missing = missing_inputs(state, files)
    if missing:
        st.caption("Before running, provide: " + "; ".join(missing) + ".")
    elif not state.target:
        st.caption(
            ":material/info: No target column selected, so target-leakage checks will be "
            "skipped. Other checks still run."
        )
    if st.button("Run audit", type="primary", disabled=bool(missing), key="run_audit"):
        st.session_state.pop("split_download", None)
        for key in ("llm_explanation", "llm_error"):
            st.session_state.pop(key, None)
        with st.spinner("Running SplitGuard checks..."):
            try:
                result = run_audit(state, files)
            except Exception as exc:  # noqa: BLE001 - converted to a safe message
                st.session_state["audit_error"] = describe_error(exc)
                st.session_state.pop("result", None)
            else:
                st.session_state["result"] = result
                st.session_state["result_fp"] = _fingerprint(state, files)
                st.session_state.pop("audit_error", None)
    if "audit_error" in st.session_state:
        _show_error(st.session_state["audit_error"])


# --------------------------------------------------------------- step 4: results


def _badge_row(f: Finding) -> None:
    with st.container(horizontal=True, gap="small", vertical_alignment="center"):
        st.badge(
            f.severity.value.upper(),
            color=SEVERITY_COLORS[f.severity.value],
            help="Severity: potential impact if the issue is real.",
        )
        st.badge(
            f"{theme.confidence_html(f.confidence.value)} {f.confidence.value} confidence - "
            f"{evidence_label(f)}",
            color=CONFIDENCE_COLORS[f.confidence.value],
            help="Confidence: how strongly the data supports this finding.",
        )
        st.caption(f"{category_label(f.category.value)} | {_code(f.finding_id)}")


def _render_finding(f: Finding, index: int = 0) -> None:
    view = finding_view(f)
    st.html(theme.anchor_html(f.finding_id))
    with st.container(key=f"sev-{f.severity.value}-{index}"):
        _badge_row(f)
        st.markdown(f"**{_msg(view['title'])}**")
        st.markdown(_msg(view["description"]))
        if f.confidence.value == "low":
            st.caption("This is a prompt to review, not evidence of leakage.")
        with st.expander("Evidence, action and limitations"):
            if view["affected_columns"]:
                st.markdown(
                    "**Affected columns:** " + ", ".join(_code(c) for c in view["affected_columns"])
                )
            st.markdown(f"**Recommended action:** {_msg(view['recommended_action'])}")
            st.markdown(
                "**Why this is not proof:**\n"
                + "\n".join(f"- {_msg(x)}" for x in view["limitations"])
            )
            st.markdown("**Evidence**")
            st.json(view["evidence"], expanded=1)


def _coverage(result: AuditResult) -> None:
    groups = group_warnings(result.warnings)
    not_run = skipped_by_reason(result)
    n_not_run = sum(len(ids) for _, _, ids in not_run)
    n_notes = sum(len(v) for v in groups.values())
    partial = partial_note_count(result)
    failed = any(status == "failed" for status, _, _ in not_run)
    # Always-visible one-line summary; the details open automatically only when rows
    # were partly assessed, timestamps tied or checks failed.
    summary = (
        f"Not fully assessed: {n_not_run} check(s) did not run and {n_notes} note(s) about "
        "missing metadata or coverage. Open the panel below for details."
    )
    if partial or failed:
        st.warning(
            _msg(summary)
            + f" **{partial} note(s) concern partly assessed rows or tied timestamps.**",
            icon=":material/visibility_off:",
        )
    elif n_not_run or n_notes:
        st.info(_msg(summary), icon=":material/visibility_off:")
    title = f"What was not fully assessed ({n_notes} note(s), {n_not_run} check(s) not run)"
    with st.expander(title, expanded=bool(partial or failed)):
        for key, heading in WARNING_GROUP_TITLES.items():
            if groups[key]:
                st.markdown(f"**{heading}**")
                st.markdown("\n".join(f"- {_msg(w)}" for w in groups[key]))
        if not_run:
            st.markdown("**Checks that did not run**")
            st.markdown(
                "\n".join(
                    f"- {_msg(reason)} ({status}): " + ", ".join(_code(i) for i in ids)
                    for status, reason, ids in not_run
                )
            )
        if not n_notes and not not_run:
            st.caption("All checks ran with full coverage of the configured metadata.")


def _findings(result: AuditResult) -> None:
    st.subheader("Findings")
    st.caption(OVERLAP_NOTE)
    if not result.findings:
        st.info(NO_PROOF_NOTE, icon=":material/info:")
        return
    present_sev = [s for s in SEVERITIES if any(f.severity.value == s for f in result.findings)]
    present_cat = [c for c in CATEGORIES if any(f.category.value == c for f in result.findings)]
    present_conf = [c for c in CONFIDENCES if any(f.confidence.value == c for f in result.findings)]
    c1, c2, c3 = st.columns(3)
    sev = c1.multiselect("Severity", present_sev, default=present_sev, key="filter_severity")
    cat = c2.multiselect(
        "Category",
        present_cat,
        default=present_cat,
        key="filter_category",
        format_func=category_label,
    )
    conf = c3.multiselect("Confidence", present_conf, default=present_conf, key="filter_confidence")
    shown = filter_findings(result.findings, sev, cat, conf)
    st.caption(f"Showing {len(shown)} of {len(result.findings)} finding(s), most severe first.")
    if not shown:
        st.info("No findings match these filters.")
    for index, f in enumerate(shown):
        _render_finding(f, index)


def _split_section(
    result: AuditResult, state: FormState, files: dict[str, LoadedFile], fresh: bool
) -> None:
    rec = result.split_recommendation
    if rec is None:
        return
    st.subheader("Split recommendation")
    st.caption("This dataset was not split. Train/test contamination was not audited.")
    recommended = rec.strategy if rec.status == "recommended" else None
    st.html(
        theme.chips_html(
            [(name, "rec" if name == recommended else "") for name in STRATEGY_HELP],
            tooltips={name: md_strip(text) for name, text in STRATEGY_HELP.items()},
        )
    )
    test_size = rec.parameters.get("test_size")
    if recommended and isinstance(test_size, (int, float)) and 0 < test_size < 1:
        st.html(
            theme.split_bar_html(
                [
                    (f"train {1 - test_size:.0%}", 1 - test_size, "train"),
                    (f"test {test_size:.0%}", test_size, "test"),
                ],
                caption=f"Proposed {recommended} split. Change the test share under "
                "2. Configure > More options. A split separates rows by this rule only; it does "
                "not remove duplicate rows or leaked features.",
            )
        )
    if rec.status == "needs_user_input":
        st.warning(
            "**SplitGuard needs more information before recommending a split.**\n\n"
            + "\n".join(f"- {_msg(m)}" for m in rec.missing_information)
            + "\n\nSet these in **2. Configure** and run the audit again.",
            icon=":material/help:",
        )
        if rec.rationale:
            st.markdown("\n".join(f"- {_msg(r)}" for r in rec.rationale))
    else:
        st.success(f"Recommended strategy: **{_msg(rec.strategy)}**", icon=":material/call_split:")
        st.markdown(_msg(STRATEGY_HELP.get(rec.strategy, "")))
        for heading, items in (
            ("Why", rec.rationale),
            ("Assumptions", rec.assumptions),
            ("Warnings", rec.warnings),
        ):
            if items:
                st.markdown(f"**{heading}**\n" + "\n".join(f"- {_msg(x)}" for x in items))
    with st.expander("Evidence, parameters and limitations"):
        st.markdown("\n".join(f"- {_msg(x)}" for x in rec.limitations))
        st.json(
            {"parameters": rec.to_dict()["parameters"], "evidence": rec.to_dict()["evidence"]},
            expanded=1,
        )
    with st.expander("How the split strategies differ"):
        st.dataframe(
            [{"strategy": k, "when to use": v} for k, v in STRATEGY_HELP.items()],
            hide_index=True,
            column_config={"when to use": st.column_config.TextColumn(width="large")},
        )
    if rec.status == "recommended":
        _split_creation(rec, state, files, fresh)


def _split_creation(rec: Any, state: FormState, files: dict[str, LoadedFile], fresh: bool) -> None:
    with st.expander("Create this split (optional)"):
        if rec.strategy == "group_and_time":
            st.info(
                "Group-and-time evaluation must be designed deliberately; SplitGuard does not create it."
            )
            return
        if not fresh:
            st.warning(
                "Settings changed since the audit ran. Run the audit again before creating a split."
            )
            return
        st.caption(
            "Nothing is split unless you confirm below. The split only separates rows by the "
            "chosen rule; it does not remove duplicates or leaked features. Audit the result."
        )
        typed = st.text_input(
            f"Type `{rec.strategy}` to confirm the strategy", key="confirm_strategy"
        )
        ack = st.checkbox("I understand this split does not remove leakage.", key="confirm_ack")
        if st.button(
            "Create split", disabled=not (typed.strip() == rec.strategy and ack), key="create_split"
        ):
            try:
                st.session_state["split_download"] = create_split_files(
                    files["data"], rec, confirmed_strategy=typed, state=state
                )
            except Exception as exc:  # noqa: BLE001
                _show_error(describe_error(exc))
        download = st.session_state.get("split_download")
        if download is not None:
            rep = download.report
            st.success(
                f"Split created: {rep['n_train_rows']} train rows, {rep['n_test_rows']} test rows."
            )
            for w in rep.get("warnings", []):
                st.warning(_msg(w))
            c1, c2, c3 = st.columns(3)
            c1.download_button(
                "train.csv", download.train_csv, "train.csv", "text/csv", key="dl_train"
            )
            c2.download_button("test.csv", download.test_csv, "test.csv", "text/csv", key="dl_test")
            c3.download_button(
                "split_report.json",
                download.report_json,
                "split_report.json",
                "application/json",
                key="dl_split_report",
            )


EXPLAINER_NOTE = (
    "Optional. A local language model, served by Ollama on this computer, writes a "
    "plain-language explanation of the findings above. It receives only a summarised copy "
    "of the report (shown below) - never your CSV rows - and it cannot change, add or remove "
    "findings. Nothing happens until you click a button."
)


def make_explainer_provider(settings: ExplainerSettings) -> ExplanationProvider:
    """Provider factory (tests replace it with a mock)."""
    return OllamaProvider(settings)


def _explanation(result: AuditResult, fresh: bool) -> None:
    try:
        defaults = ExplainerSettings.from_env()
    except ExplanationInputError as exc:
        st.error(_msg(f"Explainer settings from the environment are invalid: {exc}"))
        return
    default_model = defaults.model
    shown_model = str(st.session_state.get("llm_model", default_model)).strip() or default_model
    st.html(theme.terminal_header_html(shown_model, defaults.host))
    st.markdown("#### Plain-language explanation")
    st.caption(EXPLAINER_NOTE)
    model = st.text_input(
        "Ollama model",
        value=default_model,
        key="llm_model",
        help="Any model already installed in your local Ollama. SplitGuard never downloads models.",
    )
    try:
        settings = ExplainerSettings.from_env(model=model.strip() or None)
        payload = build_explanation_input(result)
    except ExplanationInputError as exc:
        st.error(_msg(str(exc)), icon=":material/error:")
        return
    st.caption(
        f"Model {_code(settings.model)} via Ollama at {_code(settings.host)} (local only), "
        f"timeout {settings.timeout_s:g} s."
    )
    with st.expander("Exactly what would be sent to the model"):
        st.caption(
            "An allowlisted summary of this report: findings, aggregate evidence, column names "
            "and coverage notes. No CSV rows, file names, hashes or paths."
        )
        st.json(payload, expanded=1)
    if not fresh:
        st.caption("Run the audit again before requesting an explanation.")
    provider = make_explainer_provider(settings)
    explain_clicked = st.button(
        "Explain these results",
        key="llm_explain",
        disabled=not fresh,
        type="primary",
        width="stretch",
        icon=":material/auto_awesome:",
    )
    if st.button("Check model availability", key="llm_check", width="stretch"):
        st.session_state["llm_status"] = provider.check()
    status = st.session_state.get("llm_status")
    if status is not None and status.model == settings.model:
        if status.ok:
            st.success(_msg(status.message), icon=":material/check_circle:")
        else:
            st.warning(_msg(status.message), icon=":material/cloud_off:")
    stamp = (st.session_state.get("result_fp"), settings.model)
    if explain_clicked:
        with st.spinner(f"Asking {settings.model}. Small local models can take a few minutes..."):
            try:
                st.session_state["llm_explanation"] = (stamp, explain_report(result, provider))
                st.session_state["llm_reveal"] = True
                st.session_state.pop("llm_error", None)
            except ExplainerError as exc:
                st.session_state["llm_error"] = (stamp, str(exc))
                st.session_state.pop("llm_explanation", None)
    error = st.session_state.get("llm_error")
    if error is not None and error[0] == stamp:
        st.error(
            "**No explanation was produced.** " + _msg(error[1]) + "\n\nThe audit results above "
            "are unaffected.",
            icon=":material/error:",
        )
    stored = st.session_state.get("llm_explanation")
    if stored is not None and stored[0] == stamp:
        _render_explanation(result, stored[1])


def _typed(words: list[str]) -> Any:
    for i, word in enumerate(words):
        yield word if i == 0 else " " + word
        time.sleep(0.012)


def _render_explanation(result: AuditResult, exp: Any) -> None:
    st.info(
        f"**LLM-generated narrative ({_msg(exp.model)}) - not an audit finding.** "
        + _msg(DISCLAIMER),
        icon=":material/smart_toy:",
    )
    for flag in exp.review_flags:
        st.warning(_msg(flag), icon=":material/flag:")
    if st.session_state.pop("llm_reveal", False):
        # Cosmetic reveal of the already-validated text, once, right after it arrives.
        words = _msg(exp.overview).split(" ")
        st.write_stream(_typed(words))
    else:
        st.markdown(_msg(exp.overview))
    by_id = {f.finding_id: f for f in result.findings}
    for fe in exp.findings:
        finding = by_id.get(fe.finding_id)
        with st.container(border=True):
            st.html(theme.ref_chip_html(fe.finding_id))
            if finding is not None:
                _badge_row(finding)
            st.markdown(f"**{_msg(fe.title)}**")
            st.caption(f"Model's explanation. Per the report this is a {fe.evidence_kind}.")
            st.markdown(
                "\n".join(
                    f"- **{label}:** {_msg(text)}"
                    for label, text in (
                        ("Why it matters", fe.why_it_matters),
                        ("Evidence", fe.evidence),
                        ("What to check", fe.what_to_check),
                        ("Limitations", fe.limitations),
                    )
                )
            )
            if finding is not None:
                view = finding_view(finding)
                with st.expander("Original finding from the audit (authoritative)"):
                    st.markdown(_msg(view["description"]))
                    st.markdown(f"**Recommended action:** {_msg(view['recommended_action'])}")
                    st.markdown(
                        "**Why this is not proof:**\n"
                        + "\n".join(f"- {_msg(x)}" for x in view["limitations"])
                    )
                    st.json(view["evidence"], expanded=1)
    st.markdown("**Coverage (model's summary)**")
    st.markdown(_msg(exp.coverage))
    st.caption(
        "The complete list of skipped checks and coverage warnings is in "
        "'What was not fully assessed' above."
    )
    st.markdown(f"**Caveats:** {_msg(exp.caveats)}")
    if exp.unexplained_finding_ids:
        st.caption(
            "Not explained by the model: "
            + ", ".join(_code(i) for i in exp.unexplained_finding_ids)
            + ". See the findings above."
        )
    if exp.findings_not_sent:
        st.caption(
            "Not sent to the model (size limit): "
            + ", ".join(_code(i) for i in exp.findings_not_sent)
            + ". See the findings above."
        )


def _downloads(result: AuditResult) -> None:
    st.subheader("Download")
    stable = st.checkbox(
        "Omit the generated-at timestamp (byte-identical reports for identical inputs)",
        key="dl_stable",
    )
    report = result.to_dict(include_generated_at=not stable)
    c1, c2 = st.columns(2)
    c1.download_button(
        "Full JSON report",
        json_report_bytes(result, include_generated_at=not stable),
        "splitguard_report.json",
        "application/json",
        type="primary",
        key="dl_json",
        help="The engine's complete report: configuration, checks, findings and evidence.",
    )
    c2.download_button(
        "Readable summary (Markdown)",
        markdown_report(report).encode("utf-8"),
        "splitguard_report.md",
        "text/markdown",
        key="dl_md",
        help="A concise summary generated from the JSON report.",
    )
    st.caption(
        "Reports contain file names, hashes, column names and statistics - no raw rows or local paths."
    )


def _partition_bar(result: AuditResult) -> None:
    """Rows per partition, straight from the audit result (train/test modes only)."""
    rows = {name: int(p.get("n_rows", 0)) for name, p in result.partitions.items()}
    if not {"train", "test"} <= set(rows) or not sum(rows.values()):
        return
    total = rows["train"] + rows["test"]
    st.html(
        theme.split_bar_html(
            [
                (f"train {rows['train']:,}", rows["train"] / total, "train"),
                (f"test {rows['test']:,}", rows["test"] / total, "test"),
            ],
            caption=f"Audited partitions: {rows['test'] / total:.0%} of rows are in test.",
        )
    )


def _step_results(result: AuditResult, state: FormState, files: dict[str, LoadedFile]) -> None:
    st.header("4. Results", divider="gray")
    fresh = st.session_state.get("result_fp") == _fingerprint(state, files)
    if not fresh:
        st.warning(
            "Data or settings changed since this audit ran. Click **Run audit** to update.",
            icon=":material/update:",
        )
    level, message = headline(result)
    getattr(st, level)(_msg(message))
    summary = result.summary()
    sev = summary["by_severity"]
    not_run = summary["checks_skipped"] + summary["checks_failed"]
    ring, tiles = st.columns([1.6, 1], gap="medium")
    with ring:
        st.html(theme.overview_html(sev, summary["checks_run"], not_run))
    with tiles:
        m = st.columns(2)
        m[0].metric("Findings", summary["total_findings"])
        m[1].metric("Critical / high", f"{sev['critical']} / {sev['high']}")
        m = st.columns(2)
        m[0].metric("Checks run", summary["checks_run"])
        m[1].metric(
            "Not run",
            not_run,
            help="Skipped (missing metadata or not applicable) plus failed checks.",
        )
        _partition_bar(result)
    _coverage(result)
    _split_section(result, state, files, fresh)
    grid = theme.columns_grid_html(result.findings)
    if grid:
        st.html(grid)
    main, side = st.columns([1.55, 1], gap="large")
    with main:
        _findings(result)
    with side, st.container(key="insight"):
        _explanation(result, fresh)
    _downloads(result)
    with st.expander("Engine limitations"):
        st.markdown("\n".join(f"- {_msg(x)}" for x in result.limitations))


# --------------------------------------------------------------------------- app


def run_app() -> None:
    st.set_page_config(page_title="SplitGuard", page_icon=":material/shield:", layout="wide")
    st.html(theme.CSS)
    _sidebar()
    st.html(theme.hero_html(__version__))
    intro = st.container()  # filled only while no data is loaded, so it sits above step 1
    mode, files, sample, had_error = _step_data()
    if not files and not had_error:
        with intro:
            _landing()
        return
    state = _step_configure(mode, files, sample)
    _step_run(state, files)
    result = st.session_state.get("result")
    if isinstance(result, AuditResult):
        _step_results(result, state, files)


# `streamlit run src/splitguard/ui.py` executes this file as __main__; without this
# the page would stay blank. Importing the module (ui_app.py, tests) does not run it.
if __name__ == "__main__":
    run_app()
