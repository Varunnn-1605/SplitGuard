"""Streamlit AppTest tests for the optional explanation section.

A mock provider replaces Ollama, so no model server is needed. Skipped when
Streamlit is not installed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit")

import splitguard.ui as ui  # noqa: E402
from explain_fakes import FakeProvider  # noqa: E402
from splitguard.explain import ProviderUnavailableError  # noqa: E402
from test_ui_app import run_audit, start, text, use_sample, widget  # noqa: E402


@pytest.fixture
def provider(monkeypatch):
    fake = FakeProvider(model="qwen2.5:3b-instruct")
    monkeypatch.setattr(ui, "make_explainer_provider", lambda settings: fake)
    return fake


@pytest.fixture
def audit_calls(monkeypatch):
    calls = []
    original = ui.run_audit

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(ui, "run_audit", counting)
    return calls


def audited():
    at = use_sample(start(), "train_test")
    return run_audit(at)


def test_audit_works_as_before_and_no_model_is_contacted(provider, audit_calls):
    at = audited()
    body = text(at)
    assert "Findings" in body and "TGT-001" in body
    assert "Plain-language explanation" in body
    assert "qwen2.5:3b-instruct" in body  # configured model is shown
    assert provider.calls == [] and provider.checks == 0
    assert len(audit_calls) == 1


def test_explanation_is_requested_explicitly_and_labelled(provider, audit_calls):
    at = audited()
    findings_before = text(at).count("TGT-001")
    at.button(key="llm_explain").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(provider.calls) == 1
    assert len(audit_calls) == 1  # explaining did not re-run the audit
    body = text(at)
    assert "LLM-generated narrative" in body and "not an audit finding" in body
    assert "Original finding from the audit (authoritative)" in [e.label for e in at.expander]
    assert "Why it matters" in body
    # Deterministic sections are still shown alongside the narrative.
    assert "What was not fully assessed" in " ".join(e.label for e in at.expander)
    assert text(at).count("TGT-001") > findings_before


def test_unavailable_model_shows_clear_message_and_keeps_results(monkeypatch):
    fake = FakeProvider(
        exc=ProviderUnavailableError("Cannot reach Ollama at http://127.0.0.1:11434.")
    )
    monkeypatch.setattr(ui, "make_explainer_provider", lambda settings: fake)
    at = audited()
    at.button(key="llm_explain").click().run()
    assert not at.exception
    body = text(at)
    assert "No explanation was produced" in body and "Cannot reach Ollama" in body
    assert "audit results above are unaffected" in body
    assert "TGT-001" in body


def test_availability_check_and_model_override(monkeypatch):
    seen = []

    def factory(settings):
        seen.append(settings.model)
        return FakeProvider(model=settings.model, available=False)

    monkeypatch.setattr(ui, "make_explainer_provider", factory)
    at = audited()
    at.text_input(key="llm_model").set_value("qwen2.5:7b-instruct").run()
    at.button(key="llm_check").click().run()
    assert not at.exception
    assert seen[-1] == "qwen2.5:7b-instruct"
    assert "fake model is not installed" in text(at)


def test_invalid_model_name_is_rejected_in_the_ui(provider):
    at = audited()
    at.text_input(key="llm_model").set_value("bad model; rm").run()
    assert not at.exception
    assert "Model names may contain" in text(at)
    assert provider.calls == []


def test_stale_results_disable_explanation(provider):
    at = audited()
    assert not at.button(key="llm_explain").disabled
    widget(at, "selectbox", "scenario").set_value("independent").run()
    assert "changed since this audit ran" in text(at)
    assert at.button(key="llm_explain").disabled
    assert provider.calls == []


def test_rerunning_the_audit_clears_the_previous_explanation(provider):
    at = audited()
    at.button(key="llm_explain").click().run()
    assert "LLM-generated narrative" in text(at)
    run_audit(at)
    assert "LLM-generated narrative" not in text(at)
