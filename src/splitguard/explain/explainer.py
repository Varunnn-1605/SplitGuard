"""Turn an existing audit report into a validated, non-authoritative narrative."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from splitguard.explain.output import Explanation, parse_response
from splitguard.explain.payload import PayloadLimits, build_explanation_input
from splitguard.explain.prompt import OUTPUT_SCHEMA, SYSTEM_PROMPT, build_user_message
from splitguard.explain.providers import ExplanationProvider
from splitguard.schemas import AuditResult


def explain_report(
    report: AuditResult | Mapping[str, Any],
    provider: ExplanationProvider,
    limits: PayloadLimits | None = None,
) -> Explanation:
    """Explain ``report`` with ``provider``.

    The report is read, never modified, and no audit is re-run. Only the
    allowlisted payload from :func:`build_explanation_input` is sent. Raises an
    :class:`~splitguard.explain.errors.ExplainerError` subclass on any provider or
    validation problem.
    """
    payload = build_explanation_input(report, limits)
    raw = provider.generate(SYSTEM_PROMPT, build_user_message(payload), OUTPUT_SCHEMA)
    return parse_response(raw, payload, model=provider.model)
