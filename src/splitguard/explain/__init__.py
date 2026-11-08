"""Optional plain-language explanations of SplitGuard reports by a local LLM.

The audit engine stays authoritative: this package only reads a finished
report, sends an allowlisted summary of it (never raw rows) to a local model,
validates the answer and returns it as a separate, non-authoritative
``Explanation``. It never changes the report, re-runs checks or creates
findings. It uses only the standard library; the model runs in Ollama, which
the user installs and manages.
"""

from splitguard.explain.errors import (
    ExplainerError,
    ExplanationInputError,
    InvalidModelOutputError,
    ModelNotFoundError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from splitguard.explain.explainer import explain_report
from splitguard.explain.output import DISCLAIMER, Explanation, FindingExplanation, render_text
from splitguard.explain.payload import PayloadLimits, build_explanation_input
from splitguard.explain.providers import (
    DEFAULT_HOST,
    DEFAULT_MODEL,
    Availability,
    ExplainerSettings,
    ExplanationProvider,
    OllamaProvider,
)

__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_MODEL",
    "DISCLAIMER",
    "Availability",
    "ExplainerError",
    "ExplainerSettings",
    "ExplanationInputError",
    "ExplanationProvider",
    "Explanation",
    "FindingExplanation",
    "InvalidModelOutputError",
    "ModelNotFoundError",
    "OllamaProvider",
    "PayloadLimits",
    "ProviderResponseError",
    "ProviderTimeoutError",
    "ProviderUnavailableError",
    "build_explanation_input",
    "explain_report",
    "render_text",
]
