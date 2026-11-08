"""Errors raised by the optional explanation layer.

Every message is safe to show to a user: it never contains prompt text, report
contents or dataset values.
"""

from __future__ import annotations


class ExplainerError(Exception):
    """Base class for explanation-layer errors."""


class ExplanationInputError(ExplainerError, ValueError):
    """The supplied report or settings cannot be used."""


class ProviderUnavailableError(ExplainerError):
    """The local model server could not be reached."""


class ModelNotFoundError(ExplainerError):
    """The server is running but the configured model is not installed."""


class ProviderTimeoutError(ExplainerError):
    """The model did not answer within the configured timeout."""


class ProviderResponseError(ExplainerError):
    """The server returned an error or an unusable HTTP response."""


class InvalidModelOutputError(ExplainerError):
    """The model's answer failed validation and was not shown."""
