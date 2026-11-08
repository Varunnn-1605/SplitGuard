"""Model providers for the explanation layer.

``ExplanationProvider`` is the only interface the explainer depends on, so the
local Ollama provider can be replaced without touching the audit engine.

``OllamaProvider`` talks to Ollama's documented REST API with the standard
library only. Deliberately:

* only loopback hosts are accepted (``localhost``, ``127.0.0.1``, ``::1``);
* environment variables such as ``OLLAMA_HOST``, ``OLLAMA_API_KEY`` and proxy
  settings are ignored, and redirects are refused, so a request can only reach
  the endpoint the user selected;
* nothing is downloaded or started: a missing server or model is reported as an
  error with the command the user can run themselves;
* prompts and responses are never logged.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlsplit

from splitguard.explain.errors import (
    ExplanationInputError,
    ModelNotFoundError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)

DEFAULT_MODEL = "qwen2.5:3b-instruct"
DEFAULT_HOST = "http://127.0.0.1:11434"
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
ENV_MODEL = "SPLITGUARD_LLM_MODEL"
ENV_HOST = "SPLITGUARD_OLLAMA_HOST"
ENV_TIMEOUT = "SPLITGUARD_LLM_TIMEOUT"
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")


def validate_local_host(host: str) -> str:
    """Return a normalised ``scheme://host:port`` for a loopback endpoint."""
    parts = urlsplit(host.strip())
    if parts.scheme not in {"http", "https"}:
        raise ExplanationInputError("The Ollama endpoint must start with http:// or https://.")
    if parts.username or parts.password or parts.query or parts.fragment:
        raise ExplanationInputError("The Ollama endpoint must not contain credentials or a query.")
    if parts.path not in {"", "/"}:
        raise ExplanationInputError("The Ollama endpoint must not contain a path.")
    hostname = (parts.hostname or "").lower()
    if hostname not in LOOPBACK_HOSTS:
        raise ExplanationInputError(
            "Only a local Ollama endpoint is allowed (localhost, 127.0.0.1 or ::1)."
        )
    try:
        port = parts.port or 11434
    except ValueError as exc:
        raise ExplanationInputError("The Ollama endpoint has an invalid port.") from exc
    shown = f"[{hostname}]" if ":" in hostname else hostname
    return f"{parts.scheme}://{shown}:{port}"


@dataclass(frozen=True)
class ExplainerSettings:
    """Model and generation settings. Conservative defaults for small local models."""

    model: str = DEFAULT_MODEL
    host: str = DEFAULT_HOST
    timeout_s: float = 300.0
    check_timeout_s: float = 5.0
    num_ctx: int = 6144
    num_predict: int = 1024
    temperature: float = 0.1
    top_p: float = 0.9
    seed: int = 42

    def __post_init__(self) -> None:
        model = self.model.strip() if isinstance(self.model, str) else ""
        if not _MODEL_NAME.match(model):
            raise ExplanationInputError(
                "Model names may contain letters, digits and . _ : / - (for example "
                f"{DEFAULT_MODEL})."
            )
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "host", validate_local_host(self.host))
        if not 1 <= self.timeout_s <= 3600:
            raise ExplanationInputError("The timeout must be between 1 and 3600 seconds.")
        if not 0.5 <= self.check_timeout_s <= 60:
            raise ExplanationInputError("The availability-check timeout must be 0.5-60 seconds.")
        if not 2048 <= self.num_ctx <= 32768:
            raise ExplanationInputError("num_ctx must be between 2048 and 32768 tokens.")
        if not 128 <= self.num_predict <= 4096:
            raise ExplanationInputError("num_predict must be between 128 and 4096 tokens.")
        if not 0 <= self.temperature <= 1 or not 0 < self.top_p <= 1:
            raise ExplanationInputError("temperature must be 0-1 and top_p in (0, 1].")

    @classmethod
    def from_env(cls, **overrides: Any) -> ExplainerSettings:
        """Read ``SPLITGUARD_LLM_MODEL``, ``SPLITGUARD_OLLAMA_HOST`` and
        ``SPLITGUARD_LLM_TIMEOUT``; explicit (non-None) overrides win."""
        values: dict[str, Any] = {}
        if os.environ.get(ENV_MODEL):
            values["model"] = os.environ[ENV_MODEL]
        if os.environ.get(ENV_HOST):
            values["host"] = os.environ[ENV_HOST]
        if os.environ.get(ENV_TIMEOUT):
            try:
                values["timeout_s"] = float(os.environ[ENV_TIMEOUT])
            except ValueError as exc:
                raise ExplanationInputError(f"{ENV_TIMEOUT} must be a number of seconds.") from exc
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)

    def generation_options(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
        }


@dataclass(frozen=True)
class Availability:
    """Result of a provider availability check."""

    ok: bool
    server_reachable: bool
    model_available: bool
    model: str
    endpoint: str
    message: str
    installed_models: tuple[str, ...] = field(default_factory=tuple)


@runtime_checkable
class ExplanationProvider(Protocol):
    """Anything that turns a system + user prompt into a JSON string."""

    @property
    def model(self) -> str: ...

    def generate(self, system: str, user: str, output_schema: dict[str, Any]) -> str: ...

    def check(self) -> Availability: ...


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
        return None  # a redirect becomes an HTTPError instead of being followed


def _not_running_message(endpoint: str) -> str:
    if shutil.which("ollama") is None:
        return (
            f"Cannot reach Ollama at {endpoint}, and no 'ollama' command was found. Install "
            "Ollama from https://ollama.com/download, start it, then try again."
        )
    return (
        f"Cannot reach Ollama at {endpoint}. Start it (open the Ollama app or run "
        "'ollama serve') and try again."
    )


def _missing_model_message(model: str) -> str:
    return (
        f"The model '{model}' is not installed in Ollama. Download it yourself with "
        f"'ollama pull {model}' (SplitGuard never downloads models), or choose an installed "
        "model."
    )


class OllamaProvider:
    """Local Ollama provider (``/api/tags`` and ``/api/chat``)."""

    def __init__(self, settings: ExplainerSettings | None = None) -> None:
        self.settings = settings or ExplainerSettings()
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())

    @property
    def model(self) -> str:
        return self.settings.model

    @property
    def endpoint(self) -> str:
        return self.settings.host

    def _request(self, method: str, path: str, body: dict | None, timeout: float) -> dict:
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with self._opener.open(request, timeout=timeout) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as exc:
            detail = _error_detail(exc)
            if exc.code == 404 and "not found" in detail.lower():
                raise ModelNotFoundError(_missing_model_message(self.model)) from None
            raise ProviderResponseError(
                f"Ollama returned HTTP {exc.code}" + (f": {detail}" if detail else ".")
            ) from None
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise ProviderTimeoutError(_timeout_message(timeout)) from None
            raise ProviderUnavailableError(_not_running_message(self.endpoint)) from None
        except TimeoutError:
            raise ProviderTimeoutError(_timeout_message(timeout)) from None
        except (ConnectionError, OSError):
            raise ProviderUnavailableError(_not_running_message(self.endpoint)) from None
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ProviderResponseError("Ollama's response was larger than allowed.")
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ProviderResponseError("Ollama returned a response that is not JSON.") from None
        if not isinstance(parsed, dict):
            raise ProviderResponseError("Ollama returned an unexpected response shape.")
        return parsed

    def installed_models(self) -> tuple[str, ...]:
        payload = self._request("GET", "/api/tags", None, self.settings.check_timeout_s)
        models = payload.get("models")
        if not isinstance(models, list):
            raise ProviderResponseError("Ollama's model list has an unexpected shape.")
        names = []
        for item in models:
            if isinstance(item, dict):
                name = item.get("model") or item.get("name")
                if isinstance(name, str):
                    names.append(name)
        return tuple(sorted(set(names)))

    def check(self) -> Availability:
        """Is the server reachable and the configured model installed? Never downloads."""
        try:
            names = self.installed_models()
        except (ProviderUnavailableError, ProviderTimeoutError, ProviderResponseError) as exc:
            return Availability(False, False, False, self.model, self.endpoint, str(exc))
        wanted = {self.model, f"{self.model}:latest"} if ":" not in self.model else {self.model}
        available = bool(wanted & set(names))
        message = (
            f"Ollama is running at {self.endpoint} and '{self.model}' is installed."
            if available
            else _missing_model_message(self.model)
        )
        return Availability(available, True, available, self.model, self.endpoint, message, names)

    def generate(self, system: str, user: str, output_schema: dict[str, Any]) -> str:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "format": output_schema,
            "options": self.settings.generation_options(),
        }
        reply = self._request("POST", "/api/chat", body, self.settings.timeout_s)
        if reply.get("error"):
            raise ProviderResponseError("Ollama reported an error while generating.")
        message = reply.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise ProviderResponseError("Ollama's reply did not contain a message.")
        if reply.get("done_reason") == "length":
            raise ProviderResponseError(
                "The model hit the output-token limit before finishing; the partial answer was "
                "discarded. Increase num_predict or use a report with fewer findings."
            )
        return content


def _timeout_message(timeout: float) -> str:
    return (
        f"The model did not answer within {timeout:g} seconds. Small local models on a CPU can "
        f"be slow; raise the timeout ({ENV_TIMEOUT}) or try again once the model is loaded."
    )


def _error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        body = json.loads(exc.read(4096).decode("utf-8", "replace"))
    except (OSError, ValueError):
        return ""
    detail = body.get("error") if isinstance(body, dict) else None
    if not isinstance(detail, str):
        return ""
    return re.sub(r"[\x00-\x1f\x7f]", " ", detail)[:200]
