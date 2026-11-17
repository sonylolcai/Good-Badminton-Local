"""Environment-aware endpoint and listener settings for service processes.

Only server-side processes import this module.  It never exposes GPU secrets to
the Mini Program, and it deliberately refuses a local endpoint in production
when a caller needs the business API address.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final
from urllib.parse import urlparse


_PROJECT_ROOT: Final = Path(__file__).resolve().parent
_VALID_ENVIRONMENTS: Final = {"development", "production", "test"}


class RuntimeConfigurationError(RuntimeError):
    """A service cannot safely start with the supplied environment values."""


def application_environment() -> str:
    value = os.environ.get("GOOD_BADMINTON_APP_ENV", "development").strip().lower()
    if value not in _VALID_ENVIRONMENTS:
        raise RuntimeConfigurationError(
            "GOOD_BADMINTON_APP_ENV must be development, production, or test."
        )
    return value


def load_runtime_environment() -> Path | None:
    """Load the selected ignored env file without replacing deployment values.

    A deployment platform normally injects variables directly.  Local runs can
    use `.env.development`; production may pass an explicit
    `GOOD_BADMINTON_ENV_FILE` managed outside of source control.
    """

    environment = application_environment()
    configured = os.environ.get("GOOD_BADMINTON_ENV_FILE", "").strip()
    candidate = Path(configured) if configured else _PROJECT_ROOT / f".env.{environment}"
    if not candidate.is_file():
        return None
    for raw_line in candidate.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key.replace("_", "").isalnum():
            os.environ.setdefault(key, value.strip())
    # The file must not silently switch the requested process environment.
    if application_environment() != environment:
        raise RuntimeConfigurationError(
            "GOOD_BADMINTON_APP_ENV in the environment file conflicts with the process value."
        )
    return candidate


def _valid_base_url(value: str, *, variable_name: str, production: bool) -> str:
    normalized = value.strip().rstrip("/")
    parsed = urlparse(normalized)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeConfigurationError(
            f"{variable_name} must be a credential-free http(s) base URL."
        )
    local = parsed.hostname.lower() in {"localhost", "127.0.0.1", "::1"}
    if production and (parsed.scheme != "https" or local):
        raise RuntimeConfigurationError(
            f"{variable_name} must be a non-local HTTPS URL in production."
        )
    return normalized


def business_api_base_url() -> str:
    """Return the WebUI-to-business service URL, never a GPU URL."""

    environment = application_environment()
    value = (
        os.environ.get("GOOD_BADMINTON_BUSINESS_API_URL", "").strip()
        or os.environ.get("GOOD_BADMINTON_BUSINESS_STREAM_URL", "").strip()
    )
    if not value and environment in {"development", "test"}:
        value = "http://127.0.0.1:8080"
    if not value:
        raise RuntimeConfigurationError(
            "GOOD_BADMINTON_BUSINESS_API_URL is required in production."
        )
    return _valid_base_url(
        value,
        variable_name="GOOD_BADMINTON_BUSINESS_API_URL",
        production=environment == "production",
    )


def _port(variable_name: str, default: int) -> int:
    raw_value = os.environ.get(variable_name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise RuntimeConfigurationError(f"{variable_name} must be an integer port.") from exc
    if not 1 <= value <= 65535:
        raise RuntimeConfigurationError(f"{variable_name} must be between 1 and 65535.")
    return value


def business_service_listener() -> tuple[str, int]:
    """Return host/port for the business service, defaults to local port 8080."""

    environment = application_environment()
    host = os.environ.get(
        "GOOD_BADMINTON_BUSINESS_HOST",
        "0.0.0.0" if environment == "production" else "127.0.0.1",
    ).strip()
    if not host:
        raise RuntimeConfigurationError("GOOD_BADMINTON_BUSINESS_HOST cannot be blank.")
    return host, _port("GOOD_BADMINTON_BUSINESS_PORT", 8080)


def webui_listener() -> tuple[str, int]:
    """Return host/port for the operator WebUI; reverse proxies can override it."""

    host = os.environ.get("GOOD_BADMINTON_WEBUI_HOST", "127.0.0.1").strip()
    if not host:
        raise RuntimeConfigurationError("GOOD_BADMINTON_WEBUI_HOST cannot be blank.")
    return host, _port("GOOD_BADMINTON_WEBUI_PORT", 7861)
