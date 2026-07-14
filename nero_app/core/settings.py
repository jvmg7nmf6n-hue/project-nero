from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parents[1] / "data" / "local_settings.json"


def _truthy(value: str | None, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _streamlit_secrets() -> dict[str, Any]:
    try:
        import streamlit as st
    except Exception:  # pragma: no cover - streamlit may be unavailable in CLI tools.
        return {}
    try:
        raw = dict(st.secrets)
    except Exception:
        return {}

    settings: dict[str, Any] = {}
    secret_map = {
        "TWELVE_DATA_API_KEY": "twelve_data_api_key",
        "GEMINI_API_KEY": "gemini_api_key",
        "SMTP_HOST": "smtp_host",
        "SENDER_EMAIL": "sender_email",
        "EMAIL_APP_PASSWORD": "email_app_password",
        "RECEIVER_EMAIL": "receiver_email",
        "PREFER_LIVE": "prefer_live",
        "USE_LATEST_NEWS": "use_latest_news",
        "MOBILE_ALERTS_ENABLED": "mobile_alerts_enabled",
    }
    for secret_name, setting_name in secret_map.items():
        value = raw.get(secret_name)
        if value not in (None, ""):
            settings[setting_name] = value
    smtp_port = raw.get("SMTP_PORT")
    if smtp_port not in (None, ""):
        try:
            settings["smtp_port"] = int(smtp_port)
        except (TypeError, ValueError):
            settings["smtp_port"] = 465
    return settings


def _env_settings() -> dict[str, Any]:
    settings: dict[str, Any] = {}
    env_map = {
        "TWELVE_DATA_API_KEY": "twelve_data_api_key",
        "GEMINI_API_KEY": "gemini_api_key",
        "SMTP_HOST": "smtp_host",
        "SENDER_EMAIL": "sender_email",
        "EMAIL_APP_PASSWORD": "email_app_password",
        "RECEIVER_EMAIL": "receiver_email",
    }
    for env_name, setting_name in env_map.items():
        value = os.getenv(env_name, "").strip()
        if value:
            settings[setting_name] = value

    smtp_port = os.getenv("SMTP_PORT", "").strip()
    if smtp_port:
        try:
            settings["smtp_port"] = int(smtp_port)
        except ValueError:
            settings["smtp_port"] = 465

    settings["prefer_live"] = _truthy(os.getenv("PREFER_LIVE"), default=True)
    settings["use_latest_news"] = _truthy(os.getenv("USE_LATEST_NEWS"), default=True)
    settings["mobile_alerts_enabled"] = _truthy(os.getenv("MOBILE_ALERTS_ENABLED"), default=True)
    settings.setdefault("smtp_host", "smtp.gmail.com")
    settings.setdefault("smtp_port", 465)
    return settings


def load_settings(path: Path = DEFAULT_SETTINGS_PATH) -> dict[str, Any]:
    settings = _env_settings()
    settings.update(_streamlit_secrets())
    if not path.exists():
        return settings
    try:
        local_settings = json.loads(path.read_text())
    except json.JSONDecodeError:
        return settings
    settings.update(local_settings)
    return settings


def save_settings(settings: dict[str, Any], path: Path = DEFAULT_SETTINGS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2, sort_keys=True))
