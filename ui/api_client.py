from __future__ import annotations

import os
from typing import Any, Dict

import requests
import streamlit as st


DEFAULT_API_BASE_URL = "http://localhost:8000"


def get_api_base_url() -> str:
    """Return base URL for FastAPI service."""
    # 1) ENV переменная имеет приоритет
    env_url = os.getenv("API_BASE_URL")
    if env_url:
        return env_url

    # 2) Опционально читаем из secrets, если они настроены
    try:
        if "api_base_url" in st.secrets:
            return str(st.secrets["api_base_url"])
    except Exception:
        # Нет secrets.toml — просто игнорируем и используем дефолт
        pass

    # 3) Дефолтный локальный адрес
    return DEFAULT_API_BASE_URL


@st.cache_resource(show_spinner=False)
def get_http_client() -> requests.Session:
    """Create and cache a HTTP client session."""
    session = requests.Session()
    return session


def _build_url(path: str) -> str:
    base = get_api_base_url().rstrip("/")
    return f"{base}{path}"


@st.cache_data(show_spinner=False)
def check_health() -> bool:
    """Return True if API /health endpoint responds with status ok."""
    client = get_http_client()
    try:
        resp = client.get(_build_url("/health"), timeout=3)
        if resp.status_code != 200:
            return False
        data = resp.json()
        return isinstance(data, dict) and data.get("status") == "ok"
    except Exception:
        return False


def call_predict_api(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Call /predict endpoint and return parsed JSON."""
    client = get_http_client()
    url = _build_url("/predict")
    resp = client.post(url, json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()


