from __future__ import annotations

import streamlit as st


def init_session_state_defaults() -> None:
    """Initialize default keys in st.session_state."""
    if "has_prediction" not in st.session_state:
        st.session_state["has_prediction"] = False
    if "last_request_payload" not in st.session_state:
        st.session_state["last_request_payload"] = None
    if "last_prediction_response" not in st.session_state:
        st.session_state["last_prediction_response"] = None


