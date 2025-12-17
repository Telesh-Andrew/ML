from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import streamlit as st


@st.cache_data(show_spinner=False)
def _load_model_meta() -> Dict[str, Any]:
    meta_path = Path("artifacts") / "models" / "lightgbm_smape_11.74.json"
    if not meta_path.exists():
        return {}
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def render_about_page() -> None:
    st.title("О модели")

    st.markdown(
        """
Модель прогнозирует ежедневные продажи (`sales`) для комбинаций `(date, store, item)`
на основе исторических фичей (лаги, скользящие средние и т.п.), собранных в ноутбуке
`02_model_training.ipynb`.
        """
    )

    meta = _load_model_meta()
    if not meta:
        st.warning("Файл с метаданными модели не найден.")
        return

    st.subheader("Ключевые характеристики модели")
    basic_info = {
        "model_type": meta.get("model_type"),
        "train_period": meta.get("train_period"),
        "val_period": meta.get("val_period"),
        "target": meta.get("target"),
    }
    st.json(basic_info)

    metrics = meta.get("metrics") or {}
    if metrics:
        st.subheader("Метрики качества на валидации")
        st.write(metrics)

    feature_names = meta.get("feature_names") or []
    if feature_names:
        st.subheader("Используемые фичи (первые 30)")
        st.write(feature_names[:30])


