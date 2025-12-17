from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from .api_client import call_predict_api, check_health
from .state import init_session_state_defaults


def _build_single_instance(date_value, store_value: int, item_value: int) -> Dict[str, Any]:
    return {
        "date": pd.to_datetime(date_value).date().isoformat(),
        "store": int(store_value),
        "item": int(item_value),
    }


def _run_single_prediction(date_value, store_value: int, item_value: int) -> None:
    payload = {"instances": [_build_single_instance(date_value, store_value, item_value)]}
    try:
        response = call_predict_api(payload)
    except Exception as exc:
        st.error(f"Ошибка при вызове API: {exc}")
        return

    st.session_state["has_prediction"] = True
    st.session_state["last_request_payload"] = payload
    st.session_state["last_prediction_response"] = response


def _run_batch_prediction(df: pd.DataFrame) -> None:
    required_cols = {"date", "store", "item"}
    missing = required_cols - set(df.columns)
    if missing:
        st.error(f"В загруженном файле не хватает колонок: {', '.join(sorted(missing))}")
        return

    instances: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        instances.append(
            _build_single_instance(
                row["date"],
                int(row["store"]),
                int(row["item"]),
            )
        )

    payload: Dict[str, Any] = {"instances": instances}
    try:
        response = call_predict_api(payload)
    except Exception as exc:
        st.error(f"Ошибка при вызове API: {exc}")
        return

    st.session_state["has_prediction"] = True
    st.session_state["last_request_payload"] = payload
    st.session_state["last_prediction_response"] = response


def _render_sidebar() -> None:
    st.sidebar.title("Параметры прогноза")

    api_ok = check_health()
    if api_ok:
        st.sidebar.success("API доступно")
    else:
        st.sidebar.warning("API недоступно. Проверьте, что FastAPI-сервис запущен.")

    st.sidebar.markdown("---")

    batch_mode = st.sidebar.checkbox(
        "Режим батча (загрузка CSV)", value=False, key="batch_mode"
    )

    if batch_mode:
        uploaded = st.sidebar.file_uploader(
            "Загрузите CSV с колонками date, store, item",
            type=["csv"],
            key="file_uploader_batch",
        )
        run_batch = st.sidebar.button(
            "Получить прогноз для батча", key="btn_predict_batch"
        )

        if run_batch and uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
            except Exception as exc:
                st.sidebar.error(f"Не удалось прочитать CSV: {exc}")
                return
            _run_batch_prediction(df)
        elif run_batch and uploaded is None:
            st.sidebar.error("Пожалуйста, загрузите CSV-файл.")
        return

    # Одиночный прогноз
    date_value = st.sidebar.date_input("Дата", key="input_date")
    store_value = st.sidebar.number_input(
        "Магазин (store_id)", min_value=1, max_value=10, value=1, step=1, key="input_store"
    )
    item_value = st.sidebar.number_input(
        "Товар (item_id)", min_value=1, max_value=50, value=1, step=1, key="input_item"
    )

    run_single = st.sidebar.button("Получить прогноз", key="btn_predict_single")
    if run_single:
        _run_single_prediction(date_value, store_value, item_value)


def _render_single_result(response: Dict[str, Any]) -> None:
    preds = response.get("predictions") or []
    if not preds:
        st.warning("Ответ API не содержит предсказаний.")
        return

    first = preds[0]
    value = float(first.get("prediction", 0.0))

    col_main, col_side = st.columns([2, 1])
    with col_main:
        st.metric("Прогноз продаж (шт.)", f"{value:,.2f}")
        st.write(
            f"Дата: **{first.get('date')}**, "
            f"Магазин: **{first.get('store')}**, "
            f"Товар: **{first.get('item')}**"
        )
    with col_side:
        st.write("Краткая справка")
        st.caption("Прогноз основан на обученной модели LightGBM с фичами по истории продаж.")


def _render_batch_result(response: Dict[str, Any]) -> None:
    preds = response.get("predictions") or []
    if not preds:
        st.warning("Ответ API не содержит предсказаний.")
        return

    df = pd.DataFrame(preds)
    st.subheader("Результаты батчевого прогноза")
    st.dataframe(df.sort_values(["date", "store", "item"]).reset_index(drop=True))

    with st.expander("Агрегированные показатели по батчу"):
        st.write(
            {
                "min": float(df["prediction"].min()),
                "max": float(df["prediction"].max()),
                "mean": float(df["prediction"].mean()),
            }
        )


def _render_main_result_area() -> None:
    st.title("Прогноз спроса по магазину и товару")

    if not st.session_state.get("has_prediction", False):
        st.info("Заполните параметры в сайдбаре и нажмите «Получить прогноз».")
        return

    response = st.session_state.get("last_prediction_response") or {}
    payload = st.session_state.get("last_request_payload") or {}
    instances = payload.get("instances") or []

    if len(instances) <= 1:
        _render_single_result(response)
    else:
        _render_batch_result(response)

    with st.expander("Детали прогноза"):
        st.subheader("Запрос")
        st.json(payload)
        st.subheader("Ответ модели")
        st.json(response)


def render_forecast_page() -> None:
    """Главная страница: Прогноз."""
    init_session_state_defaults()
    _render_sidebar()
    _render_main_result_area()


