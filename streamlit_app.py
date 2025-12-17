import streamlit as st

from ui.page_forecast import render_forecast_page
from ui.page_about import render_about_page


PAGES = {
    "Прогноз": render_forecast_page,
    "О модели": render_about_page,
}


def main() -> None:
    st.set_page_config(
        page_title="Store Item Demand Forecasting",
        page_icon="📈",
        layout="wide",
    )

    st.sidebar.title("Навигация")
    page_name = st.sidebar.selectbox(
        "Страница", list(PAGES.keys()), index=0, key="page_selector"
    )

    render_page = PAGES.get(page_name)
    if render_page is not None:
        render_page()
    else:
        st.error("Выбрана неизвестная страница.")


if __name__ == "__main__":
    main()


