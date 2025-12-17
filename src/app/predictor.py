from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.data.save_data import load_dataframe
from .config import MODEL_JOBLIB_PATH, MODEL_META_PATH, TEST_FEATURES_FILE
from .models import PredictionItem


class SalesPredictor:
    """Lightweight wrapper around the trained LightGBM model."""

    def __init__(self, model_path: Path | None = None, meta_path: Path | None = None) -> None:
        self.model_path: Path = model_path or MODEL_JOBLIB_PATH
        self.meta_path: Path = meta_path or MODEL_META_PATH

        self.model: Any = self._load_model(self.model_path)
        self.meta: dict[str, Any] = self._load_meta(self.meta_path)
        self.feature_names: list[str] = list(self.meta.get("feature_names", []))
        self.test_features: pd.DataFrame = self._load_test_features()

    @staticmethod
    def _load_model(model_path: Path) -> Any:
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        return joblib.load(model_path)

    @staticmethod
    def _load_meta(meta_path: Path) -> dict[str, Any]:
        if not meta_path.exists():
            return {}
        with meta_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _load_test_features() -> pd.DataFrame:
        """Load precomputed test features for lookup-based inference."""
        df = load_dataframe(TEST_FEATURES_FILE, parse_dates=["date"], verbose=False)
        return df

    def _build_features(self, items: list[PredictionItem]) -> pd.DataFrame:
        """Build feature frame from precomputed test_features_cleaned via lookup."""
        if not items:
            return pd.DataFrame()

        base_df = pd.DataFrame(
            [
                {
                    "date": it.date,
                    "store": it.store,
                    "item": it.item,
                }
                for it in items
            ]
        )

        # Приводим к datetime64
        base_df["date"] = pd.to_datetime(base_df["date"])

        # Join с предрасчитанными фичами по (date, store, item)
        merged = base_df.merge(
            self.test_features,
            on=["date", "store", "item"],
            how="left",
            suffixes=("", "_feat"),
        )

        # Проверяем, что все строки нашли соответствие
        if self.feature_names:
            missing_mask = merged[self.feature_names].isna().all(axis=1)
            if missing_mask.any():
                missing_rows = merged.loc[missing_mask, ["date", "store", "item"]]
                raise ValueError(
                    f"No precomputed features found for some rows: "
                    f"{missing_rows.to_dict(orient='records')}"
                )
            X = merged[self.feature_names].copy()
        else:
            # Фолбэк: используем все числовые колонки, кроме ключей
            exclude_cols = {"date", "store", "item"}
            numeric_cols = [
                c for c in merged.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(merged[c])
            ]
            X = merged[numeric_cols].copy()

        X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        return X

    def predict(self, items: list[PredictionItem]) -> list[float]:
        """Return non-negative sales forecasts for given items."""
        if not items:
            return []

        X = self._build_features(items)
        if X.empty:
            # Нет доступных фичей — безопасный фолбэк
            return [0.0 for _ in items]

        raw_preds = self.model.predict(X)
        preds = np.maximum(raw_preds, 0.0)
        return preds.astype(float).tolist()


