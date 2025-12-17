"""
Пайплайн обучения моделей для прогнозирования продаж.

Основные функции:
- Подготовка данных для обучения
- Обучение baseline модели (ARIMA)
- Обучение ML моделей (LightGBM, XGBoost, RandomForest)
- Сравнение моделей
- Сохранение лучшей модели
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

# ML библиотеки
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
import xgboost as xgb

# Метрики
from .metrics import smape, calculate_regression_metrics

warnings.filterwarnings('ignore')


# ============================================================================
# Подготовка данных
# ============================================================================

def prepare_data_for_training(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
    target_col: str = 'sales',
    exclude_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.DataFrame], Optional[pd.Series], 
           Optional[pd.DataFrame], Optional[pd.Series], List[str]]:
    """
    Подготовка данных для обучения моделей.
    
    ВАЖНО: Предотвращает data leakage - пропуски заполняются только на train,
    затем статистики применяются к val/test.
    
    Args:
        train_df: Обучающий датасет с фичами
        val_df: Валидационный датасет (опционально)
        test_df: Тестовый датасет (опционально)
        target_col: Название целевой колонки
        exclude_cols: Колонки для исключения из фичей
        
    Returns:
        Tuple с (X_train, y_train, X_val, y_val, X_test, y_test, feature_names)
    """
    if exclude_cols is None:
        exclude_cols = ['date', 'store', 'item', target_col]
    
    # Выбираем фичи (исключаем служебные колонки)
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    # Train
    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].copy()
    
    # Обработка пропусков на train (вычисляем статистики)
    # Заполняем медианой для числовых фичей
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    fill_values = X_train[numeric_cols].median()
    
    X_train[numeric_cols] = X_train[numeric_cols].fillna(fill_values)
    X_train = X_train.fillna(0)  # Для остальных колонок
    
    # Validation
    X_val = None
    y_val = None
    if val_df is not None:
        X_val = val_df[feature_cols].copy()
        y_val = val_df[target_col].copy()
        # Применяем статистики из train
        X_val[numeric_cols] = X_val[numeric_cols].fillna(fill_values)
        X_val = X_val.fillna(0)
    
    # Test
    X_test = None
    y_test = None
    if test_df is not None:
        X_test = test_df[feature_cols].copy()
        if target_col in test_df.columns:
            y_test = test_df[target_col].copy()
        # Применяем статистики из train
        X_test[numeric_cols] = X_test[numeric_cols].fillna(fill_values)
        X_test = X_test.fillna(0)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


# ============================================================================
# Baseline: ARIMA
# ============================================================================

class ARIMABaseline:
    """
    Простая ARIMA baseline модель для временных рядов.
    
    Для каждого (store, item) комбинируется отдельный временной ряд,
    на котором обучается ARIMA.
    """
    
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1), random_state: int = 42):
        """
        Args:
            order: Параметры ARIMA (p, d, q)
            random_state: Для воспроизводимости
        """
        self.order = order
        self.random_state = random_state
        self.models = {}  # Словарь моделей для каждой (store, item) пары
        self.is_fitted = False
        
    def fit(self, df: pd.DataFrame, target_col: str = 'sales'):
        """
        Обучение ARIMA моделей для каждой (store, item) комбинации.
        
        Args:
            df: DataFrame с колонками date, store, item, target_col
            target_col: Название целевой колонки
        """
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError:
            raise ImportError("statsmodels не установлен. Установите: pip install statsmodels")
        
        # Сортировка по дате
        df = df.sort_values(['store', 'item', 'date']).copy()
        
        # Обучение отдельной модели для каждой (store, item) пары
        for (store, item), group in df.groupby(['store', 'item']):
            ts = group.set_index('date')[target_col].sort_index()
            
            # Пропускаем слишком короткие ряды
            if len(ts) < max(self.order) + 2:
                continue
            
            try:
                model = ARIMA(ts, order=self.order)
                fitted_model = model.fit()
                self.models[(store, item)] = fitted_model
            except Exception as e:
                # Если не удалось обучить ARIMA, пропускаем
                warnings.warn(f"Не удалось обучить ARIMA для (store={store}, item={item}): {e}")
                continue
        
        self.is_fitted = True
        return self
    
    def predict(self, df: pd.DataFrame, steps: int = 1) -> np.ndarray:
        """
        Предсказание для новых данных.
        
        Args:
            df: DataFrame с колонками date, store, item
            steps: Количество шагов вперед для предсказания
            
        Returns:
            Массив предсказаний
        """
        if not self.is_fitted:
            raise ValueError("Модель не обучена! Вызовите fit() сначала.")
        
        df = df.sort_values(['store', 'item', 'date']).copy()
        predictions = []
        
        for (store, item), group in df.groupby(['store', 'item']):
            if (store, item) in self.models:
                try:
                    # Предсказание на следующее значение
                    forecast = self.models[(store, item)].forecast(steps=steps)
                    predictions.extend(forecast.values if hasattr(forecast, 'values') else [forecast])
                except Exception:
                    # Если не удалось предсказать, используем последнее значение
                    # или среднее по группе
                    last_value = group.get('sales', pd.Series([0])).iloc[-1] if 'sales' in group.columns else 0
                    predictions.extend([last_value] * len(group))
            else:
                # Если модель не обучена, используем среднее значение
                predictions.extend([0] * len(group))
        
        # Выравниваем длину
        if len(predictions) < len(df):
            predictions.extend([0] * (len(df) - len(predictions)))
        
        return np.array(predictions[:len(df)])


# ============================================================================
# Обучение ML моделей
# ============================================================================

def train_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    random_state: int = 42
) -> Tuple[Any, Dict[str, float], np.ndarray, Optional[np.ndarray]]:
    """
    Обучение одной модели и вычисление метрик.
    
    Args:
        model: Модель для обучения (должна иметь fit() и predict())
        X_train: Обучающие фичи
        y_train: Обучающая цель
        X_val: Валидационные фичи (опционально)
        y_val: Валидационная цель (опционально)
        random_state: Для воспроизводимости
        
    Returns:
        Tuple с (обученная_модель, метрики, y_pred_train, y_pred_val)
    """
    # Установка random_state если поддерживается
    if hasattr(model, 'random_state'):
        model.random_state = random_state
    if hasattr(model, 'seed'):
        model.seed = random_state
    
    # Обучение
    model.fit(X_train, y_train)
    
    # Предсказания
    y_pred_train = model.predict(X_train)
    
    # Метрики на train
    train_metrics = calculate_regression_metrics(y_train.values, y_pred_train)
    train_metrics = {f'train_{k}': v for k, v in train_metrics.items()}
    
    # Метрики на validation
    val_metrics = {}
    y_pred_val = None
    if X_val is not None and y_val is not None:
        y_pred_val = model.predict(X_val)
        val_metrics = calculate_regression_metrics(y_val.values, y_pred_val)
        val_metrics = {f'val_{k}': v for k, v in val_metrics.items()}
    
    # Объединяем метрики
    all_metrics = {**train_metrics, **val_metrics}
    
    return model, all_metrics, y_pred_train, y_pred_val


def create_default_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Создает словарь моделей по умолчанию.
    
    Args:
        random_state: Для воспроизводимости
        
    Returns:
        Словарь {название: модель}
    """
    models = {
        'LightGBM': lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=7,
            random_state=random_state,
            verbose=-1,
            n_jobs=-1
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=7,
            random_state=random_state,
            n_jobs=-1
        ),
        'RandomForest': RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1
        )
    }
    
    return models


def train_models_with_cv(
    models_dict: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_strategy: Optional[TimeSeriesSplit] = None,
    random_state: int = 42
) -> Dict[str, Dict]:
    """
    Обучение нескольких моделей с кросс-валидацией.
    
    Args:
        models_dict: Словарь {название: модель}
        X_train: Обучающие фичи
        y_train: Обучающая цель
        cv_strategy: Стратегия кросс-валидации (по умолчанию TimeSeriesSplit)
        random_state: Для воспроизводимости
        
    Returns:
        Словарь с результатами для каждой модели
    """
    if cv_strategy is None:
        cv_strategy = TimeSeriesSplit(n_splits=3)
    
    results = {}
    
    for model_name, model in models_dict.items():
        print(f"\n🔄 Обучение {model_name}...")
        
        cv_scores = {'SMAPE': [], 'RMSE': [], 'MAE': [], 'R²': []}
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv_strategy.split(X_train), 1):
            # Разделение на fold train и fold val
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]
            
            # Обучение
            model_copy = type(model)(**model.get_params()) if hasattr(model, 'get_params') else model
            model_copy.fit(X_fold_train, y_fold_train)
            
            # Предсказания
            y_pred = model_copy.predict(X_fold_val)
            
            # Метрики
            metrics = calculate_regression_metrics(y_fold_val.values, y_pred)
            for metric_name, value in metrics.items():
                cv_scores[metric_name].append(value)
        
        # Средние метрики по фолдам
        avg_metrics = {metric: np.mean(scores) for metric, scores in cv_scores.items()}
        std_metrics = {f'{metric}_std': np.std(scores) for metric, scores in cv_scores.items()}
        
        results[model_name] = {
            'metrics': {**avg_metrics, **std_metrics},
            'cv_scores': cv_scores
        }
        
        print(f"   SMAPE: {avg_metrics['SMAPE']:.4f} ± {std_metrics['SMAPE_std']:.4f}")
    
    return results


# ============================================================================
# Сравнение и анализ
# ============================================================================

def compare_models(results_dict: Dict[str, Dict]) -> pd.DataFrame:
    """
    Создает сводную таблицу для сравнения моделей.
    
    Args:
        results_dict: Словарь с результатами моделей
        
    Returns:
        DataFrame с метриками для всех моделей
    """
    comparison_data = []
    
    for model_name, result in results_dict.items():
        metrics = result.get('metrics', {})
        row = {'Model': model_name}
        row.update(metrics)
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # Сортировка по SMAPE (если есть)
    if 'SMAPE' in df.columns:
        df = df.sort_values('SMAPE')
    
    return df


def get_feature_importance(model: Any, feature_names: List[str], top_n: int = 20) -> pd.DataFrame:
    """
    Извлекает важность фичей из обученной модели.
    
    Args:
        model: Обученная модель
        feature_names: Список названий фичей
        top_n: Количество топ фичей
        
    Returns:
        DataFrame с важностью фичей
    """
    importance_dict = {}
    
    # Разные способы извлечения важности для разных моделей
    if hasattr(model, 'feature_importances_'):
        importance_values = model.feature_importances_
    elif hasattr(model, 'get_feature_importance'):
        importance_values = model.get_feature_importance()
    else:
        return pd.DataFrame({'feature': feature_names, 'importance': [0] * len(feature_names)})
    
    # Создаем DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_values
    }).sort_values('importance', ascending=False).head(top_n)
    
    return importance_df


# ============================================================================
# Сохранение модели
# ============================================================================

def save_model(
    model: Any,
    filepath: Path,
    metrics: Optional[Dict[str, float]] = None,
    feature_names: Optional[List[str]] = None
):
    """
    Сохраняет модель и метаданные.
    
    Args:
        model: Обученная модель
        filepath: Путь для сохранения (.joblib файл)
        metrics: Метрики модели (опционально)
        feature_names: Список фичей (опционально)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Сохранение модели
    joblib.dump(model, filepath)
    
    # Сохранение метаданных
    metadata = {}
    if metrics is not None:
        metadata['metrics'] = metrics
    if feature_names is not None:
        metadata['feature_names'] = feature_names
    
    if metadata:
        metadata_path = filepath.with_suffix('.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Модель сохранена: {filepath}")
    if metadata:
        print(f"   Метаданные: {metadata_path}")


def load_model(filepath: Path) -> Tuple[Any, Optional[Dict]]:
    """
    Загружает модель и метаданные.
    
    Args:
        filepath: Путь к файлу модели
        
    Returns:
        Tuple с (модель, метаданные)
    """
    filepath = Path(filepath)
    
    # Загрузка модели
    model = joblib.load(filepath)
    
    # Загрузка метаданных (если есть)
    metadata_path = filepath.with_suffix('.json')
    metadata = None
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    
    return model, metadata

