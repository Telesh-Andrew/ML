"""
Метрики для оценки моделей прогнозирования.

Основная метрика: SMAPE (Symmetric Mean Absolute Percentage Error)
"""

import numpy as np
from typing import Union
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Вычисляет SMAPE (Symmetric Mean Absolute Percentage Error).
    
    SMAPE = (100% / N) * Σ |y_true - y_pred| / ((|y_true| + |y_pred|) / 2)
    
    Особые правила:
    - Если actual = 0 и prediction = 0, то вклад в SMAPE = 0
    
    Args:
        y_true: Реальные значения (1D array)
        y_pred: Предсказанные значения (1D array)
        
    Returns:
        SMAPE значение (в процентах, обычно от 0 до 100+)
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    # Проверка на одинаковую длину
    if len(y_true) != len(y_pred):
        raise ValueError(f"Длины массивов не совпадают: {len(y_true)} != {len(y_pred)}")
    
    # Вычисляем знаменатель: (|y_true| + |y_pred|) / 2
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    
    # Вычисляем числитель: |y_true - y_pred|
    numerator = np.abs(y_true - y_pred)
    
    # Обработка случая, когда denominator = 0 (оба значения равны 0)
    # В этом случае вклад в SMAPE = 0 (согласно правилам)
    mask = denominator > 0
    if not mask.any():
        # Все значения равны 0
        return 0.0
    
    # Вычисляем SMAPE только для ненулевых знаменателей
    smape_values = np.where(mask, numerator / denominator, 0.0)
    
    # Среднее значение и умножение на 100%
    smape_score = np.mean(smape_values) * 100.0
    
    return smape_score


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Вычисляет набор метрик для регрессии.
    
    Args:
        y_true: Реальные значения
        y_pred: Предсказанные значения
        
    Returns:
        Словарь с метриками: SMAPE, RMSE, MAE, R²
    """
    metrics = {
        'SMAPE': smape(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R²': r2_score(y_true, y_pred)
    }
    
    return metrics


def smape_scorer(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Scorer функция для использования в sklearn (например, в GridSearchCV).
    
    Возвращает отрицательное значение SMAPE, так как sklearn максимизирует score.
    Для минимизации SMAPE используем отрицательное значение.
    
    Args:
        y_true: Реальные значения
        y_pred: Предсказанные значения
        
    Returns:
        -SMAPE (для минимизации через максимизацию)
    """
    return -smape(y_true, y_pred)

