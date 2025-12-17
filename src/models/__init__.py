"""
Models Module

Модуль для обучения и оценки моделей прогнозирования продаж.
"""

from .metrics import smape, calculate_regression_metrics, smape_scorer
from .train import (
    prepare_data_for_training,
    ARIMABaseline,
    train_model,
    create_default_models,
    train_models_with_cv,
    compare_models,
    get_feature_importance,
    save_model,
    load_model
)

__all__ = [
    # Метрики
    'smape',
    'calculate_regression_metrics',
    'smape_scorer',
    
    # Подготовка данных
    'prepare_data_for_training',
    
    # Baseline
    'ARIMABaseline',
    
    # Обучение
    'train_model',
    'create_default_models',
    'train_models_with_cv',
    
    # Анализ
    'compare_models',
    'get_feature_importance',
    
    # Сохранение/загрузка
    'save_model',
    'load_model'
]

