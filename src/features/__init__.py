"""
Feature Engineering Module

Модуль для создания фичей из временных рядов продаж.
Все фичи создаются с правильной обработкой data leakage.
"""

from .build_features import (
    clean_data,
    create_temporal_features,
    create_lag_features,
    create_rolling_features,
    create_ewma_features,
    create_trend_features,
    create_fourier_features,
    create_aggregated_features,
    create_ratio_features,
    create_calendar_features,
    create_interaction_features,
    create_polynomial_features,
    create_seasonal_decomposition_features,
    build_all_features,
    build_features_for_test,
    get_feature_list,
)

from .validation import (
    analyze_correlations,
    find_redundant_features,
    analyze_feature_importance,
    validate_data_leakage,
    get_feature_correlations_with_target,
)

__all__ = [
    'clean_data',
    'create_temporal_features',
    'create_lag_features',
    'create_rolling_features',
    'create_ewma_features',
    'create_trend_features',
    'create_fourier_features',
    'create_aggregated_features',
    'create_ratio_features',
    'create_calendar_features',
    'create_interaction_features',
    'create_polynomial_features',
    'create_seasonal_decomposition_features',
    'build_all_features',
    'build_features_for_test',
    'get_feature_list',
    'analyze_correlations',
    'find_redundant_features',
    'analyze_feature_importance',
    'validate_data_leakage',
    'get_feature_correlations_with_target',
]

