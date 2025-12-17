"""
Утилиты для валидации и анализа фичей.

Функции для проверки корреляций, поиска избыточных фичей,
анализа важности фичей и проверки на data leakage.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy.stats import spearmanr


def analyze_correlations(df: pd.DataFrame, 
                        target: str = 'sales',
                        threshold: float = 0.95,
                        method: str = 'pearson') -> pd.DataFrame:
    """
    Анализ корреляций между фичами и целевой переменной.
    Находит пары фичей с высокой корреляцией (>threshold).
    
    Args:
        df: DataFrame с фичами
        target: Название целевой переменной
        threshold: Порог для высокой корреляции (по умолчанию 0.95)
        method: Метод корреляции ('pearson' или 'spearman')
        
    Returns:
        DataFrame с парами фичей и их корреляцией, отсортированный по убыванию
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if target not in numeric_cols:
        print(f"⚠️ Целевая переменная '{target}' не найдена в числовых колонках")
        return pd.DataFrame()
    
    # Удаляем target из списка для анализа корреляций между фичами
    feature_cols = [col for col in numeric_cols if col != target]
    
    if len(feature_cols) < 2:
        print("⚠️ Недостаточно фичей для анализа корреляций")
        return pd.DataFrame()
    
    # Вычисляем корреляционную матрицу
    if method == 'pearson':
        corr_matrix = df[feature_cols].corr().abs()
    elif method == 'spearman':
        # Для spearman нужно вычислить попарно
        corr_pairs = []
        for i, col1 in enumerate(feature_cols):
            for col2 in feature_cols[i+1:]:
                try:
                    corr_val, _ = spearmanr(df[col1].dropna(), df[col2].dropna())
                    if not np.isnan(corr_val):
                        corr_pairs.append({
                            'feature_1': col1,
                            'feature_2': col2,
                            'correlation': abs(corr_val)
                        })
                except:
                    continue
        if corr_pairs:
            result_df = pd.DataFrame(corr_pairs).sort_values('correlation', ascending=False)
            # Фильтруем по threshold
            result_df = result_df[result_df['correlation'] >= threshold]
            return result_df
        else:
            return pd.DataFrame(columns=['feature_1', 'feature_2', 'correlation'])
    else:
        raise ValueError(f"Неизвестный метод корреляции: {method}. Используйте 'pearson' или 'spearman'")
    
    # Находим пары с высокой корреляцией (для pearson)
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if corr_val >= threshold:
                high_corr_pairs.append({
                    'feature_1': corr_matrix.columns[i],
                    'feature_2': corr_matrix.columns[j],
                    'correlation': corr_val
                })
    
    if high_corr_pairs:
        result_df = pd.DataFrame(high_corr_pairs).sort_values('correlation', ascending=False)
        return result_df
    else:
        return pd.DataFrame(columns=['feature_1', 'feature_2', 'correlation'])


def find_redundant_features(df: pd.DataFrame,
                           target: str = 'sales',
                           corr_threshold: float = 0.98,
                           method: str = 'pearson') -> List[str]:
    """
    Находит избыточные фичи (высокая корреляция друг с другом).
    
    Стратегия: Если две фичи имеют корреляцию > threshold, удаляем ту,
    которая имеет меньшую корреляцию с target.
    
    Args:
        df: DataFrame с фичами
        target: Название целевой переменной
        corr_threshold: Порог корреляции для определения избыточности
        method: Метод корреляции
        
    Returns:
        Список фичей для удаления
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if target not in numeric_cols:
        print(f"⚠️ Целевая переменная '{target}' не найдена")
        return []
    
    feature_cols = [col for col in numeric_cols if col != target]
    
    if len(feature_cols) < 2:
        return []
    
    # Находим пары с высокой корреляцией
    high_corr_pairs = analyze_correlations(df, target=target, threshold=corr_threshold, method=method)
    
    if high_corr_pairs.empty:
        return []
    
    # Вычисляем корреляции с target для всех фичей
    if method == 'pearson':
        target_corrs = df[feature_cols].corrwith(df[target]).abs()
    else:
        # Spearman корреляция
        target_corrs_dict = {}
        for col in feature_cols:
            try:
                corr_val, _ = spearmanr(df[col].dropna(), df[target].dropna())
                target_corrs_dict[col] = abs(corr_val) if not np.isnan(corr_val) else 0.0
            except:
                target_corrs_dict[col] = 0.0
        target_corrs = pd.Series(target_corrs_dict)
    
    # Для каждой пары выбираем фичу с меньшей корреляцией с target
    features_to_remove = set()
    
    for _, row in high_corr_pairs.iterrows():
        feat1, feat2 = row['feature_1'], row['feature_2']
        
        if feat1 not in target_corrs.index or feat2 not in target_corrs.index:
            continue
        
        corr1 = target_corrs[feat1]
        corr2 = target_corrs[feat2]
        
        # Удаляем фичу с меньшей корреляцией с target
        if corr1 < corr2:
            features_to_remove.add(feat1)
        else:
            features_to_remove.add(feat2)
    
    return sorted(list(features_to_remove))


def analyze_feature_importance(model, 
                              feature_names: List[str],
                              top_n: int = 20) -> pd.DataFrame:
    """
    Анализ важности фичей после обучения модели.
    
    Поддерживает модели XGBoost, LightGBM, Random Forest, CatBoost.
    
    Args:
        model: Обученная модель (должна иметь feature_importances_)
        feature_names: Список названий фичей
        top_n: Количество топ-фичей для вывода
        
    Returns:
        DataFrame с фичами и их важностью, отсортированный по убыванию
    """
    if not hasattr(model, 'feature_importances_'):
        print("⚠️ Модель не имеет атрибута feature_importances_")
        return pd.DataFrame()
    
    if len(feature_names) != len(model.feature_importances_):
        print(f"⚠️ Количество фичей ({len(feature_names)}) не совпадает с "
              f"количеством важностей ({len(model.feature_importances_)})")
        return pd.DataFrame()
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Нормализуем важность (сумма = 1)
    importance_df['importance_normalized'] = (
        importance_df['importance'] / importance_df['importance'].sum()
    )
    
    # Кумулятивная важность
    importance_df['cumulative_importance'] = (
        importance_df['importance_normalized'].cumsum()
    )
    
    return importance_df.head(top_n)


def validate_data_leakage(train_df: pd.DataFrame,
                         test_df: pd.DataFrame,
                         target: str = 'sales') -> Dict[str, bool]:
    """
    Проверка на data leakage между train и test.
    
    Проверяет:
    1. Нет ли пересечений по датам
    2. Нет ли одинаковых значений в фичах (может указывать на leakage)
    3. Правильность временного порядка
    
    Args:
        train_df: Обучающий датасет
        test_df: Тестовый датасет
        target: Название целевой переменной
        
    Returns:
        Словарь с результатами проверки
    """
    results = {
        'date_overlap': False,
        'date_order_correct': True,
        'has_target_in_test': False,
        'warnings': []
    }
    
    # Проверка 1: Пересечение дат
    if 'date' in train_df.columns and 'date' in test_df.columns:
        train_dates = set(train_df['date'].unique())
        test_dates = set(test_df['date'].unique())
        
        overlap = train_dates.intersection(test_dates)
        if overlap:
            results['date_overlap'] = True
            results['warnings'].append(
                f"⚠️ Найдено {len(overlap)} пересекающихся дат между train и test!"
            )
        else:
            results['warnings'].append("✅ Нет пересечений по датам")
        
        # Проверка 2: Временной порядок
        if train_df['date'].max() > test_df['date'].min():
            results['date_order_correct'] = False
            results['warnings'].append(
                "⚠️ Train содержит даты позже, чем test! Возможна проблема с разделением."
            )
        else:
            results['warnings'].append("✅ Временной порядок корректен")
    
    # Проверка 3: Наличие target в test
    if target in test_df.columns:
        if not test_df[target].isna().all():
            results['has_target_in_test'] = True
            results['warnings'].append(
                f"⚠️ В test есть значения в колонке '{target}'!"
            )
        else:
            results['warnings'].append(f"✅ Колонка '{target}' в test пустая (как и должно быть)")
    
    return results


def get_feature_correlations_with_target(df: pd.DataFrame,
                                       target: str = 'sales',
                                       top_n: int = 20,
                                       method: str = 'pearson') -> pd.DataFrame:
    """
    Получить топ-N фичей по корреляции с целевой переменной.
    
    Args:
        df: DataFrame с фичами
        target: Название целевой переменной
        top_n: Количество топ-фичей
        method: Метод корреляции
        
    Returns:
        DataFrame с фичами и их корреляцией с target
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if target not in numeric_cols:
        print(f"⚠️ Целевая переменная '{target}' не найдена")
        return pd.DataFrame()
    
    feature_cols = [col for col in numeric_cols if col != target]
    
    if method == 'pearson':
        correlations = df[feature_cols].corrwith(df[target]).abs().sort_values(ascending=False)
    else:
        correlations = pd.Series({
            col: abs(spearmanr(df[col], df[target])[0])
            for col in feature_cols
        }).sort_values(ascending=False)
    
    result_df = pd.DataFrame({
        'feature': correlations.index,
        'correlation': correlations.values
    }).head(top_n)
    
    return result_df

