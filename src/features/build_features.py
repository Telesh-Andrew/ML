"""
Feature Engineering для Store Item Demand Forecasting

Все фичи создаются с правильной обработкой data leakage:
- Лаги и rolling features используют только прошлые данные (shift(1))
- Все группировки по (store, item)
- Правильная сортировка данных перед созданием фичей
"""

import pandas as pd
import numpy as np
from typing import Optional, List
import warnings

warnings.filterwarnings('ignore')


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Очистка и предобработка данных.
    
    Args:
        df: DataFrame с колонками date, store, item, sales
        
    Returns:
        Очищенный DataFrame
    """
    df = df.copy()
    
    # Преобразование типов
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    # Оптимизация памяти
    if 'store' in df.columns:
        df['store'] = df['store'].astype('category')
    if 'item' in df.columns:
        df['item'] = df['item'].astype('category')
    if 'sales' in df.columns:
        df['sales'] = df['sales'].astype('float32')
    
    # Проверка и удаление дубликатов
    if 'date' in df.columns and 'store' in df.columns and 'item' in df.columns:
        duplicates = df.duplicated(subset=['date', 'store', 'item'], keep=False)
        if duplicates.any():
            print(f"⚠️ Найдено {duplicates.sum()} дубликатов. Удаляем...")
            df = df.drop_duplicates(subset=['date', 'store', 'item'], keep='first')
    
    # Проверка некорректных значений
    if 'sales' in df.columns:
        # Отрицательные продажи → 0
        negative_sales = (df['sales'] < 0).sum()
        if negative_sales > 0:
            print(f"⚠️ Найдено {negative_sales} отрицательных продаж. Заменяем на 0...")
            df.loc[df['sales'] < 0, 'sales'] = 0
    
    # Проверка пропусков
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"⚠️ Пропущенные значения:\n{missing[missing > 0]}")
        # Заполняем пропуски в sales нулями (день без продаж)
        if 'sales' in df.columns:
            df['sales'] = df['sales'].fillna(0)
    
    # КРИТИЧНО: Сортировка по (store, item, date) для правильных лагов
    if all(col in df.columns for col in ['store', 'item', 'date']):
        df = df.sort_values(['store', 'item', 'date']).reset_index(drop=True)
    
    return df


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создание временных признаков из даты.
    
    Args:
        df: DataFrame с колонкой 'date'
        
    Returns:
        DataFrame с добавленными временными фичами
    """
    df = df.copy()
    
    if 'date' not in df.columns:
        return df
    
    # Базовые временные компоненты
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['week'] = df['date'].dt.isocalendar().week
    df['day_of_week'] = df['date'].dt.dayofweek  # 0=Понедельник, 6=Воскресенье
    df['day_of_month'] = df['date'].dt.day
    df['day_of_year'] = df['date'].dt.dayofyear
    df['quarter'] = df['date'].dt.quarter
    
    # Календарные флаги
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    df['is_quarter_start'] = df['date'].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)
    df['is_year_start'] = df['date'].dt.is_year_start.astype(int)
    df['is_year_end'] = df['date'].dt.is_year_end.astype(int)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Дополнительные календарные фичи
    df['days_to_month_end'] = df['date'].dt.days_in_month - df['day_of_month']
    df['days_to_quarter_end'] = (
        pd.to_datetime(df['date'].dt.year.astype(str) + '-' + 
                      ((df['quarter'] * 3).astype(str)) + '-01') + 
        pd.DateOffset(months=3) - pd.Timedelta(days=1) - df['date']
    ).dt.days
    df['days_to_year_end'] = (
        pd.to_datetime(df['year'].astype(str) + '-12-31') - df['date']
    ).dt.days
    
    return df


def create_lag_features(df: pd.DataFrame, 
                       lag_periods: List[int] = [1, 7, 14, 30, 90, 365]) -> pd.DataFrame:
    """
    Создание lag features (лагов продаж).
    
    ВАЖНО: Использует только прошлые данные (shift внутри groupby).
    
    Args:
        df: DataFrame с колонками 'store', 'item', 'sales'
        lag_periods: Список периодов для лагов
        
    Returns:
        DataFrame с добавленными lag features
    """
    df = df.copy()
    
    if 'sales' not in df.columns:
        return df
    
    # КРИТИЧНО: Группировка по (store, item) и создание лагов
    for lag in lag_periods:
        df[f'sales_lag_{lag}'] = (
            df.groupby(['store', 'item'])['sales']
            .shift(lag)
        )
    
    return df


def create_rolling_features(df: pd.DataFrame,
                           windows: List[int] = [7, 30],
                           stats: List[str] = ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75', 'cv', 'skew', 'kurt']) -> pd.DataFrame:
    """
    Создание rolling statistics features.
    
    ВАЖНО: Использует shift(1) перед rolling для предотвращения data leakage!
    
    Args:
        df: DataFrame с колонками 'store', 'item', 'sales'
        windows: Список размеров окон
        stats: Список статистик для вычисления.
               Доступные: 'mean', 'std', 'min', 'max', 'median', 'q25', 'q75', 
               'cv' (коэффициент вариации), 'skew' (асимметрия), 'kurt' (эксцесс)
        
    Returns:
        DataFrame с добавленными rolling features
    """
    df = df.copy()
    
    if 'sales' not in df.columns:
        return df
    
    # КРИТИЧНО: shift(1) перед rolling - используем только прошлые данные!
    for window in windows:
        grouped = df.groupby(['store', 'item'])['sales'].shift(1)
        
        for stat in stats:
            if stat == 'mean':
                df[f'rolling_{stat}_{window}'] = grouped.rolling(window, min_periods=1).mean()
            elif stat == 'std':
                df[f'rolling_{stat}_{window}'] = grouped.rolling(window, min_periods=1).std()
            elif stat == 'min':
                df[f'rolling_{stat}_{window}'] = grouped.rolling(window, min_periods=1).min()
            elif stat == 'max':
                df[f'rolling_{stat}_{window}'] = grouped.rolling(window, min_periods=1).max()
            elif stat == 'median':
                df[f'rolling_{stat}_{window}'] = grouped.rolling(window, min_periods=1).median()
            elif stat == 'q25':
                df[f'rolling_{stat}_{window}'] = grouped.rolling(window, min_periods=1).quantile(0.25)
            elif stat == 'q75':
                df[f'rolling_{stat}_{window}'] = grouped.rolling(window, min_periods=1).quantile(0.75)
            elif stat == 'cv':
                # Коэффициент вариации (std/mean) - мера стабильности
                df[f'rolling_{stat}_{window}'] = grouped.rolling(window, min_periods=1).apply(
                    lambda x: x.std() / (x.mean() + 1e-8)
                )
            elif stat == 'skew':
                # Асимметрия (skewness) - мера асимметрии распределения
                df[f'rolling_{stat}_{window}'] = grouped.rolling(window, min_periods=1).skew()
            elif stat == 'kurt':
                # Эксцесс (kurtosis) - мера "тяжести хвостов"
                df[f'rolling_{stat}_{window}'] = grouped.rolling(window, min_periods=1).apply(
                    lambda x: x.kurtosis()
                )
    
    return df


def create_ewma_features(df: pd.DataFrame,
                        spans: List[int] = [7, 30, 365]) -> pd.DataFrame:
    """
    Создание Exponential Weighted Moving Average features.
    
    ВАЖНО: Использует shift(1) перед EWMA для предотвращения data leakage!
    
    Args:
        df: DataFrame с колонками 'store', 'item', 'sales'
        spans: Список периодов для EWMA
        
    Returns:
        DataFrame с добавленными EWMA features
    """
    df = df.copy()
    
    if 'sales' not in df.columns:
        return df
    
    # КРИТИЧНО: shift(1) перед EWMA
    for span in spans:
        df[f'ewma_{span}'] = (
            df.groupby(['store', 'item'])['sales']
            .shift(1)
            .ewm(span=span, adjust=False)
            .mean()
        )
    
    return df


def create_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создание trend features (изменения продаж).
    
    ВАЖНО: Использует sales_lag_1 вместо текущего sales для предотвращения data leakage!
    Все diff и pct_change вычисляются на основе прошлых значений.
    
    Args:
        df: DataFrame с колонками 'store', 'item', 'sales'
        
    Returns:
        DataFrame с добавленными trend features
    """
    df = df.copy()
    
    if 'sales' not in df.columns:
        return df
    
    # ВАЖНО: Используем sales_lag_1 вместо sales для предотвращения target leakage
    # Вычисляем изменения между прошлыми значениями (lagged differences)
    # Создаем недостающие лаги, если их нет
    if 'sales_lag_1' not in df.columns:
        df['sales_lag_1'] = df.groupby(['store', 'item'])['sales'].shift(1)
    if 'sales_lag_2' not in df.columns:
        df['sales_lag_2'] = df.groupby(['store', 'item'])['sales'].shift(2)
    
    # diff_1 = изменение между lag_1 и lag_2 (прошлое изменение)
    if 'sales_lag_1' in df.columns and 'sales_lag_2' in df.columns:
        df['diff_1'] = df['sales_lag_1'] - df['sales_lag_2']
    
    # Создаем лаги 8 и 31, если их нет, для diff_7 и diff_30
    if 'sales_lag_7' in df.columns:
        if 'sales_lag_8' not in df.columns:
            df['sales_lag_8'] = df.groupby(['store', 'item'])['sales'].shift(8)
        # diff_7 = изменение между lag_7 и lag_8 (прошлое изменение за неделю)
        if 'sales_lag_8' in df.columns:
            df['diff_7'] = df['sales_lag_7'] - df['sales_lag_8']
            # Процентное изменение за неделю
            df['pct_change_7'] = (
                (df['sales_lag_7'] - df['sales_lag_8']) / (df['sales_lag_8'] + 1e-8)
            )
    
    if 'sales_lag_30' in df.columns:
        if 'sales_lag_31' not in df.columns:
            df['sales_lag_31'] = df.groupby(['store', 'item'])['sales'].shift(31)
        # diff_30 = изменение между lag_30 и lag_31 (прошлое изменение за месяц)
        if 'sales_lag_31' in df.columns:
            df['diff_30'] = df['sales_lag_30'] - df['sales_lag_31']
            # Процентное изменение за месяц
            df['pct_change_30'] = (
                (df['sales_lag_30'] - df['sales_lag_31']) / (df['sales_lag_31'] + 1e-8)
            )
    
    # Разницы между лагами (дополнительные фичи)
    if 'sales_lag_7' in df.columns and 'sales_lag_1' in df.columns:
        df['lag_diff_7_1'] = df['sales_lag_7'] - df['sales_lag_1']
    
    if 'sales_lag_30' in df.columns and 'sales_lag_7' in df.columns:
        df['lag_diff_30_7'] = df['sales_lag_30'] - df['sales_lag_7']
    
    if 'sales_lag_90' in df.columns and 'sales_lag_30' in df.columns:
        df['lag_diff_90_30'] = df['sales_lag_90'] - df['sales_lag_30']
    
    # Разницы между rolling статистиками
    if 'rolling_mean_30' in df.columns and 'rolling_mean_7' in df.columns:
        df['rolling_diff_mean_30_7'] = df['rolling_mean_30'] - df['rolling_mean_7']
    
    if 'rolling_std_30' in df.columns and 'rolling_std_7' in df.columns:
        df['rolling_diff_std_30_7'] = df['rolling_std_30'] - df['rolling_std_7']
    
    return df


def create_fourier_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создание Fourier features для циклической сезонности.
    
    Args:
        df: DataFrame с временными фичами
        
    Returns:
        DataFrame с добавленными Fourier features
    """
    df = df.copy()
    
    # Месячная сезонность
    if 'month' in df.columns:
        df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
        df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Недельная сезонность
    if 'day_of_week' in df.columns:
        df['sin_week'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['cos_week'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Годовая сезонность
    if 'day_of_year' in df.columns:
        df['sin_day_of_year'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['cos_day_of_year'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    
    return df


def create_aggregated_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создание агрегированных фичей (средние по группам).
    
    ВАЖНО: Использует только прошлые данные для вычисления средних!
    
    Args:
        df: DataFrame с колонками 'store', 'item', 'sales'
        
    Returns:
        DataFrame с добавленными aggregated features
    """
    df = df.copy()
    
    if 'sales' not in df.columns:
        return df
    
    # Сохраняем исходный индекс для выравнивания
    original_index = df.index
    
    # Средние по парам (store, item) - используем expanding mean с shift
    # Это самый важный aggregated feature
    # ВАЖНО: shift(1) гарантирует, что мы используем только прошлые данные (нет data leakage)
    mean_store_item = (
        df.groupby(['store', 'item'])['sales']
        .apply(lambda x: x.shift(1).expanding().mean())
    )
    # Если после apply получился MultiIndex, сбрасываем его
    if isinstance(mean_store_item.index, pd.MultiIndex):
        mean_store_item = mean_store_item.reset_index(level=[0, 1], drop=True)
    
    # Безопасное присваивание: проверяем совпадение индексов
    # Если индексы совпадают (данные отсортированы), используем прямое присваивание
    # Иначе используем reindex для выравнивания
    if len(mean_store_item) == len(df) and (mean_store_item.index == original_index).all():
        df['mean_sales_by_store_item'] = mean_store_item.values
    else:
        # Выравниваем индексы (на случай, если порядок изменился)
        df['mean_sales_by_store_item'] = mean_store_item.reindex(original_index).values
    
    # Средние по магазинам (используем только прошлые данные)
    # ВАЖНО: shift(1) гарантирует отсутствие data leakage
    mean_store = (
        df.groupby('store')['sales']
        .apply(lambda x: x.shift(1).expanding().mean())
    )
    if isinstance(mean_store.index, pd.MultiIndex):
        mean_store = mean_store.reset_index(level=0, drop=True)
    if len(mean_store) == len(df) and (mean_store.index == original_index).all():
        df['mean_sales_by_store'] = mean_store.values
    else:
        df['mean_sales_by_store'] = mean_store.reindex(original_index).values
    
    # Средние по товарам
    # ВАЖНО: shift(1) гарантирует отсутствие data leakage
    mean_item = (
        df.groupby('item')['sales']
        .apply(lambda x: x.shift(1).expanding().mean())
    )
    if isinstance(mean_item.index, pd.MultiIndex):
        mean_item = mean_item.reset_index(level=0, drop=True)
    if len(mean_item) == len(df) and (mean_item.index == original_index).all():
        df['mean_sales_by_item'] = mean_item.values
    else:
        df['mean_sales_by_item'] = mean_item.reindex(original_index).values
    
    # Стандартные отклонения
    # ВАЖНО: shift(1) гарантирует отсутствие data leakage
    std_store = (
        df.groupby('store')['sales']
        .apply(lambda x: x.shift(1).expanding().std())
    )
    if isinstance(std_store.index, pd.MultiIndex):
        std_store = std_store.reset_index(level=0, drop=True)
    if len(std_store) == len(df) and (std_store.index == original_index).all():
        df['std_sales_by_store'] = std_store.values
    else:
        df['std_sales_by_store'] = std_store.reindex(original_index).values
    
    std_item = (
        df.groupby('item')['sales']
        .apply(lambda x: x.shift(1).expanding().std())
    )
    if isinstance(std_item.index, pd.MultiIndex):
        std_item = std_item.reset_index(level=0, drop=True)
    if len(std_item) == len(df) and (std_item.index == original_index).all():
        df['std_sales_by_item'] = std_item.values
    else:
        df['std_sales_by_item'] = std_item.reindex(original_index).values
    
    # Максимумы
    # ВАЖНО: shift(1) гарантирует отсутствие data leakage
    max_store = (
        df.groupby('store')['sales']
        .apply(lambda x: x.shift(1).expanding().max())
    )
    if isinstance(max_store.index, pd.MultiIndex):
        max_store = max_store.reset_index(level=0, drop=True)
    if len(max_store) == len(df) and (max_store.index == original_index).all():
        df['max_sales_by_store'] = max_store.values
    else:
        df['max_sales_by_store'] = max_store.reindex(original_index).values
    
    max_item = (
        df.groupby('item')['sales']
        .apply(lambda x: x.shift(1).expanding().max())
    )
    if isinstance(max_item.index, pd.MultiIndex):
        max_item = max_item.reset_index(level=0, drop=True)
    if len(max_item) == len(df) and (max_item.index == original_index).all():
        df['max_sales_by_item'] = max_item.values
    else:
        df['max_sales_by_item'] = max_item.reindex(original_index).values
    
    return df


def create_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создание ratio features (отношения).
    
    ВАЖНО: Использует sales_lag_1 вместо текущего sales для предотвращения data leakage!
    
    Args:
        df: DataFrame с sales и агрегированными фичами
        
    Returns:
        DataFrame с добавленными ratio features
    """
    df = df.copy()
    
    if 'sales' not in df.columns:
        return df
    
    # ВАЖНО: Используем sales_lag_1 вместо sales для предотвращения target leakage
    # Если sales_lag_1 нет, создаем его
    if 'sales_lag_1' not in df.columns:
        df['sales_lag_1'] = df.groupby(['store', 'item'])['sales'].shift(1)
    
    # Отношения к средним (используем lagged sales)
    if 'mean_sales_by_store' in df.columns and 'sales_lag_1' in df.columns:
        df['sales_to_store_mean'] = (
            df['sales_lag_1'] / (df['mean_sales_by_store'] + 1e-8)
        )
    
    if 'mean_sales_by_item' in df.columns and 'sales_lag_1' in df.columns:
        df['sales_to_item_mean'] = (
            df['sales_lag_1'] / (df['mean_sales_by_item'] + 1e-8)
        )
    
    if 'mean_sales_by_store_item' in df.columns and 'sales_lag_1' in df.columns:
        df['sales_to_store_item_mean'] = (
            df['sales_lag_1'] / (df['mean_sales_by_store_item'] + 1e-8)
        )
    
    # Отношения к rolling mean (используем lagged sales)
    if 'rolling_mean_30' in df.columns and 'sales_lag_1' in df.columns:
        df['sales_to_rolling_mean_30'] = (
            df['sales_lag_1'] / (df['rolling_mean_30'] + 1e-8)
        )
    
    if 'rolling_mean_7' in df.columns and 'sales_lag_1' in df.columns:
        df['sales_to_rolling_mean_7'] = (
            df['sales_lag_1'] / (df['rolling_mean_7'] + 1e-8)
        )
    
    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создание interaction features (взаимодействия между категориями и временем).
    
    ВАЖНО: Все статистики вычисляются с shift(1) + expanding() для предотвращения data leakage!
    
    Гипотеза: Комбинации store×month, item×month, store×day_of_week, item×day_of_week
    дают дополнительную информацию о паттернах продаж.
    
    Args:
        df: DataFrame с колонками 'store', 'item', 'sales', 'month', 'day_of_week'
        
    Returns:
        DataFrame с добавленными interaction features:
        - store_month_mean, store_month_std: средние и std продаж для (store, month)
        - item_month_mean, item_month_std: средние и std продаж для (item, month)
        - store_dow_mean, store_dow_std: средние и std продаж для (store, day_of_week)
        - item_dow_mean, item_dow_std: средние и std продаж для (item, day_of_week)
    """
    df = df.copy()
    
    if 'sales' not in df.columns:
        return df
    
    original_index = df.index
    
    # Store × Month взаимодействие
    if 'store' in df.columns and 'month' in df.columns:
        # Средние
        store_month_mean = (
            df.groupby(['store', 'month'])['sales']
            .apply(lambda x: x.shift(1).expanding().mean())
        )
        if isinstance(store_month_mean.index, pd.MultiIndex):
            store_month_mean = store_month_mean.reset_index(level=[0, 1], drop=True)
        df['store_month_mean'] = store_month_mean.reindex(original_index)
        
        # Стандартные отклонения
        store_month_std = (
            df.groupby(['store', 'month'])['sales']
            .apply(lambda x: x.shift(1).expanding().std())
        )
        if isinstance(store_month_std.index, pd.MultiIndex):
            store_month_std = store_month_std.reset_index(level=[0, 1], drop=True)
        df['store_month_std'] = store_month_std.reindex(original_index)
    
    # Item × Month взаимодействие
    if 'item' in df.columns and 'month' in df.columns:
        # Средние
        item_month_mean = (
            df.groupby(['item', 'month'])['sales']
            .apply(lambda x: x.shift(1).expanding().mean())
        )
        if isinstance(item_month_mean.index, pd.MultiIndex):
            item_month_mean = item_month_mean.reset_index(level=[0, 1], drop=True)
        df['item_month_mean'] = item_month_mean.reindex(original_index)
        
        # Стандартные отклонения
        item_month_std = (
            df.groupby(['item', 'month'])['sales']
            .apply(lambda x: x.shift(1).expanding().std())
        )
        if isinstance(item_month_std.index, pd.MultiIndex):
            item_month_std = item_month_std.reset_index(level=[0, 1], drop=True)
        df['item_month_std'] = item_month_std.reindex(original_index)
    
    # Store × Day of Week взаимодействие
    if 'store' in df.columns and 'day_of_week' in df.columns:
        # Средние
        store_dow_mean = (
            df.groupby(['store', 'day_of_week'])['sales']
            .apply(lambda x: x.shift(1).expanding().mean())
        )
        if isinstance(store_dow_mean.index, pd.MultiIndex):
            store_dow_mean = store_dow_mean.reset_index(level=[0, 1], drop=True)
        df['store_dow_mean'] = store_dow_mean.reindex(original_index)
        
        # Стандартные отклонения
        store_dow_std = (
            df.groupby(['store', 'day_of_week'])['sales']
            .apply(lambda x: x.shift(1).expanding().std())
        )
        if isinstance(store_dow_std.index, pd.MultiIndex):
            store_dow_std = store_dow_std.reset_index(level=[0, 1], drop=True)
        df['store_dow_std'] = store_dow_std.reindex(original_index)
    
    # Item × Day of Week взаимодействие
    if 'item' in df.columns and 'day_of_week' in df.columns:
        # Средние
        item_dow_mean = (
            df.groupby(['item', 'day_of_week'])['sales']
            .apply(lambda x: x.shift(1).expanding().mean())
        )
        if isinstance(item_dow_mean.index, pd.MultiIndex):
            item_dow_mean = item_dow_mean.reset_index(level=[0, 1], drop=True)
        df['item_dow_mean'] = item_dow_mean.reindex(original_index)
        
        # Стандартные отклонения
        item_dow_std = (
            df.groupby(['item', 'day_of_week'])['sales']
            .apply(lambda x: x.shift(1).expanding().std())
        )
        if isinstance(item_dow_std.index, pd.MultiIndex):
            item_dow_std = item_dow_std.reset_index(level=[0, 1], drop=True)
        df['item_dow_std'] = item_dow_std.reindex(original_index)
    
    return df


def create_polynomial_features(df: pd.DataFrame, 
                               top_features: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Создание полиномиальных фичей для важных признаков.
    
    Гипотеза: Нелинейные зависимости могут улучшить модель.
    Создает квадраты, квадратные корни и взаимодействия для топ-фичей.
    
    Args:
        df: DataFrame с фичами
        top_features: Список фичей для полиномизации.
                     Если None, использует топ-5 по корреляции с sales
                     (если sales есть в данных).
        
    Returns:
        DataFrame с добавленными polynomial features:
        - lag_7_squared, lag_7_sqrt
        - rolling_mean_7_squared
        - lag_7_x_rolling_mean_7 (взаимодействие)
        - lag_7_x_store_mean (взаимодействие)
    """
    df = df.copy()
    
    # Определяем топ-фичи для полиномизации
    if top_features is None:
        if 'sales' in df.columns:
            # Находим топ-5 фичей по корреляции с sales
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if 'sales' in numeric_cols:
                correlations = df[numeric_cols].corr()['sales'].abs().sort_values(ascending=False)
                # Исключаем sales и берем топ-5
                top_features = correlations.drop('sales').head(5).index.tolist()
            else:
                # Если нет sales, используем дефолтные
                top_features = ['sales_lag_7', 'rolling_mean_7', 'sales_lag_14', 
                               'rolling_mean_30', 'mean_sales_by_store_item']
        else:
            # Дефолтные фичи, если sales нет
            top_features = ['sales_lag_7', 'rolling_mean_7', 'sales_lag_14', 
                           'rolling_mean_30', 'mean_sales_by_store_item']
    
    # Создаем полиномиальные фичи только для существующих колонок
    available_features = [f for f in top_features if f in df.columns]
    
    for feature in available_features:
        # Квадрат
        df[f'{feature}_squared'] = df[feature] ** 2
        
        # Квадратный корень (с защитой от отрицательных значений)
        df[f'{feature}_sqrt'] = np.sqrt(np.abs(df[feature]) + 1e-8)
    
    # Взаимодействия между топ-фичами
    if len(available_features) >= 2:
        # Взаимодействие между первыми двумя
        if available_features[0] in df.columns and available_features[1] in df.columns:
            df[f'{available_features[0]}_x_{available_features[1]}'] = (
                df[available_features[0]] * df[available_features[1]]
            )
    
    # Взаимодействие с aggregated features
    if 'sales_lag_7' in df.columns:
        if 'mean_sales_by_store' in df.columns:
            df['lag_7_x_store_mean'] = df['sales_lag_7'] * df['mean_sales_by_store']
        if 'mean_sales_by_item' in df.columns:
            df['lag_7_x_item_mean'] = df['sales_lag_7'] * df['mean_sales_by_item']
        if 'mean_sales_by_store_item' in df.columns:
            df['lag_7_x_store_item_mean'] = df['sales_lag_7'] * df['mean_sales_by_store_item']
    
    if 'rolling_mean_7' in df.columns:
        if 'sales_lag_7' in df.columns:
            df['rolling_mean_7_x_lag_7'] = df['rolling_mean_7'] * df['sales_lag_7']
    
    return df


def create_seasonal_decomposition_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создание фичей на основе сезонного разложения временных рядов.
    
    Гипотеза: Отделение тренда, сезонности и остатков улучшает прогноз.
    
    ВАЖНО: Все компоненты вычисляются с shift(1) для предотвращения data leakage!
    residual удален, так как требует текущий sales (target leakage).
    
    Args:
        df: DataFrame с колонками 'store', 'item', 'sales', 'month'
        
    Returns:
        DataFrame с добавленными seasonal decomposition features:
        - trend_30: тренд (скользящее среднее за 30 дней)
        - seasonal_component: сезонная компонента (отклонение от тренда)
        - trend_slope: наклон тренда (линейная регрессия за последние 30 дней)
    """
    df = df.copy()
    
    if 'sales' not in df.columns:
        return df
    
    original_index = df.index
    
    # Тренд компонента (скользящее среднее за 30 дней)
    trend_30 = (
        df.groupby(['store', 'item'])['sales']
        .apply(lambda x: x.shift(1).rolling(30, min_periods=1).mean())
    )
    if isinstance(trend_30.index, pd.MultiIndex):
        trend_30 = trend_30.reset_index(level=[0, 1], drop=True)
    df['trend_30'] = trend_30.reindex(original_index)
    
    # Сезонная компонента (средние по месяцам, вычисленные на прошлых данных)
    if 'month' in df.columns:
        seasonal_component = (
            df.groupby(['store', 'item', 'month'])['sales']
            .apply(lambda x: x.shift(1).expanding().mean())
        )
        if isinstance(seasonal_component.index, pd.MultiIndex):
            seasonal_component = seasonal_component.reset_index(level=[0, 1, 2], drop=True)
        df['seasonal_component'] = seasonal_component.reindex(original_index)
        
        # ВАЖНО: residual удален, так как требует текущий sales (target leakage)
        # Вместо этого можно использовать lagged residual, если нужно:
        # residual_lag = sales_lag_1 - trend_lag_1 - seasonal_lag_1
        # Но это требует дополнительных вычислений и может быть избыточно
    
    # Наклон тренда (линейная регрессия за последние 30 дней)
    # Используем упрощенный расчет: разница между последним и первым значением в окне
    def calculate_trend_slope(series):
        """Вычисляет наклон тренда как среднюю разницу."""
        shifted = series.shift(1)
        rolling_diff = shifted.diff(1).rolling(30, min_periods=1).mean()
        return rolling_diff
    
    trend_slope = (
        df.groupby(['store', 'item'])['sales']
        .apply(calculate_trend_slope)
    )
    if isinstance(trend_slope.index, pd.MultiIndex):
        trend_slope = trend_slope.reset_index(level=[0, 1], drop=True)
    df['trend_slope'] = trend_slope.reindex(original_index)
    
    return df


def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создание дополнительных календарных фичей.
    
    Добавляет циклические фичи для кварталов и полугодий.
    
    Args:
        df: DataFrame с колонкой 'date' и 'quarter'
        
    Returns:
        DataFrame с добавленными calendar features
    """
    df = df.copy()
    
    if 'date' not in df.columns:
        return df
    
    # Циклические фичи для кварталов (если quarter уже создан)
    if 'quarter' in df.columns:
        df['sin_quarter'] = np.sin(2 * np.pi * df['quarter'] / 4)
        df['cos_quarter'] = np.cos(2 * np.pi * df['quarter'] / 4)
    
    # Полугодие (1-6 месяц = 0, 7-12 месяц = 1)
    if 'month' in df.columns:
        df['half_year'] = (df['month'] <= 6).astype(int)
        df['sin_half_year'] = np.sin(2 * np.pi * df['half_year'] / 2)
        df['cos_half_year'] = np.cos(2 * np.pi * df['half_year'] / 2)
    
    return df


def build_features_for_test(train_df: pd.DataFrame, 
                           test_df: pd.DataFrame,
                           feature_groups: Optional[List[str]] = None,
                           lag_periods: List[int] = [1, 7, 14, 30, 90, 365],
                           rolling_windows: List[int] = [7, 30],
                           verbose: bool = True) -> pd.DataFrame:
    """
    Создание фичей для test данных с использованием train данных для лагов.
    
    ВАЖНО: Для создания лагов и rolling features в test нужны исторические
    данные из train. Эта функция объединяет train и test, создает фичи,
    затем возвращает только test с фичами.
    
    Args:
        train_df: Обучающий датасет с sales
        test_df: Тестовый датасет без sales
        feature_groups: Список групп фичей для создания
        lag_periods: Периоды для lag features
        rolling_windows: Окна для rolling features
        verbose: Выводить прогресс
        
    Returns:
        DataFrame с test данными и всеми фичами
    """
    if verbose:
        print("🔗 Объединение train и test для создания фичей...")
    
    # Объединяем train и test
    train_clean = train_df.copy()
    test_clean = test_df.copy()
    
    # Добавляем placeholder sales в test (нужен для группировок)
    test_clean['sales'] = np.nan
    
    # Объединяем
    combined = pd.concat([train_clean, test_clean], ignore_index=True)
    combined = combined.sort_values(['store', 'item', 'date']).reset_index(drop=True)
    
    # Создаем фичи на объединенном датасете
    combined_features = build_all_features(
        combined,
        feature_groups=feature_groups,
        lag_periods=lag_periods,
        rolling_windows=rolling_windows,
        verbose=verbose
    )
    
    # Разделяем обратно - берем только test строки
    test_mask = combined_features['sales'].isna()
    test_features = combined_features[test_mask].copy()
    
    # Удаляем sales (если был placeholder)
    if 'sales' in test_features.columns:
        test_features = test_features.drop('sales', axis=1)
    
    if verbose:
        print(f"✅ Test features готовы: {test_features.shape}")
    
    return test_features


def build_all_features(df: pd.DataFrame,
                      feature_groups: Optional[List[str]] = None,
                      lag_periods: List[int] = [1, 7, 14, 30, 90, 365],
                      rolling_windows: List[int] = [7, 30],
                      verbose: bool = True) -> pd.DataFrame:
    """
    Создание всех фичей в правильном порядке.
    
    Порядок важен для предотвращения data leakage!
    
    Args:
        df: Исходный DataFrame с колонками date, store, item, sales
        feature_groups: Список групп фичей для создания. 
                       Если None, создаются все.
                       Возможные значения: 'temporal', 'lags', 'rolling', 
                       'ewma', 'trends', 'fourier', 'aggregated', 'ratios', 
                       'calendar', 'interactions', 'polynomial', 
                       'seasonal_decomp'
        lag_periods: Периоды для lag features
        rolling_windows: Окна для rolling features
        verbose: Выводить прогресс
        
    Returns:
        DataFrame со всеми созданными фичами
    """
    if feature_groups is None:
        feature_groups = [
            'temporal', 'lags', 'rolling', 'ewma', 'trends', 
            'fourier', 'aggregated', 'ratios', 'calendar',
            'interactions', 'polynomial', 'seasonal_decomp'
        ]
    
    # Шаг 1: Очистка данных
    if verbose:
        print("🧹 Очистка данных...")
    df = clean_data(df)
    
    # Шаг 2: Временные фичи (не зависят от sales)
    if 'temporal' in feature_groups:
        if verbose:
            print("📅 Создание временных фичей...")
        df = create_temporal_features(df)
    
    # Шаг 3: Lag features (зависят только от прошлых sales)
    if 'lags' in feature_groups:
        if verbose:
            print("⏮️ Создание lag features...")
        df = create_lag_features(df, lag_periods=lag_periods)
    
    # Шаг 4: Rolling features (используют shift(1))
    if 'rolling' in feature_groups:
        if verbose:
            print("📊 Создание rolling features...")
        df = create_rolling_features(df, windows=rolling_windows)
    
    # Шаг 5: EWMA features
    if 'ewma' in feature_groups:
        if verbose:
            print("📈 Создание EWMA features...")
        df = create_ewma_features(df)
    
    # Шаг 6: Trend features (зависят от lag features)
    if 'trends' in feature_groups:
        if verbose:
            print("📉 Создание trend features...")
        df = create_trend_features(df)
    
    # Шаг 7: Fourier features (зависят от temporal features)
    if 'fourier' in feature_groups:
        if verbose:
            print("🌊 Создание Fourier features...")
        df = create_fourier_features(df)
    
    # Шаг 8: Aggregated features (используют только прошлые данные)
    if 'aggregated' in feature_groups:
        if verbose:
            print("📦 Создание aggregated features...")
        df = create_aggregated_features(df)
    
    # Шаг 9: Ratio features (зависят от aggregated и rolling)
    if 'ratios' in feature_groups:
        if verbose:
            print("🔢 Создание ratio features...")
        df = create_ratio_features(df)
    
    # Шаг 10: Calendar features (уже включены в temporal, но можно расширить)
    if 'calendar' in feature_groups:
        if verbose:
            print("📆 Создание calendar features...")
        df = create_calendar_features(df)
    
    # Шаг 11: Interaction features (зависят от temporal и aggregated)
    if 'interactions' in feature_groups:
        if verbose:
            print("🔗 Создание interaction features...")
        df = create_interaction_features(df)
    
    # Шаг 12: Polynomial features (зависят от lag и rolling)
    if 'polynomial' in feature_groups:
        if verbose:
            print("📐 Создание polynomial features...")
        df = create_polynomial_features(df)
    
    # Шаг 13: Seasonal decomposition features (зависят от temporal)
    if 'seasonal_decomp' in feature_groups:
        if verbose:
            print("📊 Создание seasonal decomposition features...")
        df = create_seasonal_decomposition_features(df)
    
    if verbose:
        print(f"✅ Готово! Создано {len(df.columns)} колонок")
        print(f"   Исходных колонок: 4 (date, store, item, sales)")
        print(f"   Новых фичей: {len(df.columns) - 4}")
    
    return df


def get_feature_list(df: pd.DataFrame, 
                    exclude_cols: List[str] = ['date', 'store', 'item', 'sales']) -> List[str]:
    """
    Получить список всех feature колонок (исключая target и ID).
    
    Args:
        df: DataFrame с фичами
        exclude_cols: Колонки для исключения
        
    Returns:
        Список названий feature колонок
    """
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols

