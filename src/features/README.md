# Feature Engineering Module

Модуль для создания фичей из временных рядов продаж с правильной обработкой data leakage.

## 🎯 Основные принципы

1. **Нет Data Leakage**: Все лаги и rolling features используют только прошлые данные (`shift(1)`)
2. **Правильная группировка**: Все временные фичи создаются внутри `groupby(['store', 'item'])`
3. **Правильная сортировка**: Данные должны быть отсортированы по `(store, item, date)` перед созданием фичей

## 📦 Структура

```
src/features/
├── __init__.py          # Экспорт функций
├── build_features.py    # Основной модуль с функциями создания фичей
├── validation.py        # Валидационные утилиты (НОВОЕ)
├── example_usage.py     # Пример использования
└── README.md           # Эта документация
```

## 🚀 Быстрый старт

### Базовое использование

```python
from src.data.load_data import load_train, load_test
from src.features.build_features import build_all_features, build_features_for_test

# Загрузка данных
train = load_train()
test = load_test()

# Создание фичей для train (теперь включает новые группы!)
train_features = build_all_features(
    train,
    feature_groups=None,  # Все группы фичей (включая interactions, polynomial, seasonal_decomp)
    lag_periods=[1, 7, 14, 30, 90, 365],
    rolling_windows=[7, 30],
    verbose=True
)

# Создание фичей для test (использует train для лагов)
test_features = build_features_for_test(
    train,
    test,
    feature_groups=None,
    lag_periods=[1, 7, 14, 30, 90, 365],
    rolling_windows=[7, 30],
    verbose=True
)
```

### Создание только базовых фичей

```python
# Только временные фичи и лаги
train_features = build_all_features(
    train,
    feature_groups=['temporal', 'lags', 'rolling'],
    verbose=True
)
```

## 📊 Группы фичей

### 1. Temporal Features (Временные)
- `year`, `month`, `week`, `day_of_week`, `day_of_month`, `day_of_year`, `quarter`
- `is_month_start`, `is_month_end`, `is_quarter_start`, `is_quarter_end`
- `is_weekend`, `days_to_month_end`, `days_to_quarter_end`, `days_to_year_end`

### 2. Lag Features (Лаги)
- `sales_lag_1`, `sales_lag_7`, `sales_lag_14`, `sales_lag_30`, `sales_lag_90`, `sales_lag_365`

### 3. Rolling Features (Скользящие статистики) - УЛУЧШЕНО
- `rolling_mean_7`, `rolling_mean_30`
- `rolling_std_7`, `rolling_std_30`
- `rolling_min_7`, `rolling_max_7`, `rolling_median_7`, `rolling_median_30`
- `rolling_q25_7`, `rolling_q75_7`, `rolling_q25_30`, `rolling_q75_30`
- `rolling_cv_7`, `rolling_cv_30` (коэффициент вариации) - НОВОЕ
- `rolling_skew_7`, `rolling_skew_30` (асимметрия) - НОВОЕ
- `rolling_kurt_7`, `rolling_kurt_30` (эксцесс) - НОВОЕ

### 4. EWMA Features (Экспоненциально взвешенные средние)
- `ewma_7`, `ewma_30`, `ewma_365`

### 5. Trend Features (Тренды) - РАСШИРЕНО
- `diff_1`, `diff_7`, `diff_30` (абсолютные изменения между прошлыми значениями)
- `pct_change_7`, `pct_change_30` (процентные изменения между прошлыми значениями)
- `lag_diff_7_1`, `lag_diff_30_7`, `lag_diff_90_30` (разницы между лагами) - НОВОЕ
- `rolling_diff_mean_30_7`, `rolling_diff_std_30_7` (разницы между rolling) - НОВОЕ

**ВАЖНО**: Все diff и pct_change вычисляются на основе прошлых значений (lagged), а не текущего sales, чтобы предотвратить data leakage.

### 6. Fourier Features (Циклические)
- `sin_month`, `cos_month`
- `sin_week`, `cos_week`
- `sin_day_of_year`, `cos_day_of_year`

### 7. Aggregated Features (Агрегаты)
- `mean_sales_by_store`, `mean_sales_by_item`, `mean_sales_by_store_item`
- `std_sales_by_store`, `std_sales_by_item`
- `max_sales_by_store`, `max_sales_by_item`

### 8. Ratio Features (Отношения)
- `sales_to_store_mean`, `sales_to_item_mean`, `sales_to_store_item_mean`
- `sales_to_rolling_mean_7`, `sales_to_rolling_mean_30`

**ВАЖНО**: Все ratio features используют `sales_lag_1` вместо текущего `sales` для предотвращения target leakage.

### 9. Calendar Features (Календарные)
- `sin_quarter`, `cos_quarter` (циклические фичи для кварталов)
- `half_year`, `sin_half_year`, `cos_half_year` (полугодие)

### 10. Interaction Features (Взаимодействия) - НОВОЕ
- `store_month_mean`, `store_month_std` (средние и std для store × month)
- `item_month_mean`, `item_month_std` (средние и std для item × month)
- `store_dow_mean`, `store_dow_std` (средние и std для store × day_of_week)
- `item_dow_mean`, `item_dow_std` (средние и std для item × day_of_week)

### 11. Polynomial Features (Полиномиальные) - НОВОЕ
- `{feature}_squared`, `{feature}_sqrt` (квадрат и квадратный корень для топ-фичей)
- `lag_7_x_rolling_mean_7` (взаимодействие между фичами)
- `lag_7_x_store_mean` (взаимодействие с aggregated features)

### 12. Seasonal Decomposition Features (Сезонное разложение) - НОВОЕ
- `trend_30` (тренд - скользящее среднее за 30 дней)
- `seasonal_component` (сезонная компонента)
- `trend_slope` (наклон тренда)

**ВАЖНО**: `residual` удален, так как требует текущий `sales` (target leakage).

### 14. Advanced Rolling Statistics - УЛУЧШЕНО
- `rolling_cv_7`, `rolling_cv_30` (коэффициент вариации)
- `rolling_skew_7`, `rolling_skew_30` (асимметрия)
- `rolling_kurt_7`, `rolling_kurt_30` (эксцесс)

## ⚠️ Важные моменты

### Data Leakage Prevention

Все функции автоматически предотвращают data leakage:

```python
# ❌ НЕПРАВИЛЬНО (data leakage):
df['rolling_mean'] = df.groupby(['store', 'item'])['sales'].rolling(7).mean()
df['ratio'] = df['sales'] / df['mean_sales']  # Использует текущий sales!
df['diff'] = df['sales'] - df['sales_lag_1']  # Использует текущий sales!

# ✅ ПРАВИЛЬНО (используется в модуле):
df['rolling_mean'] = (
    df.groupby(['store', 'item'])['sales']
    .shift(1)  # Сдвиг на 1 день назад
    .rolling(7)
    .mean()
)
df['ratio'] = df['sales_lag_1'] / df['mean_sales']  # Использует lagged sales
df['diff'] = df['sales_lag_1'] - df['sales_lag_2']  # Использует lagged sales
```

**Критические исправления (после аудита)**:
- ✅ Ratio features теперь используют `sales_lag_1` вместо текущего `sales`
- ✅ Trend features (diff, pct_change) теперь используют lagged sales
- ✅ Residual из seasonal decomposition удален (требовал текущий sales)

### Обработка test данных

Для test данных нужны исторические данные из train для создания лагов:

```python
# Используйте build_features_for_test вместо build_all_features
test_features = build_features_for_test(train, test, ...)
```

Эта функция:
1. Объединяет train и test
2. Создает фичи на объединенном датасете
3. Возвращает только test с фичами

### Пропуски в начале временных рядов

При создании лагов (особенно `lag_365`) первые строки будут иметь `NaN`. Это нормально:
- Можно удалить строки с `NaN` в критичных лагах
- Или заполнить средним значением по store-item паре
- Или использовать `min_periods=1` в rolling features

## 🔧 Функции модуля

### Основные функции

- `clean_data(df)` - Очистка и предобработка данных
- `build_all_features(df, ...)` - Создание всех фичей для train
- `build_features_for_test(train_df, test_df, ...)` - Создание фичей для test
- `get_feature_list(df)` - Получить список всех feature колонок

### Функции создания отдельных групп фичей

- `create_temporal_features(df)` - Временные фичи
- `create_lag_features(df, lag_periods)` - Lag features
- `create_rolling_features(df, windows, stats)` - Rolling statistics
- `create_ewma_features(df, spans)` - EWMA features
- `create_trend_features(df)` - Trend features
- `create_fourier_features(df)` - Fourier features
- `create_aggregated_features(df)` - Aggregated features
- `create_ratio_features(df)` - Ratio features
- `create_calendar_features(df)` - Calendar features (реализовано: циклические фичи)
- `create_interaction_features(df)` - Interaction features (НОВОЕ)
- `create_polynomial_features(df, top_features)` - Polynomial features (НОВОЕ)
- `create_seasonal_decomposition_features(df)` - Seasonal decomposition (НОВОЕ)

### Валидационные утилиты (НОВОЕ)

- `analyze_correlations(df, target, threshold)` - Анализ корреляций между фичами
- `find_redundant_features(df, target, corr_threshold)` - Поиск избыточных фичей
- `analyze_feature_importance(model, feature_names)` - Анализ важности фичей
- `validate_data_leakage(train_df, test_df)` - Проверка на data leakage
- `get_feature_correlations_with_target(df, target, top_n)` - Топ-фичи по корреляции

## 📝 Примеры использования

### Пример 1: Минимальный набор фичей

```python
train_features = build_all_features(
    train,
    feature_groups=['temporal', 'lags', 'rolling'],
    lag_periods=[7, 30, 365],
    rolling_windows=[7, 30],
    verbose=True
)
```

### Пример 2: Все фичи (включая новые)

```python
train_features = build_all_features(
    train,
    feature_groups=None,  # Все группы (включая interactions, polynomial, seasonal_decomp)
    lag_periods=[1, 7, 14, 30, 90, 365],
    rolling_windows=[7, 30],
    verbose=True
)
```

### Пример 3: Использование валидационных утилит

```python
from src.features.validation import (
    get_feature_correlations_with_target,
    analyze_correlations,
    find_redundant_features
)

# Топ-20 фичей по корреляции с sales
top_features = get_feature_correlations_with_target(
    train_features, target='sales', top_n=20
)

# Анализ высоких корреляций между фичами
high_corr = analyze_correlations(
    train_features, target='sales', threshold=0.95
)

# Поиск избыточных фичей для удаления
redundant = find_redundant_features(
    train_features, target='sales', corr_threshold=0.98
)
```

### Пример 3: Сохранение фичей

```python
import pandas as pd
from pathlib import Path

# Создание фичей
train_features = build_all_features(train, verbose=True)
test_features = build_features_for_test(train, test, verbose=True)

# Сохранение
output_dir = Path('data/processed')
output_dir.mkdir(parents=True, exist_ok=True)

from src.data.save_data import save_dataframe

save_dataframe(train_features, output_dir / 'train_features.csv')
save_dataframe(test_features, output_dir / 'test_features.csv')
```

## 🐛 Troubleshooting

### Проблема: Много NaN в фичах

**Причина**: Лаги и rolling features создают NaN в начале временных рядов.

**Решение**: 
- Удалить строки с NaN: `df = df.dropna(subset=['sales_lag_365'])`
- Или заполнить: `df = df.fillna(method='bfill')` (не рекомендуется)

### Проблема: Медленная работа

**Причина**: Создание всех фичей для большого датасета может быть медленным.

**Решение**:
- Используйте только нужные группы фичей
- Уменьшите количество lag_periods и rolling_windows
- Используйте `dask` для больших датасетов

### Проблема: Data leakage в валидации

**Причина**: Неправильное использование фичей в cross-validation.

**Решение**:
- Используйте `TimeSeriesSplit` вместо обычного `KFold`
- Убедитесь, что данные отсортированы по дате
- Не используйте будущие данные для создания фичей

## 📚 Дополнительные ресурсы

- [Документация pandas](https://pandas.pydata.org/docs/)
- [Time Series Feature Engineering](https://www.kaggle.com/learn/time-series)
- [Preventing Data Leakage](https://www.kaggle.com/code/alexisbcook/data-leakage)

