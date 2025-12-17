# System Patterns: Architecture & Design Decisions

## Общая Архитектура

### High-Level View
```
Raw Data → Preprocessing → Feature Engineering → Model Training → Inference → Predictions
    ↓           ↓                  ↓                   ↓              ↓          ↓
  CSV files   cleaned data    engineered features   saved model    API     submission
```

### Компонентная Структура
```
├── Data Layer
│   ├── Raw data loading
│   ├── Data validation
│   └── Train/test splitting
│
├── Feature Layer
│   ├── Temporal features (day, week, month, year)
│   ├── Lag features (sales_lag_7, sales_lag_30)
│   ├── Rolling statistics (mean, std, min, max)
│   └── Categorical encoding (store, item)
│
├── Model Layer
│   ├── Baseline models (naive, moving average)
│   ├── Classical ML (Linear Regression, Random Forest)
│   ├── Gradient Boosting (XGBoost, LightGBM, CatBoost)
│   └── Deep Learning (LSTM, GRU - опционально)
│
├── Inference Layer
│   ├── Model loading
│   ├── Feature transformation pipeline
│   └── Prediction generation
│
└── API Layer (FastAPI)
    ├── /predict endpoint
    ├── /batch_predict endpoint
    └── /health endpoint
```

## Ключевые Паттерны

### 1. Iterative Development Pattern
**Философия**: От простого к сложному, каждая итерация должна работать end-to-end.

**Iteration 0 - Baseline**:
- Naive forecast (прошлогоднее значение)
- Moving average (последние 30 дней)
- Цель: установить нижнюю границу качества (~20-25% SMAPE)

**Iteration 1 - Feature Engineering + Classical ML**:
- Временные признаки
- Лаги и rolling windows
- Linear Regression / Random Forest
- Цель: SMAPE ~18-20%

**Iteration 2 - Gradient Boosting**:
- XGBoost / LightGBM с оптимизированными гиперпараметрами
- Cross-validation по времени
- Цель: SMAPE ~13-15%

**Iteration 3 - Advanced Methods**:
- Ансамбли моделей
- Глубокое обучение (LSTM/GRU)
- Цель: SMAPE < 13%

### 2. Time Series Split Pattern
```python
# НЕ использовать обычный K-Fold!
# Временные ряды требуют forward-chaining CV:

Train: [2013-2014-2015-2016] → Validate: [2017]
Train: [2013-2014-2015-2016-2017] → Test: [2018]
```

### 3. Feature Store Pattern
Централизованное хранилище фичей:
- `data/processed/features_train.csv`
- `data/processed/features_test.csv`
- Версионирование фичей
- Reproducibility через сохранение трансформеров

### 4. Pipeline Pattern
Все трансформации данных в единый sklearn Pipeline:
```python
pipeline = Pipeline([
    ('imputer', SimpleImputer()),
    ('scaler', StandardScaler()),
    ('model', XGBRegressor())
])
```
Преимущества:
- Нет data leakage
- Легко сериализовать
- Простое применение к новым данным

### 5. Model Registry Pattern
Структура для сохранения артефактов:
```
artifacts/
├── models/
│   ├── baseline_v1.pkl
│   ├── xgboost_v2.pkl
│   └── best_model.pkl
├── features/
│   └── feature_transformer.pkl
└── metadata/
    └── model_config.json
```

## Архитектурные Решения

### Global vs Local Models
**Решение**: Начать с **Global Model** (одна модель для всех store-item пар)

**Обоснование**:
- ✅ Проще в разработке и поддержке
- ✅ Лучше обобщается при малом количестве данных
- ✅ store и item используются как categorical features
- ❌ Может упустить специфические паттерны отдельных товаров

**План**: Если global модель даст SMAPE > 15%, попробовать local models для топ-20% товаров по объему продаж.

### Feature Engineering Strategy
**Приоритет 1** (Must have):
- Временные признаки: year, month, week, dayofweek
- Лаги: sales_lag_7, sales_lag_14, sales_lag_30, sales_lag_365
- Rolling stats: rolling_mean_7, rolling_std_7

**Приоритет 2** (Should have):
- Exponential weighted moving average
- Тренды: diff_7, diff_30
- Seasonal decomposition features

**Приоритет 3** (Nice to have):
- Взаимодействие store × item
- Fourier features для сезонности
- External data (если доступны)

### Data Leakage Prevention
**Критично**: При создании лагов и rolling features использовать только прошлые данные!
```python
# ❌ WRONG: использует будущие данные
df['rolling_mean'] = df['sales'].rolling(7).mean()

# ✅ CORRECT: shift перед rolling
df['rolling_mean'] = df['sales'].shift(1).rolling(7).mean()
```

## Масштабируемость

### Current Scale
- 500 временных рядов
- ~913K строк в train
- Умещается в память одной машины

### Future Scale (если потребуется)
- Параллельная обработка через joblib/multiprocessing
- Dask для out-of-memory вычислений
- Ray для distributed training

