# Tech Context: Technologies & Tools

## Technology Stack

### Core
- **Python**: 3.9+ (рекомендуется 3.10 или 3.11)
- **OS**: Cross-platform (разработка на Windows, потенциально Linux для prod)

### Data Processing
- **pandas**: 2.x - основная библиотека для работы с данными
- **numpy**: 1.24+ - численные вычисления
- **polars**: (опционально) для ускорения обработки больших данных

### Machine Learning
**Classical ML**:
- **scikit-learn**: 1.3+ - базовые модели, preprocessing, pipelines
- **XGBoost**: 2.0+ - gradient boosting
- **LightGBM**: 4.x - быстрый gradient boosting
- **CatBoost**: 1.2+ - gradient boosting с нативной поддержкой categorical features

**Time Series Specific** (опционально):
- **statsmodels**: ARIMA, SARIMAX, seasonal decomposition
- **prophet**: Facebook's forecasting tool

**Deep Learning** (Iteration 3):
- **PyTorch**: 2.x - для LSTM/GRU моделей
- **TensorFlow/Keras**: альтернатива PyTorch

### Visualization & Analysis
- **matplotlib**: 3.7+ - базовая визуализация
- **seaborn**: 0.12+ - статистические графики
- **plotly**: интерактивные графики для dashboard

### Development Tools
- **jupyter**: notebook для экспериментов
- **ipython**: enhanced REPL
- **black**: code formatter
- **ruff**: быстрый linter (замена flake8)

### API & Production
- **FastAPI**: 0.104+ - современный async web framework
- **pydantic**: 2.x - валидация данных и схемы
- **uvicorn**: ASGI server

### Deployment (опционально)
- **Docker**: контейнеризация
- **docker-compose**: локальная оркестрация

### Version Control & Artifacts
- **Git**: version control
- **DVC**: (опционально) version control для данных и моделей
- **MLflow**: (опционально) experiment tracking

## Development Environment

### Setup
```bash
# Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows

# Установка зависимостей
pip install -r requirements.txt

# Для разработки
pip install -r requirements-dev.txt  # если будет создан
```

### Project Structure Conventions
```
shop_project/
├── data/
│   ├── raw/              # Неизменяемые исходные данные
│   └── processed/        # Обработанные данные, фичи
│
├── notebooks/            # Jupyter notebooks для экспериментов
│   └── *.ipynb
│
├── src/                  # Production-ready код
│   ├── data/
│   ├── features/
│   ├── models/
│   └── api/
│
├── artifacts/            # Сохраненные модели, трансформеры
│   ├── models/
│   ├── scalers/
│   └── features/
│
├── submissions/          # Kaggle submissions
│
├── tests/                # Unit и integration tests
│
├── memory-bank/          # Документация проекта
│
├── .gitignore
├── requirements.txt
├── README.md
└── Dockerfile            # Для деплоя
```

## Technical Constraints

### Computational Resources
- **Kaggle Kernels**: 
  - RAM: 16GB limit
  - CPU: 4 cores
  - Time: 9 hours max
  - No GPU for this competition

### Data Constraints
- **Train size**: ~913K rows, ~6MB
- **Test size**: ~45K rows (3 months forecast)
- **Memory footprint**: После feature engineering ожидается ~50-100MB
- **Легко умещается в RAM** даже на слабой машине

### Performance Requirements
- **Training**: Допустимо до 1 часа на full training
- **Inference**: 
  - Single prediction: < 100ms
  - Batch (500 items): < 10 seconds
- **SMAPE**: целевой показатель < 15%

## Dependencies Management

### Core Dependencies (requirements.txt)
```
# Data
pandas>=2.0.0
numpy>=1.24.0

# ML
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Utils
tqdm>=4.65.0
joblib>=1.3.0
```

### API Dependencies (добавить при создании API)
```
# API
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.0.0
python-multipart>=0.0.6
```

### Development Dependencies
```
# Development
jupyter>=1.0.0
ipython>=8.12.0
black>=23.0.0
ruff>=0.1.0

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0
```

## Known Technical Issues

### Issue 1: Data Leakage Risk
**Проблема**: Временные ряды требуют особого внимания к созданию фичей  
**Решение**: Всегда использовать `.shift()` перед созданием lag/rolling features

### Issue 2: Memory Efficiency
**Проблема**: Создание множества lag features может раздуть датасет  
**Решение**: Использовать `dtype` оптимизацию (int8, int16 вместо int64)

### Issue 3: Cross-Validation
**Проблема**: Обычный KFold нарушает временную структуру  
**Решение**: TimeSeriesSplit или manual train/val/test split по датам

## Security & Privacy
- ✅ Данные публичные (Kaggle)
- ✅ Нет PII (Personally Identifiable Information)
- ✅ Нет специальных compliance требований

