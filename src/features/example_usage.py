"""
Пример использования модуля feature engineering.

Этот скрипт демонстрирует, как использовать build_features для создания
всех фичей из исходных данных.
"""

import sys
from pathlib import Path

# Добавляем корневую папку проекта в путь
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.load_data import load_train, load_test
from src.data.save_data import save_dataframe, load_dataframe
from src.features.build_features import (
    build_all_features, 
    build_features_for_test,
    get_feature_list
)


def main():
    """Основная функция для демонстрации."""
    
    print("=" * 80)
    print("🚀 ПРИМЕР ИСПОЛЬЗОВАНИЯ FEATURE ENGINEERING")
    print("=" * 80)
    
    # Загрузка данных
    print("\n📥 Загрузка данных...")
    train = load_train()
    test = load_test()
    
    print(f"   Train: {train.shape}")
    print(f"   Test: {test.shape}")
    
    # Создание фичей для train (с sales)
    print("\n🔧 Создание фичей для train...")
    train_features = build_all_features(
        train,
        feature_groups=None,  # Все группы
        lag_periods=[1, 7, 14, 30, 90, 365],
        rolling_windows=[7, 30],
        verbose=True
    )
    
    # Получение списка фичей
    feature_cols = get_feature_list(train_features)
    print(f"\n📊 Создано {len(feature_cols)} фичей:")
    print(f"   Примеры: {feature_cols[:10]}")
    
    # Сохранение (опционально)
    output_dir = project_root / 'data' / 'processed'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем train features в CSV (удобно для анализа и корреляций)
    output_file = output_dir / 'train_features.csv'
    save_dataframe(train_features, output_file)
    
    # Для test нужно создать фичи с использованием train данных для лагов
    print("\n🔧 Создание фичей для test...")
    print("   ⚠️ Используем train данные для создания лагов в test...")
    
    # Используем специальную функцию для test
    test_features = build_features_for_test(
        train,
        test,
        feature_groups=None,
        lag_periods=[1, 7, 14, 30, 90, 365],
        rolling_windows=[7, 30],
        verbose=True
    )
    
    # Сохраняем test features в CSV
    output_file = output_dir / 'test_features.csv'
    save_dataframe(test_features, output_file)
    
    # Демонстрация: анализ корреляций с помощью валидационных утилит
    print("\n📊 Демонстрация валидационных утилит:")
    from src.features.validation import (
        get_feature_correlations_with_target,
        analyze_correlations,
        find_redundant_features
    )
    
    loaded_train = load_dataframe(output_file.parent / 'train_features.csv', verbose=False)
    
    # Топ-10 фичей по корреляции с sales
    top_features = get_feature_correlations_with_target(
        loaded_train, target='sales', top_n=10
    )
    print(f"\n   Топ-10 фичей по корреляции с sales:")
    print(f"   {top_features.to_string(index=False)}")
    
    # Анализ высоких корреляций между фичами
    high_corr = analyze_correlations(loaded_train, target='sales', threshold=0.95)
    if not high_corr.empty:
        print(f"\n   Найдено {len(high_corr)} пар фичей с корреляцией > 0.95:")
        print(f"   {high_corr.head(5).to_string(index=False)}")
    else:
        print("\n   ✅ Нет пар фичей с очень высокой корреляцией (>0.95)")
    
    # Поиск избыточных фичей
    redundant = find_redundant_features(loaded_train, target='sales', corr_threshold=0.98)
    if redundant:
        print(f"\n   Рекомендуется удалить {len(redundant)} избыточных фичей:")
        print(f"   {redundant[:10]}")
    else:
        print("\n   ✅ Избыточных фичей не найдено")
    
    print("\n" + "=" * 80)
    print("✅ ГОТОВО!")
    print("=" * 80)
    print(f"\n📈 Статистика:")
    print(f"   Train features: {train_features.shape}")
    print(f"   Test features: {test_features.shape}")
    print(f"   Всего фичей: {len(feature_cols)}")


if __name__ == '__main__':
    import pandas as pd
    main()

