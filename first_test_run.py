"""
Первый тестовый запуск гибридного предиктора.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

# Добавляем текущую директорию в путь для импорта
sys.path.append('.')

from config import PredictorConfig
from predictor_mq.models.hybrid_predictor import HybridPredictor

def load_data(file_path):
    """
    Загружает данные из CSV файла
    
    Параметры:
    file_path (str): путь к файлу
    
    Возвращает:
    tuple: (массив цен, массив объемов)
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Загружено {len(df)} строк данных из {file_path}")
        
        # Определяем колонки с ценой и объемом
        price_columns = ['price', 'close']
        price_column = next((col for col in price_columns if col in df.columns), None)
        
        if not price_column:
            price_column = df.columns[0]
            print(f"Используем первую колонку: {price_column}")
        
        prices = df[price_column].values
        
        # Проверяем наличие данных по объемам
        volume_columns = ['volume', 'volume_base']
        volume_column = next((col for col in volume_columns if col in df.columns), None)
        
        volumes = None
        if volume_column:
            volumes = df[volume_column].values
        
        return prices, volumes
    
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        print("Генерируем синтетические данные...")
        
        # Генерируем синтетические данные
        np.random.seed(42)
        n_points = 5000
        prices = np.cumsum(np.random.normal(0, 1, n_points)) + 1000
        return prices, None

def main():
    # Создаем директорию для отчетов, если её нет
    os.makedirs("reports", exist_ok=True)
    
    # Загружаем данные для теста
    data_path = "data/train/example_data.csv"
    prices, volumes = load_data(data_path)
    
    # Создаем минимальную конфигурацию для теста
    config = PredictorConfig(
        window_size=500,      # Меньшее окно для быстрого теста
        prediction_depth=10,  # Меньше глубина предсказания
        state_length=4,       # Стандартная длина состояния
        significant_change_pct=0.01,  # Стандартный порог изменения
        quantiles=(0.1, 0.5, 0.9),    # Основные квантили
        confidence_threshold=0.55,    # Стандартный порог уверенности
        max_coverage=0.05            # Ограничиваем количество предсказаний
    )
    
    print("\nИспользуемая конфигурация:")
    print(config)
    
    # Создаем и запускаем предиктор
    print("\nЗапуск предиктора...")
    predictor = HybridPredictor(config)
    
    try:
        results = predictor.run_on_data(
            prices, 
            volumes=volumes, 
            verbose=True
        )
        
        # Выводим результаты
        print("\nРезультаты первого запуска:")
        print(f"- Всего предсказаний: {predictor.total_predictions}")
        print(f"- Правильных предсказаний: {predictor.correct_predictions}")
        print(f"- Успешность: {predictor.success_rate * 100:.2f}%")
        
        # Сохраняем результаты
        save_path = "reports/first_test_result.png"
        predictor.visualize_results(prices, results, save_path)
        print(f"Визуализация сохранена в {save_path}")
        
    except Exception as e:
        print(f"Ошибка при выполнении предиктора: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()