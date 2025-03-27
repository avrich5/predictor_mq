"""
validate_model.py

Пример валидации гибридного предиктора.
"""
import os
import sys

# Добавляем корневую директорию проекта в путь для импорта
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from config import get_config
from predictor_mq.models.hybrid_predictor import HybridPredictor

# Создаем директорию для отчетов, если её нет
os.makedirs("validation_results", exist_ok=True)


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
            print(f"Не найдена колонка с ценой. Доступные колонки: {df.columns.tolist()}")
            price_column = df.columns[0]
            print(f"Используем первую колонку: {price_column}")
        
        prices = df[price_column].values
        
        # Проверяем наличие данных по объемам
        volume_columns = ['volume', 'volume_base']
        volume_column = next((col for col in volume_columns if col in df.columns), None)
        
        volumes = None
        if volume_column:
            volumes = df[volume_column].values
            print(f"Используются данные объемов из колонки {volume_column}")
        
        return prices, volumes
    
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        print("Генерируем синтетические данные...")
        
        # Генерируем синтетические данные
        np.random.seed(42)
        n_points = 10000
        prices = np.cumsum(np.random.normal(0, 1, n_points)) + 1000
        return prices, None


def get_timestamp():
    """Возвращает текущую метку времени в формате YYYYMMDD_HHMMSS"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_validation_on_dataset(data_file, config):
    """
    Запускает валидацию на одном наборе данных
    
    Параметры:
    data_file (str): путь к файлу данных
    config: конфигурация предиктора
    
    Возвращает:
    dict: результаты валидации
    """
    print(f"\nЗапуск валидации на наборе данных: {data_file}")
    
    # Загружаем данные
    prices, volumes = load_data(data_file)
    
    # Создаем и запускаем предиктор
    predictor = HybridPredictor(config)
    
    # Запускаем предиктор
    try:
        results = predictor.run_on_data(
            prices, 
            volumes=volumes, 
            verbose=True, 
            detect_plateau=True
        )
    except Exception as e:
        print(f"Ошибка при выполнении валидации: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Выводим результаты
    print(f"Валидация завершена:")
    print(f"- Всего предсказаний: {predictor.total_predictions}")
    print(f"- Правильных предсказаний: {predictor.correct_predictions}")
    print(f"- Успешность: {predictor.success_rate * 100:.2f}%")
    
    # Формируем результат
    validation_result = {
        'dataset': os.path.basename(data_file),
        'prices': prices,
        'results': results,
        'predictor': predictor,
        'success_rate': predictor.success_rate * 100,
        'total_predictions': predictor.total_predictions,
        'correct_predictions': predictor.correct_predictions,
        'coverage': (predictor.total_predictions / len(prices)) * 100
    }
    
    return validation_result


def run_validation_pipeline(data_dir, preset_name="high_volatility"):
    """
    Запускает валидацию на всех наборах данных в директории
    
    Параметры:
    data_dir (str): директория с файлами данных
    preset_name (str): имя предустановки конфигурации
    
    Возвращает:
    dict: результаты валидации
    """
    # Получаем конфигурацию
    config = get_config(preset_name)
    print(f"Используемая конфигурация: {preset_name}")
    print(config)
    
    # Находим все CSV файлы в директории
    data_files = [
        os.path.join(data_dir, f) for f in os.listdir(data_dir)
        if f.endswith('.csv')
    ]
    
    if not data_files:
        print(f"Не найдены CSV файлы в директории {data_dir}")
        return None
    
    print(f"Найдено {len(data_files)} файлов для валидации")
    
    # Выполняем валидацию для каждого набора данных
    validation_results = {}
    timestamp = get_timestamp()
    
    for data_file in data_files:
        dataset_name = os.path.basename(data_file).replace('.csv', '')
        
        # Запускаем валидацию
        validation_result = run_validation_on_dataset(data_file, config)
        
        if validation_result:
            validation_results[dataset_name] = validation_result
            
            # Сохраняем результаты
            save_path = f"validation_results/{dataset_name}_{timestamp}.png"
            print(f"Сохранение визуализации в {save_path}")
            validation_result['predictor'].visualize_results(
                validation_result['prices'],
                validation_result['results'],
                save_path
            )
            
            # Генерируем отчет
            report_path = f"validation_results/{dataset_name}_{timestamp}.md"
            print(f"Сохранение отчета в {report_path}")
            validation_result['predictor'].generate_report(
                validation_result['results'],
                report_path,
                validation_result['prices']
            )
    
    # Создаем сводную таблицу
    if validation_results:
        summary_df = pd.DataFrame([
            {
                'Dataset': name,
                'Success Rate (%)': result['success_rate'],
                'Total Predictions': result['total_predictions'],
                'Correct Predictions': result['correct_predictions'],
                'Coverage (%)': result['coverage']
            }
            for name, result in validation_results.items()
        ])
        
        # Сохраняем таблицу
        summary_path = f"validation_results/summary_{timestamp}.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Сводная таблица сохранена в {summary_path}")
        
        # Выводим таблицу
        print("\nСводная таблица результатов:")
        print(summary_df.to_string(index=False))
    
    return validation_results


def main():
    # Запускаем валидацию на всех наборах данных
    data_dir = "data/validation"
    
    # Выбираем конфигурацию для валидации
    preset_name = "optimized"  # Используем оптимизированную конфигурацию
    
    print(f"Запуск валидации на данных из {data_dir}")
    validation_results = run_validation_pipeline(data_dir, preset_name)
    
    if not validation_results:
        print("Валидация не выполнена или не дала результатов.")
        return
    
    print("\nВалидация завершена.")


if __name__ == "__main__":
    main()