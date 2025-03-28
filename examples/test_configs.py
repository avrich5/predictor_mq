"""
test_configs.py

Скрипт для тестирования различных конфигураций предиктора.
Останавливает каждый прогон на 4000 итерации для быстрого сравнения.
"""
import os
import sys
import pandas as pd
import numpy as np
import time
from datetime import datetime

# Добавляем корневую директорию проекта в путь для импорта
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# Импорт из проекта
from config import PredictorConfig
from predictor_mq.models.hybrid_predictor import HybridPredictor

# Создаем директорию для результатов, если её нет
os.makedirs("config_tests", exist_ok=True)

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

def create_config(significant_change_pct, confidence_threshold):
    """
    Создает конфигурацию с заданными параметрами
    
    Параметры:
    significant_change_pct (float): порог значимого изменения
    confidence_threshold (float): порог уверенности
    
    Возвращает:
    PredictorConfig: объект конфигурации
    """
    return PredictorConfig(
        window_size=500,
        prediction_depth=15,
        state_length=5,
        significant_change_pct=significant_change_pct,
        quantiles=(0.05, 0.25, 0.5, 0.75, 0.95),
        min_samples_for_regression=3,
        confidence_threshold=confidence_threshold,
        max_coverage=0.8,
        plateau_window=500
    )

def test_config(config, prices, volumes=None, max_iterations=4000):
    """
    Тестирует конфигурацию на данных
    
    Параметры:
    config (PredictorConfig): конфигурация для тестирования
    prices (numpy.array): массив цен
    volumes (numpy.array, optional): массив объемов
    max_iterations (int): максимальное количество итераций
    
    Возвращает:
    dict: результаты тестирования
    """
    print("\nТестирование конфигурации:")
    print(config)
    
    # Создаем и запускаем предиктор
    predictor = HybridPredictor(config)
    
    # Ограничиваем данные для быстрого тестирования
    limited_prices = prices[:min(len(prices), max_iterations + config.prediction_depth)]
    limited_volumes = None
    if volumes is not None:
        limited_volumes = volumes[:min(len(volumes), max_iterations + config.prediction_depth)]
    
    # Замеряем время выполнения
    start_time = time.time()
    
    # Запускаем предиктор
    try:
        results = predictor.run_on_data(
            limited_prices, 
            volumes=limited_volumes, 
            verbose=True, 
            detect_plateau=False  # Отключаем обнаружение плато для полного прогона
        )
    except Exception as e:
        print(f"Ошибка при выполнении тестирования: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Вычисляем время выполнения
    execution_time = time.time() - start_time
    
    # Собираем статистику по состояниям
    state_stats = predictor.get_state_statistics()
    unique_states = len(state_stats) if not state_stats.empty else 0
    
    # Формируем результат
    test_result = {
        'significant_change_pct': config.significant_change_pct,
        'confidence_threshold': config.confidence_threshold,
        'total_predictions': predictor.total_predictions,
        'correct_predictions': predictor.correct_predictions,
        'success_rate': predictor.success_rate * 100,
        'coverage': (predictor.total_predictions / len(limited_prices)) * 100,
        'unique_states': unique_states,
        'execution_time': execution_time
    }
    
    print("Результаты тестирования:")
    print(f"- Всего предсказаний: {test_result['total_predictions']}")
    print(f"- Правильных предсказаний: {test_result['correct_predictions']}")
    print(f"- Успешность: {test_result['success_rate']:.2f}%")
    print(f"- Покрытие: {test_result['coverage']:.2f}%")
    print(f"- Уникальных состояний: {test_result['unique_states']}")
    print(f"- Время выполнения: {test_result['execution_time']:.2f} сек.")
    
    return test_result

def main():
    # Загружаем данные для тестирования
    data_path = "data/train/btc_price_data.csv"
    prices, volumes = load_data(data_path)
    
    # Определяем конфигурации для тестирования
    configs = [
        {"significant_change_pct": 0.002, "confidence_threshold": 0.003},  # Базовая
        {"significant_change_pct": 0.001, "confidence_threshold": 0.003},  # Меньший порог изменений
        {"significant_change_pct": 0.003, "confidence_threshold": 0.003},  # Больший порог изменений
        {"significant_change_pct": 0.002, "confidence_threshold": 0.002},  # Меньший порог уверенности
        {"significant_change_pct": 0.002, "confidence_threshold": 0.004},  # Больший порог уверенности
        {"significant_change_pct": 0.001, "confidence_threshold": 0.002},  # Минимальные пороги
        {"significant_change_pct": 0.003, "confidence_threshold": 0.004},  # Максимальные пороги
        {"significant_change_pct": 0.0015, "confidence_threshold": 0.0025}  # Средние значения
    ]
    
    # Тестируем каждую конфигурацию
    results = []
    
    for i, config_params in enumerate(configs):
        print(f"\n--- Тестирование конфигурации {i+1}/{len(configs)} ---")
        
        # Создаем конфигурацию
        config = create_config(
            config_params["significant_change_pct"], 
            config_params["confidence_threshold"]
        )
        
        # Тестируем конфигурацию
        result = test_config(config, prices, volumes, max_iterations=4000)
        
        if result:
            results.append(result)
    
    # Создаем сводную таблицу
    if results:
        results_df = pd.DataFrame(results)
        
        # Сортируем по успешности
        sorted_by_success = results_df.sort_values('success_rate', ascending=False)
        print("\nРезультаты, отсортированные по успешности:")
        print(sorted_by_success.to_string(index=False))
        
        # Сортируем по покрытию
        sorted_by_coverage = results_df.sort_values('coverage', ascending=False)
        print("\nРезультаты, отсортированные по покрытию:")
        print(sorted_by_coverage.to_string(index=False))
        
        # Сохраняем результаты
        timestamp = get_timestamp()
        csv_path = f"config_tests/results_{timestamp}.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"\nРезультаты сохранены в {csv_path}")
        
        # Создаем рекомендации на основе результатов
        best_overall = None
        max_score = 0
        
        for _, row in results_df.iterrows():
            # Простая метрика: success_rate * sqrt(coverage)
            # Даем больший вес успешности, но учитываем и покрытие
            score = row['success_rate'] * np.sqrt(row['coverage'])
            
            if score > max_score:
                max_score = score
                best_overall = row
        
        if best_overall is not None:
            print("\nРекомендуемая конфигурация:")
            print(f"- significant_change_pct: {best_overall['significant_change_pct']}")
            print(f"- confidence_threshold: {best_overall['confidence_threshold']}")
            print(f"- Успешность: {best_overall['success_rate']:.2f}%")
            print(f"- Покрытие: {best_overall['coverage']:.2f}%")
    else:
        print("Не удалось получить результаты тестирования.")

if __name__ == "__main__":
    main()