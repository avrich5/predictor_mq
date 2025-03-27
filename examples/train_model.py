"""
train_model.py

Пример обучения и оптимизации гибридного предиктора.
"""
import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime

# Добавляем корневую директорию проекта в путь для импорта
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# Теперь можно импортировать из корневой директории
from config import get_config
from predictor_mq.models.hybrid_predictor import HybridPredictor

# Создаем директорию для отчетов, если её нет
os.makedirs("reports", exist_ok=True)


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


def main():
    # Загружаем данные для обучения
    data_path = "data/train/btc_price_data.csv"
    prices, volumes = load_data(data_path)
    
    # Создаем предиктор с оптимизированной конфигурацией
    config = get_config("quick_test")
    # config = get_config("standard")
    # config = get_config("optimized")
    print("\nИспользуемая конфигурация:")
    print(config)
    
    # Создаем и запускаем предиктор
    print("\nЗапуск предиктора...")
    predictor = HybridPredictor(config)
    results = predictor.run_on_data(
        prices, 
        volumes=volumes, 
        verbose=True, 
        detect_plateau=True
    )
    
    # Выводим результаты
    print("\nРезультаты обучения:")
    print(f"- Всего предсказаний: {predictor.total_predictions}")
    print(f"- Правильных предсказаний: {predictor.correct_predictions}")
    print(f"- Успешность: {predictor.success_rate * 100:.2f}%")
    
    # Сохраняем результаты
    timestamp = get_timestamp()
    save_path = f"reports/training_result_{timestamp}.png"
    predictor.visualize_results(prices, results, save_path)
    
    # Генерируем отчет
    report_path = f"reports/training_report_{timestamp}.md"
    predictor.generate_report(results, report_path, prices)
    print(f"Отчет сохранен в {report_path}")
    
    print("\nОбучение завершено.")


if __name__ == "__main__":
    main()