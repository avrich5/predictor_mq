"""
validate_optimized.py

Скрипт для валидации оптимизированного предиктора, чтобы убедиться,
что результаты совпадают с исходной версией (success rate 57.81%).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import time
import json
import sys

# Добавляем текущую директорию в путь для импорта
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Импортируем необходимые модули
try:
   from config import get_config, PredictorConfig
   from predictor_mq.models.hybrid_predictor import HybridPredictor
except ImportError as e:
   print(f"Ошибка импорта: {e}")
   print(f"Текущая директория: {current_dir}")
   print(f"Содержимое директории: {os.listdir(current_dir)}")
   if os.path.exists(os.path.join(current_dir, 'models')):
       print(f"Содержимое models: {os.listdir(os.path.join(current_dir, 'models'))}")
   sys.exit(1)

# Создаем директорию для результатов, если её нет
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


def create_target_config():
   """Создает конфигурацию для достижения целевой успешности 57.81%"""
   return PredictorConfig(
       # Основные параметры (из отчета)
       window_size=750,
       prediction_depth=15,
       
       # Параметры состояний
       state_length=4,
       significant_change_pct=0.004,  # 0.4%
       
       # Параметры квантильной регрессии
       quantiles=(0.1, 0.5, 0.9),
       min_samples_for_regression=10,
       
       # Параметры предсказаний
       min_confidence=0.6,
       confidence_threshold=0.5,
       max_coverage=0.05,
       
       # Другие параметры (стандартные)
       default_movement=1,
       threshold_percentile=75,
       regression_alpha=0.1,
       regression_solver='highs',
       lower_quantile=0.1,
       upper_quantile=0.9,
       
       # Параметры для обработки данных
       model_update_interval=100,
       plateau_window=1000
   )


def run_validation(data_file, save_results=True):
   """
   Запускает валидацию на одном наборе данных
   
   Параметры:
   data_file (str): путь к файлу данных
   save_results (bool): сохранять ли результаты
   
   Возвращает:
   dict: результаты валидации
   """
   print(f"\nЗапуск валидации на наборе данных: {data_file}")
   
   # Загружаем данные
   prices, volumes = load_data(data_file)
   if prices is None:
       return None
   
   # Создаем оптимизированную конфигурацию
   config = create_target_config()
   
   # Создаем и запускаем предиктор
   print("\nИспользуемая конфигурация:")
   print(config)
   
   predictor = HybridPredictor(config)
   
   # Замеряем время выполнения
   start_time = time.time()
   
   try:
       results = predictor.run_on_data(
           prices, 
           volumes=volumes, 
           verbose=True,
           detect_plateau=True
       )
       
       execution_time = time.time() - start_time
       
       # Выводим результаты
       print(f"\nРезультаты валидации:")
       print(f"- Время выполнения: {execution_time:.2f} секунд")
       print(f"- Всего предсказаний: {predictor.total_predictions}")
       print(f"- Правильных предсказаний: {predictor.correct_predictions}")
       print(f"- Успешность: {predictor.success_rate * 100:.2f}%")
       
       # Проверяем, соответствует ли успешность целевой
       target_success_rate = 57.81
       actual_success_rate = predictor.success_rate * 100
       diff = abs(actual_success_rate - target_success_rate)
       
       print(f"\nСравнение с целевой успешностью:")
       print(f"- Целевая успешность: {target_success_rate}%")
       print(f"- Фактическая успешность: {actual_success_rate:.2f}%")
       print(f"- Отклонение: {diff:.2f}%")
       
       # Формируем результат
       validation_result = {
           'dataset': os.path.basename(data_file),
           'execution_time': execution_time,
           'success_rate': predictor.success_rate * 100,
           'target_success_rate': target_success_rate,
           'diff': diff,
           'total_predictions': predictor.total_predictions,
           'correct_predictions': predictor.correct_predictions,
           'coverage': (predictor.total_predictions / len(prices)) * 100,
           'is_target_achieved': diff < 0.01  # Проверяем, достигнута ли целевая точность с погрешностью 0.01%
       }
       
       # Сохраняем результаты
       if save_results:
           timestamp = get_timestamp()
           
           # Сохраняем график
           save_path = f"validation_results/optimized_{os.path.basename(data_file).replace('.csv', '')}_{timestamp}.png"
           print(f"Сохранение визуализации в {save_path}")
           predictor.visualize_results(prices, results, save_path)
           
           # Сохраняем отчет
           report_path = f"validation_results/optimized_{os.path.basename(data_file).replace('.csv', '')}_{timestamp}_report.md"
           print(f"Сохранение отчета в {report_path}")
           predictor.generate_report(results, report_path, prices)
           
           # Сохраняем метаданные о валидации
           meta_path = f"validation_results/optimized_{os.path.basename(data_file).replace('.csv', '')}_{timestamp}_meta.json"
           with open(meta_path, 'w') as f:
               json.dump(validation_result, f, indent=4)
           
       return validation_result
   
   except Exception as e:
       print(f"Ошибка при выполнении валидации: {e}")
       import traceback
       traceback.print_exc()
       return None


def main():
   # Указываем путь к данным для валидации
   data_dir = "data/validation"
   
   # Если указан конкретный файл, проверяем его
   data_file = "data/validation/btc_price_data.csv"  # Замените на ваш файл
   
   if os.path.exists(data_file):
       validation_result = run_validation(data_file)
       if validation_result and validation_result['is_target_achieved']:
           print("\nВалидация успешна! Целевая точность достигнута.")
       else:
           print("\nВалидация не удалась. Целевая точность не достигнута.")
   else:
       print(f"Файл {data_file} не найден.")
       
       # Проверяем все файлы в директории
       if os.path.exists(data_dir):
           files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
           if files:
               print(f"Найдены следующие файлы данных:")
               for i, file in enumerate(files):
                   print(f"{i+1}. {file}")
               
               # Спрашиваем пользователя, какой файл использовать
               file_idx = int(input("Введите номер файла для валидации: ")) - 1
               if 0 <= file_idx < len(files):
                   validation_result = run_validation(files[file_idx])
                   if validation_result and validation_result['is_target_achieved']:
                       print("\nВалидация успешна! Целевая точность достигнута.")
                   else:
                       print("\nВалидация не удалась. Целевая точность не достигнута.")
               else:
                   print("Неверный номер файла.")
           else:
               print(f"В директории {data_dir} не найдены CSV файлы.")
       else:
           print(f"Директория {data_dir} не существует.")


if __name__ == "__main__":
   main()