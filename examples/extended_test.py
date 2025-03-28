"""
extended_test.py

Расширенный скрипт для тестирования различных параметров конфигурации предиктора.
Позволяет исследовать влияние расширенного набора параметров на работу модели.
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
from collections import defaultdict
import itertools
import json

# Добавляем корневую директорию проекта в путь для импорта
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# Импорт из проекта
from config import PredictorConfig
from predictor_mq.models.hybrid_predictor import HybridPredictor

# Создаем директории для результатов, если их нет
os.makedirs("experiment_results", exist_ok=True)

class PredictionTracker:
    """
    Класс для отслеживания и анализа предсказаний
    """
    def __init__(self, prices, prediction_depth):
        self.prices = prices
        self.prediction_depth = prediction_depth
        self.predictions = []
        self.confidence_bins = np.linspace(0, 1, 11)  # 10 интервалов для уверенности
        self.confidence_accuracy = defaultdict(lambda: {'correct': 0, 'total': 0})
        self.success_by_state = defaultdict(lambda: {'correct': 0, 'total': 0})
        
    def add_prediction(self, idx, prediction, confidence, state, threshold):
        """
        Добавляет предсказание в трекер и проверяет его правильность
        """
        if idx + self.prediction_depth >= len(self.prices):
            return  # Невозможно проверить предсказание
        
        # Определяем фактическое изменение
        current_price = self.prices[idx]
        future_price = self.prices[idx + self.prediction_depth]
        pct_change = (future_price - current_price) / current_price
        
        # Определяем фактическое движение
        actual_movement = 0
        if pct_change > threshold:
            actual_movement = 1  # Рост
        elif pct_change < -threshold:
            actual_movement = 2  # Падение
        
        # Проверяем правильность предсказания, если это не "не знаю"
        is_correct = False
        if prediction != 0:
            is_correct = (prediction == actual_movement)
            
            # Находим бин для уверенности
            confidence_bin = np.digitize(confidence, self.confidence_bins) - 1
            bin_label = f"{self.confidence_bins[confidence_bin]:.1f}-{self.confidence_bins[confidence_bin+1]:.1f}"
            
            # Обновляем статистику по уверенности
            self.confidence_accuracy[bin_label]['total'] += 1
            if is_correct:
                self.confidence_accuracy[bin_label]['correct'] += 1
            
            # Обновляем статистику по состояниям
            state_key = str(state)
            self.success_by_state[state_key]['total'] += 1
            if is_correct:
                self.success_by_state[state_key]['correct'] += 1
        
        # Сохраняем предсказание для дальнейшего анализа
        self.predictions.append({
            'index': idx,
            'price': current_price,
            'future_price': future_price,
            'prediction': prediction,
            'confidence': confidence,
            'state': state,
            'threshold': threshold,
            'pct_change': pct_change * 100,  # Переводим в проценты
            'actual_movement': actual_movement,
            'is_correct': is_correct if prediction != 0 else None
        })
    
    def get_confidence_accuracy_df(self):
        """
        Возвращает DataFrame с анализом точности в зависимости от уверенности
        """
        data = []
        for bin_label, stats in sorted(self.confidence_accuracy.items()):
            if stats['total'] > 0:
                accuracy = (stats['correct'] / stats['total']) * 100
                data.append({
                    'confidence_range': bin_label,
                    'correct': stats['correct'],
                    'total': stats['total'],
                    'accuracy': accuracy
                })
        
        return pd.DataFrame(data)
    
    def get_state_accuracy_df(self):
        """
        Возвращает DataFrame с анализом точности в зависимости от состояния
        """
        data = []
        for state, stats in sorted(self.success_by_state.items(), key=lambda x: -x[1]['total']):
            if stats['total'] > 0:
                accuracy = (stats['correct'] / stats['total']) * 100
                data.append({
                    'state': state,
                    'correct': stats['correct'],
                    'total': stats['total'],
                    'accuracy': accuracy
                })
        
        return pd.DataFrame(data)
    
    def generate_detailed_analysis(self, config_params):
        """
        Генерирует детальный анализ предсказаний
        
        Параметры:
        config_params (dict): параметры конфигурации
        
        Возвращает:
        dict: результаты анализа
        """
        if not self.predictions:
            return None
        
        predictions_df = pd.DataFrame(self.predictions)
        
        # Фильтруем только реальные предсказания (не "не знаю")
        real_predictions = predictions_df[predictions_df['prediction'] != 0]
        
        # Общая статистика
        total_predictions = len(real_predictions)
        if total_predictions > 0:
            correct_predictions = real_predictions['is_correct'].sum()
            success_rate = (correct_predictions / total_predictions) * 100
        else:
            correct_predictions = 0
            success_rate = 0
        
        # Покрытие (% точек с предсказаниями)
        coverage = (total_predictions / len(predictions_df)) * 100
        
        # Статистика по типам предсказаний
        up_predictions = real_predictions[real_predictions['prediction'] == 1]
        down_predictions = real_predictions[real_predictions['prediction'] == 2]
        
        up_success = up_predictions['is_correct'].mean() * 100 if len(up_predictions) > 0 else 0
        down_success = down_predictions['is_correct'].mean() * 100 if len(down_predictions) > 0 else 0
        
        # Анализ по уровню уверенности
        confidence_accuracy_df = self.get_confidence_accuracy_df()
        
        # Анализ по состояниям
        state_accuracy_df = self.get_state_accuracy_df()
        unique_states = len(state_accuracy_df)
        
        # Формируем результат
        analysis_result = {
            'config_params': config_params,
            'total_predictions': total_predictions,
            'correct_predictions': correct_predictions,
            'success_rate': success_rate,
            'coverage': coverage,
            'unique_states': unique_states,
            'up_predictions': len(up_predictions),
            'up_success_rate': up_success,
            'down_predictions': len(down_predictions),
            'down_success_rate': down_success,
            'confidence_analysis': confidence_accuracy_df,
            'state_analysis': state_accuracy_df,
            'predictions': predictions_df
        }
        
        return analysis_result

def load_data(file_path):
    """
    Загружает данные из CSV файла
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

def create_config_extended(significant_change_pct, confidence_threshold, 
                           state_length=5, min_samples_for_regression=3,
                           quantiles=(0.05, 0.25, 0.5, 0.75, 0.95),
                           window_size=500, prediction_depth=15):
    """
    Создает расширенную конфигурацию с дополнительными параметрами
    """
    return PredictorConfig(
        window_size=window_size,
        prediction_depth=prediction_depth,
        state_length=state_length,
        significant_change_pct=significant_change_pct,
        quantiles=quantiles,
        min_samples_for_regression=min_samples_for_regression,
        confidence_threshold=confidence_threshold,
        max_coverage=0.8,
        plateau_window=500
    )

def test_config_with_tracking(prices, volumes, config_params, max_iterations=4000):
    """
    Тестирует конфигурацию и отслеживает все предсказания
    
    Параметры:
    prices (numpy.array): массив цен
    volumes (numpy.array, optional): массив объемов
    config_params (dict): параметры конфигурации
    max_iterations (int): максимальное количество итераций
    
    Возвращает:
    dict: результаты тестирования и анализа
    """
    print(f"\nТестирование конфигурации: {config_params}")
    
    # Создаем конфигурацию и предиктор
    config = create_config_extended(**config_params)
    predictor = HybridPredictor(config)
    
    # Выводим ключевую информацию о конфигурации
    print(f"Используемая конфигурация:")
    print(config)
    
    # Ограничиваем данные для быстрого тестирования
    limited_prices = prices[:min(len(prices), max_iterations + config.prediction_depth)]
    limited_volumes = None
    if volumes is not None:
        limited_volumes = volumes[:min(len(volumes), max_iterations + config.prediction_depth)]
    
    # Создаем трекер предсказаний
    tracker = PredictionTracker(limited_prices, config.prediction_depth)
    
    # Сохраняем оригинальный метод
    original_predict_at_point = predictor.predict_at_point
    
    def predict_at_point_with_tracking(prices, idx, max_predictions=None, current_predictions=0):
        """
        Обертка для метода predict_at_point с отслеживанием предсказаний
        """
        result = original_predict_at_point(prices, idx, max_predictions, current_predictions)
        
        # Получаем порог и состояние для этой точки
        dynamic_threshold = predictor._calculate_dynamic_threshold(prices, idx)
        current_state = predictor._get_state(prices, idx, dynamic_threshold)
        
        # Добавляем предсказание в трекер
        tracker.add_prediction(
            idx=idx,
            prediction=result['prediction'],
            confidence=result['confidence'],
            state=current_state,
            threshold=dynamic_threshold
        )
        
        return result
    
    # Подменяем метод
    predictor.predict_at_point = predict_at_point_with_tracking
    
    # Запускаем предиктор
    try:
        start_time = time.time()
        results = predictor.run_on_data(
            limited_prices, 
            volumes=limited_volumes, 
            verbose=True, 
            detect_plateau=False  # Отключаем обнаружение плато для полного прогона
        )
        execution_time = time.time() - start_time
    except Exception as e:
        print(f"Ошибка при выполнении тестирования: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Генерируем детальный анализ
    analysis_result = tracker.generate_detailed_analysis(config_params)
    
    if analysis_result:
        analysis_result['execution_time'] = execution_time
        
        # Выводим основные результаты
        print("\nРезультаты анализа:")
        print(f"- Всего предсказаний: {analysis_result['total_predictions']}")
        print(f"- Правильных предсказаний: {analysis_result['correct_predictions']}")
        print(f"- Общая успешность: {analysis_result['success_rate']:.2f}%")
        print(f"- Покрытие: {analysis_result['coverage']:.2f}% точек")
        print(f"- Уникальных состояний: {analysis_result['unique_states']}")
        print(f"- Предсказания роста: {analysis_result['up_predictions']} (успешность: {analysis_result['up_success_rate']:.2f}%)")
        print(f"- Предсказания падения: {analysis_result['down_predictions']} (успешность: {analysis_result['down_success_rate']:.2f}%)")
        print(f"- Время выполнения: {analysis_result['execution_time']:.2f} сек.")
    
    return analysis_result

def calculate_score(result):
    """
    Рассчитывает интегральную оценку для конфигурации
    """
    if not result:
        return 0
    
    # Базовая оценка на основе успешности
    score = result['success_rate'] * 2
    
    # Учитываем покрытие
    coverage_factor = min(1.0, np.sqrt(result['coverage'] / 20))  # Нормализуем, 20% покрытие = 1
    score *= coverage_factor
    
    # Бонус за разницу между успешностью и 50% (случайное угадывание)
    if result['success_rate'] > 50:
        score += (result['success_rate'] - 50) * 0.5
    
    # Сбалансированность предсказаний (рост vs падение)
    if result['up_predictions'] > 0 and result['down_predictions'] > 0:
        balance = min(result['up_predictions'], result['down_predictions']) / max(result['up_predictions'], result['down_predictions'])
        score *= (0.5 + 0.5 * balance)  # От 0.5 до 1.0 в зависимости от баланса
    
    return score

def visualize_experiment_results(results, experiment_name, save_dir):
    """
    Визуализирует результаты эксперимента
    
    Параметры:
    results (list): список результатов анализа
    experiment_name (str): имя эксперимента
    save_dir (str): директория для сохранения графиков
    """
    if not results:
        return
    
    # Создаем DataFrame из результатов
    result_data = []
    for i, result in enumerate(results):
        if not result:
            continue
            
        # Добавляем базовую информацию
        entry = {
            'config_index': i+1,
            'success_rate': result['success_rate'],
            'coverage': result['coverage'],
            'unique_states': result['unique_states'],
            'up_predictions': result['up_predictions'],
            'down_predictions': result['down_predictions'],
            'up_success_rate': result['up_success_rate'],
            'down_success_rate': result['down_success_rate'],
            'execution_time': result['execution_time']
        }
        
        # Добавляем параметры конфигурации
        for key, value in result['config_params'].items():
            if isinstance(value, (int, float, str, bool)):
                entry[key] = value
            elif isinstance(value, tuple) and all(isinstance(x, (int, float)) for x in value):
                entry[f'{key}_count'] = len(value)
        
        # Рассчитываем интегральную оценку
        entry['score'] = calculate_score(result)
        
        result_data.append(entry)
    
    # Создаем DataFrame
    results_df = pd.DataFrame(result_data)
    
    # Сохраняем результаты
    results_csv = os.path.join(save_dir, f"{experiment_name}_results.csv")
    results_df.to_csv(results_csv, index=False)
    
    # Создаем график для сравнения успешности и покрытия
    plt.figure(figsize=(12, 8))
    
    # Основные метрики
    metric_colors = {
        'success_rate': 'blue',
        'coverage': 'green',
        'score': 'purple'
    }
    
    # Создаем график для каждой ключевой метрики
    for metric, color in metric_colors.items():
        if metric in results_df.columns:
            plt.plot(results_df['config_index'], results_df[metric], 
                    marker='o', color=color, label=metric)
    
    # Настраиваем график
    plt.axhline(y=50, color='red', linestyle='--', alpha=0.5)
    plt.xlabel('Конфигурация')
    plt.ylabel('Значение')
    plt.title(f'Результаты эксперимента: {experiment_name}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Создаем подписи для конфигураций
    tick_labels = []
    for i, row in results_df.iterrows():
        if 'significant_change_pct' in row and 'confidence_threshold' in row:
            label = f"{i+1}: {row['significant_change_pct']:.4f}/{row['confidence_threshold']:.4f}"
        else:
            label = f"Config {i+1}"
        tick_labels.append(label)
    
    plt.xticks(results_df['config_index'], tick_labels, rotation=45)
    plt.tight_layout()
    
    # Сохраняем график
    metrics_path = os.path.join(save_dir, f"{experiment_name}_metrics.png")
    plt.savefig(metrics_path)
    plt.close()
    
    # Создаем график соотношения успешности и покрытия
    plt.figure(figsize=(10, 8))
    
    # Размер точек зависит от оценки
    scatter = plt.scatter(results_df['coverage'], results_df['success_rate'], 
                          s=results_df['score']*5, c=results_df['config_index'], 
                          cmap='viridis', alpha=0.7)
    
    # Добавляем подписи к точкам
    for i, row in results_df.iterrows():
        plt.annotate(f"Config {row['config_index']}", 
                     xy=(row['coverage'], row['success_rate']),
                     xytext=(5, 5), textcoords='offset points')
    
    plt.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Случайное угадывание')
    plt.xlabel('Покрытие (%)')
    plt.ylabel('Успешность (%)')
    plt.title(f'Соотношение успешности и покрытия: {experiment_name}')
    plt.grid(True, alpha=0.3)
    plt.colorbar(scatter, label='Конфигурация')
    plt.legend()
    
    # Сохраняем график
    scatter_path = os.path.join(save_dir, f"{experiment_name}_scatter.png")
    plt.savefig(scatter_path)
    plt.close()
    
    # Анализируем влияние параметров (если есть достаточно данных)
    if len(results_df) >= 4:
        # Находим параметры с различными значениями
        varying_params = []
        for column in results_df.columns:
            if column not in ['config_index', 'success_rate', 'coverage', 'score', 
                           'unique_states', 'up_predictions', 'down_predictions', 
                           'up_success_rate', 'down_success_rate', 'execution_time']:
                if len(results_df[column].unique()) > 1:
                    varying_params.append(column)
        
        # Создаем графики для каждого варьируемого параметра
        for param in varying_params:
            plt.figure(figsize=(10, 6))
            
            # Сортируем данные по параметру
            sorted_df = results_df.sort_values(param)
            
            # Строим зависимость метрик от параметра
            plt.plot(sorted_df[param], sorted_df['success_rate'], 'b-', marker='o', label='Успешность (%)')
            plt.plot(sorted_df[param], sorted_df['coverage'], 'g-', marker='s', label='Покрытие (%)')
            
            # Добавляем тренд для успешности
            if len(sorted_df) > 2:
                try:
                    z = np.polyfit(sorted_df[param], sorted_df['success_rate'], 1)
                    p = np.poly1d(z)
                    plt.plot(sorted_df[param], p(sorted_df[param]), "b--", alpha=0.5)
                except:
                    pass
            
            plt.axhline(y=50, color='r', linestyle='--', alpha=0.5)
            plt.xlabel(param)
            plt.ylabel('Значение')
            plt.title(f'Влияние параметра {param} на метрики')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            
            # Сохраняем график
            param_path = os.path.join(save_dir, f"{experiment_name}_{param}.png")
            plt.savefig(param_path)
            plt.close()
    
    return results_df

def run_experiment(experiment_name, config_params_list, prices, volumes, max_iterations=4000):
    """
    Запускает эксперимент с набором конфигураций
    
    Параметры:
    experiment_name (str): название эксперимента
    config_params_list (list): список словарей с параметрами конфигураций
    prices (numpy.array): массив цен
    volumes (numpy.array, optional): массив объемов
    max_iterations (int): максимальное количество итераций
    
    Возвращает:
    tuple: (директория с результатами, датафрейм с результатами)
    """
    timestamp = get_timestamp()
    experiment_dir = f"experiment_results/{experiment_name}_{timestamp}"
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Сохраняем параметры эксперимента
    params_file = os.path.join(experiment_dir, "experiment_params.json")
    with open(params_file, 'w') as f:
        json.dump({
            'experiment_name': experiment_name,
            'timestamp': timestamp,
            'config_count': len(config_params_list),
            'max_iterations': max_iterations
        }, f, indent=2)
    
    # Тестируем каждую конфигурацию
    results = []
    
    for i, config_params in enumerate(config_params_list):
        print(f"\n=== Конфигурация {i+1}/{len(config_params_list)} для эксперимента '{experiment_name}' ===")
        
        # Тестируем конфигурацию
        result = test_config_with_tracking(prices, volumes, config_params, max_iterations)
        
        # Сохраняем результат
        if result:
            # Создаем директорию для этой конфигурации
            config_dir = os.path.join(experiment_dir, f"config_{i+1}")
            os.makedirs(config_dir, exist_ok=True)
            
            # Сохраняем параметры конфигурации
            config_file = os.path.join(config_dir, "config_params.json")
            with open(config_file, 'w') as f:
                # Преобразуем кортежи в списки для JSON
                serializable_config = {k: list(v) if isinstance(v, tuple) else v for k, v in config_params.items()}
                json.dump(serializable_config, f, indent=2)
            
            # Сохраняем предсказания
            predictions_csv = os.path.join(config_dir, "predictions.csv")
            result['predictions'].to_csv(predictions_csv, index=False)
            
            # Сохраняем анализ по уверенности
            confidence_csv = os.path.join(config_dir, "confidence.csv")
            result['confidence_analysis'].to_csv(confidence_csv, index=False)
            
            # Сохраняем анализ по состояниям
            states_csv = os.path.join(config_dir, "states.csv")
            result['state_analysis'].to_csv(states_csv, index=False)
        
        results.append(result)
    
    # Визуализируем результаты
    results_df = visualize_experiment_results(results, experiment_name, experiment_dir)
    
    return experiment_dir, results_df

def experiment_1_significant_change_confidence():
    """
    Эксперимент 1: Влияние significant_change_pct и confidence_threshold
    """
    # Создаем сетку значений
    significant_change_values = [0.001, 0.002, 0.003]
    confidence_threshold_values = [0.002, 0.003, 0.004]
    
    # Генерируем все комбинации
    config_params_list = []
    for sig_change in significant_change_values:
        for conf_threshold in confidence_threshold_values:
            config_params_list.append({
                'significant_change_pct': sig_change,
                'confidence_threshold': conf_threshold
            })
    
    return "sig_change_confidence", config_params_list

def experiment_2_state_length():
    """
    Эксперимент 2: Влияние длины состояния (state_length)
    """
    # Используем оптимальные параметры из эксперимента 1
    base_params = {
        'significant_change_pct': 0.002,  # Оптимальные параметры (пример)
        'confidence_threshold': 0.003     # Оптимальные параметры (пример)
    }
    
    # Варьируем state_length
    config_params_list = []
    for state_length in [3, 4, 5, 6, 7]:
        params = base_params.copy()
        params['state_length'] = state_length
        config_params_list.append(params)
    
    return "state_length", config_params_list

def experiment_3_min_samples():
    """
    Эксперимент 3: Влияние min_samples_for_regression
    """
    # Используем оптимальные параметры из экспериментов 1 и 2
    base_params = {
        'significant_change_pct': 0.002,  # Пример
        'confidence_threshold': 0.003,    # Пример
        'state_length': 5                 # Пример
    }
    
    # Варьируем min_samples_for_regression
    config_params_list = []
    for min_samples in [3, 5, 7, 10, 15]:
        params = base_params.copy()
        params['min_samples_for_regression'] = min_samples
        config_params_list.append(params)
    
    return "min_samples", config_params_list

def experiment_4_quantiles():
    """
    Эксперимент 4: Влияние набора квантилей
    """
    # Используем оптимальные параметры из экспериментов 1, 2 и 3
    base_params = {
        'significant_change_pct': 0.002,  # Пример
        'confidence_threshold': 0.003,    # Пример
        'state_length': 5,                # Пример
        'min_samples_for_regression': 3   # Пример
    }
    
    # Варьируем наборы квантилей
    quantile_sets = [
        (0.1, 0.5, 0.9),  # Минимальный набор
        (0.05, 0.25, 0.5, 0.75, 0.95),  # Стандартный набор
        (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99),  # Расширенный набор
        (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95),  # Сфокусированный на центре
        (0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99)   # Сфокусированный на крайних значениях
    ]
    
    config_params_list = []
    for quantiles in quantile_sets:
        params = base_params.copy()
        params['quantiles'] = quantiles
        config_params_list.append(params)
    
    return "quantiles", config_params_list

def experiment_5_state_length_significant_change():
    """
    Эксперимент 5: Взаимодействие state_length и significant_change_pct
    """
    # Базовые параметры
    base_params = {
        'confidence_threshold': 0.003    # Пример
    }
    
    # Комбинации параметров
    combinations = [
        {'state_length': 3, 'significant_change_pct': 0.001},  # Короткие состояния + низкий порог
        {'state_length': 3, 'significant_change_pct': 0.003},  # Короткие состояния + высокий порог
        {'state_length': 7, 'significant_change_pct': 0.001},  # Длинные состояния + низкий порог
        {'state_length': 7, 'significant_change_pct': 0.003}   # Длинные состояния + высокий порог
    ]
    
    config_params_list = []
    for combo in combinations:
        params = base_params.copy()
        params.update(combo)
        config_params_list.append(params)
    
    return "state_sig_change", config_params_list

def experiment_6_quantiles_confidence():
    """
    Эксперимент 6: Взаимодействие набора квантилей и confidence_threshold
    """
    # Базовые параметры
    base_params = {
        'significant_change_pct': 0.002,  # Пример
        'state_length': 5                 # Пример
    }
    
    # Комбинации параметров
    combinations = [
        {'quantiles': (0.1, 0.5, 0.9), 'confidence_threshold': 0.002},  # Простой набор + низкий порог
        {'quantiles': (0.1, 0.5, 0.9), 'confidence_threshold': 0.004},  # Простой набор + высокий порог
        {'quantiles': (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99), 'confidence_threshold': 0.002},  # Сложный набор + низкий порог
        {'quantiles': (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99), 'confidence_threshold': 0.004}   # Сложный набор + высокий порог
    ]
    
    config_params_list = []
    for combo in combinations:
        params = base_params.copy()
        params.update(combo)
        config_params_list.append(params)
    
    return "quantiles_confidence", config_params_list

def experiment_7_window_size():
    """
    Эксперимент 7: Влияние размера окна (window_size)
    """
    # Базовые параметры (оптимальные из предыдущих экспериментов)
    base_params = {
        'significant_change_pct': 0.002,  # Пример
        'confidence_threshold': 0.003,    # Пример
        'state_length': 5,                # Пример
        'quantiles': (0.05, 0.25, 0.5, 0.75, 0.95)  # Пример
    }
    
    # Варьируем window_size
    config_params_list = []
    for window_size in [250, 500, 750, 1000]:
        params = base_params.copy()
        params['window_size'] = window_size
        config_params_list.append(params)
    
    return "window_size", config_params_list

def experiment_8_prediction_depth():
    """
    Эксперимент 8: Влияние глубины прогноза (prediction_depth)
    """
    # Базовые параметры (оптимальные из предыдущих экспериментов)
    base_params = {
        'significant_change_pct': 0.002,  # Пример
        'confidence_threshold': 0.003,    # Пример
        'state_length': 5,                # Пример
        'quantiles': (0.05, 0.25, 0.5, 0.75, 0.95)  # Пример
    }
    
    # Варьируем prediction_depth
    config_params_list = []
    for prediction_depth in [5, 10, 15, 20, 30]:
        params = base_params.copy()
        params['prediction_depth'] = prediction_depth
        config_params_list.append(params)
    
    return "prediction_depth", config_params_list

def experiment_9_window_prediction_ratio():
    """
    Эксперимент 9: Взаимодействие window_size и prediction_depth
    """
    # Базовые параметры
    base_params = {
        'significant_change_pct': 0.002,  # Пример
        'confidence_threshold': 0.003,    # Пример
        'state_length': 5                 # Пример
    }
    
    # Комбинации параметров с разными соотношениями
    combinations = [
        {'window_size': 250, 'prediction_depth': 5},    # 50:1
        {'window_size': 500, 'prediction_depth': 15},   # ~33:1
        {'window_size': 750, 'prediction_depth': 30},   # 25:1
        {'window_size': 1000, 'prediction_depth': 10},  # 100:1
        {'window_size': 250, 'prediction_depth': 30}    # ~8:1
    ]
    
    config_params_list = []
    for combo in combinations:
        params = base_params.copy()
        params.update(combo)
        config_params_list.append(params)
    
    return "window_prediction_ratio", config_params_list

def main():
    # Загружаем данные для тестирования
    data_path = "data/train/btc_price_data.csv"
    prices, volumes = load_data(data_path)
    
    # Выбираем эксперименты для запуска
    # Можно добавить/удалить эксперименты, изменив порядок или раскомментировав нужные
    experiments = [
        experiment_1_significant_change_confidence,  # Влияние significant_change_pct и confidence_threshold
        experiment_2_state_length,                   # Влияние длины состояния
        experiment_3_min_samples,                    # Влияние min_samples_for_regression
        experiment_4_quantiles,                      # Влияние набора квантилей
        # experiment_5_state_length_significant_change, # Взаимодействие state_length и significant_change_pct
        # experiment_6_quantiles_confidence,          # Взаимодействие квантилей и порога уверенности
        # experiment_7_window_size,                   # Влияние размера окна
        # experiment_8_prediction_depth,              # Влияние глубины прогноза
        # experiment_9_window_prediction_ratio,       # Взаимодействие window_size и prediction_depth
    ]
    
    # Можно изменить максимальное количество итераций для ускорения тестирования
    max_iterations = 4000
    
    # Можно указать максимальное время для каждого эксперимента (в секундах)
    max_experiment_time = 3600  # 1 час
    
    results_summary = []
    
    # Запускаем выбранные эксперименты
    for experiment_func in experiments:
        print(f"\n\n{'='*50}")
        print(f"==== Запуск эксперимента: {experiment_func.__name__} ====")
        print(f"{'='*50}\n")
        
        # Получаем название эксперимента и список конфигураций
        experiment_name, config_params_list = experiment_func()
        
        # Выводим информацию о конфигурациях
        print(f"Количество конфигураций для тестирования: {len(config_params_list)}")
        print("Параметры конфигураций:")
        for i, params in enumerate(config_params_list):
            print(f"  Конфигурация {i+1}: {params}")
        print(f"Максимальное количество итераций: {max_iterations}")
        
        start_time = time.time()
        
        # Запускаем эксперимент
        experiment_dir, results_df = run_experiment(
            experiment_name, 
            config_params_list, 
            prices, 
            volumes, 
            max_iterations=max_iterations
        )
        
        # Вычисляем время выполнения эксперимента
        execution_time = time.time() - start_time
        
        if results_df is not None:
            # Находим лучшую конфигурацию
            if 'score' in results_df.columns and not results_df.empty:
                best_idx = results_df['score'].idxmax()
                best_config = results_df.loc[best_idx]
                
                print(f"\nЭксперимент завершен за {execution_time:.2f} секунд")
                print(f"Лучшая конфигурация (№{int(best_config['config_index'])}):")
                
                # Выводим параметры лучшей конфигурации
                for key, value in best_config.items():
                    if key not in ['config_index', 'execution_time']:
                        print(f"  {key}: {value}")
                
                # Сохраняем информацию об эксперименте
                results_summary.append({
                    'experiment_name': experiment_name,
                    'experiment_func': experiment_func.__name__,
                    'directory': experiment_dir,
                    'config_count': len(config_params_list),
                    'best_score': best_config['score'],
                    'best_config_index': int(best_config['config_index']),
                    'best_success_rate': best_config['success_rate'],
                    'best_coverage': best_config['coverage'],
                    'execution_time': execution_time
                })
    
    # Выводим сводку по всем экспериментам
    if results_summary:
        print("\n\n" + "="*70)
        print("==== ИТОГОВАЯ СВОДКА РЕЗУЛЬТАТОВ ЭКСПЕРИМЕНТОВ ====")
        print("="*70 + "\n")
        
        # Создаем таблицу результатов
        summary_df = pd.DataFrame(results_summary)
        if not summary_df.empty:
            print(summary_df[['experiment_name', 'best_score', 'best_success_rate', 
                             'best_coverage', 'best_config_index', 'execution_time']].to_string(index=False))
        
        # Сохраняем сводную таблицу
        timestamp = get_timestamp()
        summary_path = f"experiment_results/summary_{timestamp}.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nСводная таблица сохранена в {summary_path}")
        
        # Выводим общие рекомендации
        print("\nОБЩИЕ РЕКОМЕНДАЦИИ:")
        
        # Находим лучший эксперимент по score
        if 'best_score' in summary_df.columns:
            best_exp_idx = summary_df['best_score'].idxmax()
            best_exp = summary_df.loc[best_exp_idx]
            
            print(f"Наилучшие результаты показал эксперимент '{best_exp['experiment_name']}'")
            print(f"с конфигурацией №{best_exp['best_config_index']}")
            print(f"Успешность: {best_exp['best_success_rate']:.2f}%")
            print(f"Покрытие: {best_exp['best_coverage']:.2f}%")
            print(f"Интегральная оценка: {best_exp['best_score']:.2f}")
            print(f"Детальные результаты находятся в директории: {best_exp['directory']}")

if __name__ == "__main__":
    main()