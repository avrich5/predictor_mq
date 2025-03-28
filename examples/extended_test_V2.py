"""
extended_test_V2.py

Script for systematic testing of high-confidence, selective prediction configurations.
Based on the breakthrough with ~67% success rate in first_test_run.py.
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
from collections import defaultdict
import json

# Добавляем корневую директорию проекта в путь для импорта
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# Импорт из проекта
from config import PredictorConfig
from predictor_mq.models.hybrid_predictor import HybridPredictor

# Создаем директории для результатов, если их нет
results_base_dir = "high_conf_results"
os.makedirs(results_base_dir, exist_ok=True)

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
            'down_predictions': len(down_predictions),
            'up_success_rate': up_success,
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

def create_high_conf_config(
    significant_change_pct=0.01, 
    confidence_threshold=0.55, 
    max_coverage=0.05,
    prediction_depth=10,
    state_length=4,
    quantiles=(0.1, 0.5, 0.9),
    window_size=500,
    min_samples_for_regression=3
):
    """
    Создает конфигурацию для тестирования с высоким порогом уверенности
    """
    return PredictorConfig(
        window_size=window_size,
        prediction_depth=prediction_depth,
        state_length=state_length,
        significant_change_pct=significant_change_pct,
        quantiles=quantiles,
        min_samples_for_regression=min_samples_for_regression,
        confidence_threshold=confidence_threshold,
        max_coverage=max_coverage,
        plateau_window=500
    )

def test_config_with_tracking(prices, volumes, config_params, max_iterations=None):
    """
    Тестирует конфигурацию и отслеживает все предсказания
    
    Параметры:
    prices (numpy.array): массив цен
    volumes (numpy.array, optional): массив объемов
    config_params (dict): параметры конфигурации
    max_iterations (int, optional): максимальное количество итераций
    
    Возвращает:
    dict: результаты тестирования и анализа
    """
    print(f"\nТестирование конфигурации: {config_params}")
    
    # Создаем конфигурацию и предиктор
    config = create_high_conf_config(**config_params)
    predictor = HybridPredictor(config)
    
    # Выводим ключевую информацию о конфигурации
    print(f"Используемая конфигурация:")
    print(config)
    
    # Ограничиваем данные для быстрого тестирования, если указано
    if max_iterations:
        limited_prices = prices[:min(len(prices), max_iterations + config.prediction_depth)]
        limited_volumes = None
        if volumes is not None:
            limited_volumes = volumes[:min(len(volumes), max_iterations + config.prediction_depth)]
    else:
        limited_prices = prices
        limited_volumes = volumes
    
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
    с больше акцентом на успешность, чем на покрытие
    """
    if not result:
        return 0
    
    # Оценка за успешность более весомая для высокоточных моделей
    # Особо поощряем успешность выше 60%
    if result['success_rate'] > 60:
        success_score = result['success_rate'] * 3
    else:
        success_score = result['success_rate'] * 2
    
    # Учитываем покрытие, но с меньшим весом
    # Для высокоточных моделей даже 3-5% покрытия может быть ценным
    coverage_factor = min(1.0, result['coverage'] / 5)  # 5% покрытие = 1
    
    # Бонус за разницу между успешностью и 50% (случайное угадывание)
    accuracy_bonus = max(0, (result['success_rate'] - 50) * 2)
    
    # Баланс предсказаний (рост vs падение)
    if result['up_predictions'] > 0 and result['down_predictions'] > 0:
        balance = min(result['up_predictions'], result['down_predictions']) / max(result['up_predictions'], result['down_predictions'])
        balance_factor = (0.6 + 0.4 * balance)  # От 0.6 до 1.0 в зависимости от баланса
    else:
        balance_factor = 0.6  # Штраф за односторонние предсказания
    
    # Минимальное количество предсказаний для достоверности
    if result['total_predictions'] < 10:
        prediction_factor = result['total_predictions'] / 10
    else:
        prediction_factor = 1.0
    
    # Итоговая оценка
    score = success_score * coverage_factor * balance_factor * prediction_factor + accuracy_bonus
    
    return score

def visualize_experiment_results(results, experiment_name, save_dir):
    """
    Визуализирует результаты эксперимента
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
            'total_predictions': result['total_predictions'],
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
            label = f"{i+1}: {row['significant_change_pct']:.3f}/{row['confidence_threshold']:.2f}"
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
    
    # Размер точек зависит от количества предсказаний
    scatter = plt.scatter(results_df['coverage'], results_df['success_rate'], 
                          s=results_df['total_predictions']*2, c=results_df['config_index'], 
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
    
    # Анализируем влияние параметров
    param_columns = [col for col in results_df.columns if col not in 
                    ['config_index', 'success_rate', 'coverage', 'score', 'unique_states', 
                     'up_predictions', 'down_predictions', 'up_success_rate', 'down_success_rate', 
                     'execution_time', 'total_predictions']]
    
    for param in param_columns:
        # Проверяем, варьируется ли параметр
        if len(results_df[param].unique()) > 1:
            plt.figure(figsize=(10, 6))
            
            # Сортируем данные по параметру
            sorted_df = results_df.sort_values(param)
            
            # Строим зависимость метрик от параметра
            plt.plot(sorted_df[param], sorted_df['success_rate'], 'b-', marker='o', label='Успешность (%)')
            plt.plot(sorted_df[param], sorted_df['coverage'], 'g-', marker='s', label='Покрытие (%)')
            
            # Добавляем размер точек в зависимости от количества предсказаний
            sizes = sorted_df['total_predictions'] * 2
            for i, (x, y1, y2, s) in enumerate(zip(sorted_df[param], sorted_df['success_rate'], 
                                                 sorted_df['coverage'], sizes)):
                plt.scatter([x, x], [y1, y2], s=s, alpha=0.5, c=['blue', 'green'])
            
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

def run_experiment(experiment_name, config_params_list, prices, volumes, max_iterations=None):
    """
    Запускает эксперимент с набором конфигураций
    """
    timestamp = get_timestamp()
    experiment_dir = os.path.join(results_base_dir, f"{experiment_name}_{timestamp}")
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

def experiment_1_confidence_thresholds():
    """
    Эксперимент 1: Влияние высоких значений confidence_threshold
    """
    # Используем базовые параметры из успешного теста
    base_params = {
        'significant_change_pct': 0.01,
        'max_coverage': 0.05,
        'prediction_depth': 10,
        'state_length': 4
    }
    
    # Тестируем различные пороги уверенности
    config_params_list = []
    for confidence_threshold in [0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8]:
        params = base_params.copy()
        params['confidence_threshold'] = confidence_threshold
        config_params_list.append(params)
    
    return "high_confidence_thresholds", config_params_list

def experiment_2_max_coverage():
    """
    Эксперимент 2: Влияние различных значений max_coverage
    """
    # Используем базовые параметры из успешного теста
    base_params = {
        'significant_change_pct': 0.01,
        'confidence_threshold': 0.55,
        'prediction_depth': 10,
        'state_length': 4
    }
    
    # Тестируем различные значения max_coverage
    config_params_list = []
    for max_coverage in [0.01, 0.03, 0.05, 0.08, 0.1, 0.15]:
        params = base_params.copy()
        params['max_coverage'] = max_coverage
        config_params_list.append(params)
    
    return "max_coverage_variations", config_params_list

def experiment_3_significant_change():
    """
    Эксперимент 3: Влияние significant_change_pct
    """
    # Используем базовые параметры из успешного теста
    base_params = {
        'confidence_threshold': 0.55,
        'max_coverage': 0.05,
        'prediction_depth': 10,
        'state_length': 4
    }
    
    # Тестируем различные значения significant_change_pct
    config_params_list = []
    for significant_change_pct in [0.005, 0.008, 0.01, 0.012, 0.015]:
        params = base_params.copy()
        params['significant_change_pct'] = significant_change_pct
        config_params_list.append(params)
    
    return "significant_change_variations", config_params_list

def experiment_4_prediction_depth():
    """
    Эксперимент 4: Влияние prediction_depth
    """
    # Используем базовые параметры из успешного теста
    base_params = {
        'significant_change_pct': 0.01,
        'confidence_threshold': 0.55,
        'max_coverage': 0.05,
        'state_length': 4
    }
    
    # Тестируем различные значения prediction_depth
    config_params_list = []
    for prediction_depth in [5, 8, 10, 12, 15, 20]:
        params = base_params.copy()
        params['prediction_depth'] = prediction_depth
        config_params_list.append(params)
    
    return "prediction_depth_variations", config_params_list

def experiment_5_state_length():
    """
    Эксперимент 5: Влияние state_length
    """
    # Используем базовые параметры из успешного теста
    base_params = {
        'significant_change_pct': 0.01,
        'confidence_threshold': 0.55,
        'max_coverage': 0.05,
        'prediction_depth': 10
    }
    
    # Тестируем различные значения state_length
    config_params_list = []
    for state_length in [3, 4, 5, 6, 7]:
        params = base_params.copy()
        params['state_length'] = state_length
        config_params_list.append(params)
    
    return "state_length_variations", config_params_list

def experiment_6_optimal_combinations():
    """
    Эксперимент 6: Комбинации лучших параметров из предыдущих экспериментов
    (Предполагается, что эксперименты 1-5 уже проведены и проанализированы)
    """
    # Эти значения нужно заменить на основе результатов предыдущих экспериментов
    best_confidence = 0.55  # Пример - необходимо заменить на основе эксперимента 1
    best_max_coverage = 0.05  # Пример - необходимо заменить на основе эксперимента 2
    best_significant_change = 0.01  # Пример - необходимо заменить на основе эксперимента 3
    best_prediction_depth = 10  # Пример - необходимо заменить на основе эксперимента 4
    best_state_length = 4  # Пример - необходимо заменить на основе эксперимента 5
    
    # Создаем комбинации из лучших параметров с небольшими вариациями
    config_params_list = [
        # Оптимальная конфигурация на основе предыдущих экспериментов
        {
            'confidence_threshold': best_confidence,
            'max_coverage': best_max_coverage,
            'significant_change_pct': best_significant_change,
            'prediction_depth': best_prediction_depth,
            'state_length': best_state_length
        },
        # Вариация 1: Немного более строгие условия для более высокой точности
        {
            'confidence_threshold': best_confidence + 0.05,
            'max_coverage': best_max_coverage * 0.8,
            'significant_change_pct': best_significant_change * 1.1,
            'prediction_depth': best_prediction_depth,
            'state_length': best_state_length
        },
        # Вариация 2: Немного более мягкие условия для большего покрытия
        {
            'confidence_threshold': best_confidence - 0.05,
            'max_coverage': best_max_coverage * 1.2,
            'significant_change_pct': best_significant_change * 0.9,
            'prediction_depth': best_prediction_depth,
            'state_length': best_state_length
        },
        # Вариация 3: Фокус на краткосрочных предсказаниях
        {
            'confidence_threshold': best_confidence,
            'max_coverage': best_max_coverage,
            'significant_change_pct': best_significant_change,
            'prediction_depth': int(best_prediction_depth * 0.7),
            'state_length': best_state_length
        },
        # Вариация 4: Фокус на долгосрочных предсказаниях
        {
            'confidence_threshold': best_confidence,
            'max_coverage': best_max_coverage,
            'significant_change_pct': best_significant_change,
            'prediction_depth': int(best_prediction_depth * 1.3),
            'state_length': best_state_length
        }
    ]
    
    return "optimal_combinations", config_params_list

def experiment_7_risk_reward_balance():
    """
    Эксперимент 7: Баланс риска и доходности через соотношение confidence_threshold и significant_change_pct
    """
    # Базовые параметры
    base_params = {
        'max_coverage': 0.05,
        'prediction_depth': 10,
        'state_length': 4
    }
    
    # Комбинации параметров, представляющие разные профили риск-доходность
    config_params_list = [
        # Высокий риск, высокая доходность (низкая уверенность, большие изменения)
        {**base_params, 'confidence_threshold': 0.4, 'significant_change_pct': 0.015},
        
        # Средний риск, средняя доходность (средняя уверенность, средние изменения)
        {**base_params, 'confidence_threshold': 0.55, 'significant_change_pct': 0.01},
        
        # Низкий риск, низкая доходность (высокая уверенность, малые изменения)
        {**base_params, 'confidence_threshold': 0.7, 'significant_change_pct': 0.005},
        
        # Асимметричный профиль 1 (высокая уверенность, большие изменения)
        {**base_params, 'confidence_threshold': 0.7, 'significant_change_pct': 0.015},
        
        # Асимметричный профиль 2 (низкая уверенность, малые изменения)
        {**base_params, 'confidence_threshold': 0.4, 'significant_change_pct': 0.005}
    ]
    
    return "risk_reward_profiles", config_params_list

def experiment_8_window_size():
    """
    Эксперимент 8: Влияние размера окна (window_size)
    """
    # Базовые параметры
    base_params = {
        'significant_change_pct': 0.01,
        'confidence_threshold': 0.55,
        'max_coverage': 0.05,
        'prediction_depth': 10,
        'state_length': 4
    }
    
    # Тестируем различные размеры окна
    config_params_list = []
    for window_size in [300, 400, 500, 600, 750]:
        params = base_params.copy()
        params['window_size'] = window_size
        config_params_list.append(params)
    
    return "window_size_variations", config_params_list

def main():
    # Загружаем данные для тестирования
    data_path = "data/train/btc_price_data.csv"  # Используем основной файл данных
    prices, volumes = load_data(data_path)
    
    # Создаем временную метку для текущего запуска
    run_timestamp = get_timestamp()
    print(f"Запуск тестирования высокоточных конфигураций: {run_timestamp}")
    
    # Выбираем эксперименты для запуска
    # Можно комментировать/раскомментировать нужные эксперименты
    experiments = [
        experiment_1_confidence_thresholds,  # Влияние порога уверенности
        experiment_2_max_coverage,           # Влияние максимального покрытия
        experiment_3_significant_change,     # Влияние порога изменения
        experiment_4_prediction_depth,       # Влияние глубины прогноза
        experiment_5_state_length,           # Влияние длины состояния
        # experiment_6_optimal_combinations,   # Комбинации лучших параметров (запускать после анализа экспериментов 1-5)
        experiment_7_risk_reward_balance,    # Баланс риска и доходности
        experiment_8_window_size,            # Влияние размера окна
    ]
    
    # Можно ограничить количество итераций для ускорения экспериментов
    # Для полноценного тестирования рекомендуется использовать None
    max_iterations = 9000  # или None для полного набора данных
    
    results_summary = []
    
    # Запускаем выбранные эксперименты
    for experiment_func in experiments:
        print(f"\n\n{'='*70}")
        print(f"==== Запуск эксперимента: {experiment_func.__name__} ====")
        print(f"{'='*70}\n")
        
        # Получаем название эксперимента и список конфигураций
        experiment_name, config_params_list = experiment_func()
        
        # Выводим информацию о конфигурациях
        print(f"Количество конфигураций для тестирования: {len(config_params_list)}")
        print("Параметры конфигураций:")
        for i, params in enumerate(config_params_list):
            print(f"  Конфигурация {i+1}: {params}")
        print(f"Максимальное количество итераций: {max_iterations or 'Все данные'}")
        
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
        
        if results_df is not None and not results_df.empty:
            # Находим лучшую конфигурацию
            if 'score' in results_df.columns:
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
                    'best_total_predictions': best_config['total_predictions'],
                    'execution_time': execution_time
                })
    
    # Выводим сводку по всем экспериментам
    if results_summary:
        print("\n\n" + "="*80)
        print("==== ИТОГОВАЯ СВОДКА РЕЗУЛЬТАТОВ ЭКСПЕРИМЕНТОВ ====")
        print("="*80 + "\n")
        
        # Создаем таблицу результатов
        summary_df = pd.DataFrame(results_summary)
        if not summary_df.empty:
            print(summary_df[['experiment_name', 'best_score', 'best_success_rate', 
                             'best_coverage', 'best_total_predictions', 'best_config_index', 'execution_time']].to_string(index=False))
        
        # Сохраняем сводную таблицу
        summary_path = f"{results_base_dir}/summary_{run_timestamp}.csv"
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
            print(f"Количество предсказаний: {best_exp['best_total_predictions']}")
            print(f"Интегральная оценка: {best_exp['best_score']:.2f}")
            print(f"Детальные результаты находятся в директории: {best_exp['directory']}")
            print("\nДля использования лучшей конфигурации обновите параметры в config.py")

if __name__ == "__main__":
    main()