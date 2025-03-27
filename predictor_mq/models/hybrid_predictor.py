"""
hybrid_predictor.py

Оптимизированная реализация гибридного предиктора, объединяющего марковские цепи 
и квантильную регрессию для прогнозирования временных рядов.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler
from .standardized_quantile_regression import QuantileRegressionModel
from collections import OrderedDict  

# class QuantileRegressionModel:
#     """
#     Модель для предсказания квантилей будущего изменения цены.
#     """
    
#     def __init__(self, quantiles, alpha=0.1, min_samples=5, solver='highs'):
#         """
#         Инициализация модели квантильной регрессии
        
#         Параметры:
#         quantiles (tuple): квантили для предсказания
#         alpha (float): параметр регуляризации для модели
#         min_samples (int): минимальное количество образцов для обучения
#         solver (str): решатель для квантильной регрессии
#         """
#         self.quantiles = quantiles
#         self.alpha = alpha
#         self.min_samples = min_samples
#         self.solver = solver
#         self.models = {}  # Словарь с моделями для разных квантилей
#         self.scaler = StandardScaler()  # Нормализация признаков
#         self.is_fitted = False  # Флаг, указывающий, обучена ли модель
    
#     def fit(self, X, y):
#         """
#         Обучает модель на исторических данных
        
#         Параметры:
#         X (numpy.array): признаки (каждая строка - вектор признаков для одного наблюдения)
#         y (numpy.array): целевые значения (процентное изменение цены)
        
#         Возвращает:
#         self: обученная модель
#         """
#         # Проверяем, что данных достаточно для обучения
#         if len(X) < self.min_samples:
#             self.is_fitted = False
#             return self
        
#         # Нормализуем признаки
#         X_scaled = self.scaler.fit_transform(X)
        
#         # Обучаем модель для каждого квантиля
#         for q in self.quantiles:
#             try:
#                 model = QuantileRegressor(quantile=q, alpha=self.alpha, solver=self.solver)
#                 model.fit(X_scaled, y)
#                 self.models[q] = model
#             except Exception as e:
#                 print(f"Ошибка при обучении модели для квантиля {q}: {e}")
#                 continue
        
#         self.is_fitted = len(self.models) > 0
#         return self
    
#     def predict_single(self, X):
#         """
#         Делает предсказание для одного наблюдения
        
#         Параметры:
#         X (numpy.array): вектор признаков
        
#         Возвращает:
#         dict: предсказания для разных квантилей
#         """
#         if not self.is_fitted:
#             return None
        
#         # Проверяем формат входных данных
#         X_reshaped = X.reshape(1, -1) if len(X.shape) == 1 else X
        
#         # Проверяем размерность признаков
#         if X_reshaped.shape[1] != self.scaler.n_features_in_:
#             return None
        
#         # Нормализуем признаки
#         X_scaled = self.scaler.transform(X_reshaped)
        
#         # Делаем предсказания для каждого квантиля
#         predictions = {}
#         for q, model in self.models.items():
#             predictions[q] = model.predict(X_scaled)[0]
        
#         return predictions


class HybridPredictor:
    """
    Гибридный предиктор на основе марковского процесса и квантильной регрессии.
    """
    
    def __init__(self, config):
        """
        Инициализация предиктора
        
        Параметры:
        config: конфигурация параметров предиктора
        """
        self.config = config
        
        # Статистика предсказаний
        self.total_predictions = 0
        self.correct_predictions = 0
        self.success_rate = 0.0
        self.point_statistics = {}
        
        # Статистика по состояниям
        self.state_statistics = defaultdict(lambda: {'total': 0, 'correct': 0})
        
        # Кэш изменений цен
        self.price_changes = None
        
        # Квантильные модели для состояний
        self.quantile_models = {}
        
        # Настройки для моделей регрессии берем из конфигурации
        self.min_samples = config.min_samples_for_regression
        self.solver = config.regression_solver
        
        # Базовая модель
        self.base_quantile_model = QuantileRegressionModel(
            quantiles=self.config.quantiles,
            alpha=self.config.regression_alpha,
            min_samples=self.config.min_samples_for_regression,
            solver=self.config.regression_solver
        )
        
        # Вывод основных параметров
        print(f"Инициализация гибридного предиктора с параметрами:")
        print(f"- window_size: {self.config.window_size}")
        print(f"- prediction_depth: {self.config.prediction_depth}")
        print(f"- state_length: {self.config.state_length}")
        print(f"- quantiles: {self.config.quantiles}")
        print(f"- confidence_threshold: {self.config.confidence_threshold}")

    def reset(self):
        """Сбрасывает состояние предиктора перед новым запуском."""
        self.current_state = None
        self.state_history = []
        self.models = {}  # Сброс моделей для каждого состояния
        self.point_statistics = {}  # Сброс статистики по точкам
        self.total_predictions = 0
        self.correct_predictions = 0
        self.success_rate = 0.0
        self.success_rate_history = []  # Сброс истории успешности
        self.training_samples = {}  # Сброс обучающих образцов

    def _precompute_changes(self, prices):
        """Предварительно вычисляет относительные изменения цен с векторизацией"""
        if self.price_changes is None:
            n = len(prices) - 1
            self.price_changes = np.zeros(n)
            
            # Векторизованный расчет изменений
            mask = prices[:-1] != 0
            price_diffs = np.diff(prices)
            self.price_changes[mask] = price_diffs[mask] / prices[:-1][mask]

    def _calculate_dynamic_threshold(self, prices, idx):
        """Вычисляет динамический порог для определения значимого изменения с кэшированием"""
        # Инициализируем кэш, если его нет
        if not hasattr(self, '_threshold_cache'):
            self._threshold_cache = {}
        
        # Инициализируем максимальный размер кэша, если он не задан
        if not hasattr(self, '_threshold_cache_max_size'):
            self._threshold_cache_max_size = 10000
        
        # Проверяем, есть ли результат в кэше
        if idx in self._threshold_cache:
            return self._threshold_cache[idx]
        
        # Оригинальная логика вычисления порога
        start_idx = max(0, idx - self.config.window_size)
        if start_idx >= len(self.price_changes):
            return self.config.significant_change_pct
        
        changes = self.price_changes[start_idx:idx-1]
        if len(changes) < self.config.min_samples_for_threshold:
            return self.config.significant_change_pct
        
        # Используем указанный в конфигурации процентиль
        threshold = np.percentile(np.abs(changes), self.config.threshold_percentile)
        
        # Контроль размера кэша
        if len(self._threshold_cache) >= self._threshold_cache_max_size:
            # Удаляем первый элемент (наиболее старый)
            self._threshold_cache.pop(next(iter(self._threshold_cache.keys())))
        
        # Сохраняем в кэш
        self._threshold_cache[idx] = threshold
        return threshold
    
    def _determine_movement(self, current_price, next_price, threshold):
        """Определяет направление движения с учетом порога значимого изменения"""
        if current_price == 0:
            return 0  # Избегаем деления на ноль
        
        pct_change = (next_price - current_price) / current_price
        
        if pct_change > threshold:
            return 1  # Значимый рост
        elif pct_change < -threshold:
            return 2  # Значимое падение
        else:
            return 0  # Незначительное изменение
    
    def _get_state(self, prices, idx, threshold):
        """Определяет текущее состояние рынка с кэшированием"""
        from collections import OrderedDict  # Перемещаем импорт внутрь метода
        
        if idx < self.config.state_length:
            return None  # Недостаточно данных
        
        # Инициализируем кэш, если его нет
        if not hasattr(self, '_state_cache') or not isinstance(self._state_cache, OrderedDict):
            self._state_cache = OrderedDict()
        
        # Инициализируем максимальный размер кэша, если он не задан
        if not hasattr(self, '_state_cache_max_size'):
            self._state_cache_max_size = 10000  # Максимальный размер кэша
        
        # Используем индекс и порог как ключ кэша
        cache_key = (idx, threshold)
        if cache_key in self._state_cache:
            # Перемещаем к концу для LRU-стратегии
            value = self._state_cache.pop(cache_key)
            self._state_cache[cache_key] = value
            return value
        
        # Оригинальная логика получения состояния
        movements = []
        for i in range(idx - self.config.state_length, idx):
            movement = self._determine_movement(prices[i], prices[i+1], threshold)
            if movement == 0:
                if len(movements) > 0:
                    movement = movements[-1]  # Продолжаем предыдущее движение
                else:
                    movement = self.config.default_movement  # По умолчанию нейтральное движение
            movements.append(movement)
        
        state = tuple(movements)
        
        # Контроль размера кэша
        if len(self._state_cache) >= self._state_cache_max_size:
            self._state_cache.popitem(last=False)  # Удаляем самый старый элемент
        
        self._state_cache[cache_key] = state
        return state
    
    def _determine_outcome(self, prices, idx, threshold):
        """Определяет фактический исход через prediction_depth точек"""
        if idx + self.config.prediction_depth >= len(prices):
            return None  # Нет данных для проверки
        
        current_price = prices[idx]
        future_price = prices[idx + self.config.prediction_depth]
        
        return self._determine_movement(current_price, future_price, threshold)
    
    def _extract_features(self, prices, idx):
        """Извлекает признаки для квантильной регрессии с обработкой ошибок"""
        try:
            start_idx = max(0, idx - self.config.window_size)
            window = prices[start_idx:idx+1]
            
            # Если недостаточно данных, возвращаем нулевой вектор размерности из конфигурации
            if len(window) < self.config.min_window_for_features:
                return np.zeros(self.config.basic_feature_count)
            
            features = []
            
            # 1. Последнее изменение
            try:
                recent_change = prices[idx] / prices[idx-1] - 1 if prices[idx-1] != 0 else 0
            except Exception as e:
                print(f"Ошибка при расчете последнего изменения для idx={idx}: {e}")
                recent_change = 0
            features.append(recent_change)
            
            # 2. Волатильность
            try:
                changes = np.array([
                    prices[i+1]/prices[i] - 1 
                    for i in range(start_idx, idx) 
                    if prices[i] != 0 and i+1 <= idx
                ])
                volatility = np.std(changes) if len(changes) > 0 else 0
            except Exception as e:
                print(f"Ошибка при расчете волатильности для idx={idx}: {e}")
                volatility = 0
            features.append(volatility)
            
            # 3. Отклонение от скользящей средней
            try:
                ma_period = min(self.config.ma_period, len(window))
                if ma_period > 0:
                    sma = np.mean(window[-ma_period:])
                    deviation = prices[idx] / sma - 1 if sma != 0 else 0
                else:
                    deviation = 0
            except Exception as e:
                print(f"Ошибка при расчете отклонения от скользящей средней для idx={idx}: {e}")
                deviation = 0
            features.append(deviation)
            
            # 4. Средняя скорость изменения
            try:
                speed_period = self.config.speed_period
                if len(window) >= speed_period:
                    speed = (prices[idx] / prices[idx-speed_period+1] - 1) / (speed_period-1) if prices[idx-speed_period+1] != 0 else 0
                else:
                    speed = 0
            except Exception as e:
                print(f"Ошибка при расчете средней скорости изменения для idx={idx}: {e}")
                speed = 0
            features.append(speed)
            
            # 5. Ускорение изменения
            try:
                accel_period = self.config.accel_period
                if len(window) >= accel_period:
                    prev_speed = (prices[idx-speed_period+1] / prices[idx-accel_period+1] - 1) / (speed_period-1) if prices[idx-accel_period+1] != 0 else 0
                    acceleration = speed - prev_speed
                else:
                    acceleration = 0
            except Exception as e:
                print(f"Ошибка при расчете ускорения изменения для idx={idx}: {e}")
                acceleration = 0
            features.append(acceleration)
            
            # Добавляем дополнительные признаки из конфигурации
            if hasattr(self.config, 'use_extended_features') and self.config.use_extended_features:
                try:
                    for feature_func in self.config.extended_feature_functions:
                        feature_value = feature_func(prices, idx, window)
                        features.append(feature_value)
                except Exception as e:
                    print(f"Ошибка при расчете дополнительных признаков для idx={idx}: {e}")
                    # Добавляем нули вместо неудачно вычисленных признаков
                    for _ in range(len(self.config.extended_feature_functions)):
                        features.append(0)
            
            return np.array(features)
        except Exception as e:
            print(f"Общая ошибка при извлечении признаков для idx={idx}: {e}")
            # Возвращаем вектор нулей в случае ошибки
            return np.zeros(self.config.basic_feature_count + 
                        (len(self.config.extended_feature_functions) if hasattr(self.config, 'use_extended_features') 
                            and self.config.use_extended_features else 0))
    
    def _extract_features_optimized(self, prices, idx):
        """Извлекает признаки для квантильной регрессии с оптимизацией"""
        try:
            start_idx = max(0, idx - self.config.window_size)
            window = prices[start_idx:idx+1]
            
            if len(window) < self.config.min_window_for_features:
                return np.zeros(self.config.basic_feature_count)
            
            # Инициализируем массив признаков
            features = []
            
            # Предварительно вычисляем скользящие средние для всего окна
            if not hasattr(self, '_ma_cache'):
                self._ma_cache = {}
                self._ma_cache_max_size = 1000  # Ограничиваем размер кэша
            
            window_key = (start_idx, idx)
            ma_period = min(self.config.ma_period, len(window))
            
            # Получаем значение скользящей средней из кэша или вычисляем
            if window_key not in self._ma_cache:
                if ma_period > 0:
                    # Эффективный расчет скользящей средней с использованием свертки
                    kernel = np.ones(ma_period) / ma_period
                    ma_values = np.convolve(window, kernel, mode='valid')
                    
                    # Контролируем размер кэша
                    if len(self._ma_cache) >= self._ma_cache_max_size:
                        # Удаляем первый элемент (наиболее старый)
                        self._ma_cache.pop(next(iter(self._ma_cache)))
                    
                    self._ma_cache[window_key] = ma_values
            
            # Оптимизируем вычисление изменений цен для всего окна сразу
            price_ratios = np.zeros(len(window) - 1)
            valid_indices = prices[start_idx:idx] != 0
            price_ratios[valid_indices] = prices[start_idx+1:idx+1][valid_indices] / prices[start_idx:idx][valid_indices] - 1
            
            # 1. Последнее изменение (берем из предварительно вычисленных значений)
            try:
                recent_change = price_ratios[-1] if len(price_ratios) > 0 else 0
            except Exception as e:
                print(f"Ошибка при расчете последнего изменения для idx={idx}: {e}")
                recent_change = 0
            features.append(recent_change)
            
            # 2. Волатильность (используем предварительно вычисленные изменения)
            try:
                changes = price_ratios[~np.isnan(price_ratios)]  # Фильтруем NaN значения
                volatility = np.std(changes) if len(changes) > 0 else 0
            except Exception as e:
                print(f"Ошибка при расчете волатильности для idx={idx}: {e}")
                volatility = 0
            features.append(volatility)
            
            # 3. Отклонение от скользящей средней
            try:
                if ma_period > 0:
                    # Используем кэшированное значение
                    ma_values = self._ma_cache.get(window_key)
                    if ma_values is not None and len(ma_values) > 0:
                        sma = ma_values[-1]  # Последнее значение скользящей средней
                    else:
                        sma = np.mean(window[-ma_period:])
                    deviation = prices[idx] / sma - 1 if sma != 0 else 0
                else:
                    deviation = 0
            except Exception as e:
                print(f"Ошибка при расчете отклонения от скользящей средней для idx={idx}: {e}")
                deviation = 0
            features.append(deviation)
            
            # 4. Средняя скорость изменения (оптимизированное вычисление)
            try:
                speed_period = self.config.speed_period
                if len(window) >= speed_period:
                    # Векторизованное вычисление
                    if prices[idx-speed_period+1] != 0:
                        speed = (prices[idx] / prices[idx-speed_period+1] - 1) / (speed_period-1)
                    else:
                        speed = 0
                else:
                    speed = 0
            except Exception as e:
                print(f"Ошибка при расчете средней скорости изменения для idx={idx}: {e}")
                speed = 0
            features.append(speed)
            
            # 5. Ускорение изменения
            try:
                accel_period = self.config.accel_period
                if len(window) >= accel_period:
                    if prices[idx-accel_period+1] != 0 and prices[idx-speed_period+1] != 0:
                        prev_speed = (prices[idx-speed_period+1] / prices[idx-accel_period+1] - 1) / (speed_period-1)
                        acceleration = speed - prev_speed
                    else:
                        acceleration = 0
                else:
                    acceleration = 0
            except Exception as e:
                print(f"Ошибка при расчете ускорения изменения для idx={idx}: {e}")
                acceleration = 0
            features.append(acceleration)
            
            # Добавляем дополнительные признаки из конфигурации
            if hasattr(self.config, 'use_extended_features') and self.config.use_extended_features:
                try:
                    for feature_func in self.config.extended_feature_functions:
                        feature_value = feature_func(prices, idx, window)
                        features.append(feature_value)
                except Exception as e:
                    print(f"Ошибка при расчете дополнительных признаков для idx={idx}: {e}")
                    # Добавляем нули вместо неудачно вычисленных признаков
                    for _ in range(len(self.config.extended_feature_functions)):
                        features.append(0)
            
            return np.array(features)
        except Exception as e:
            print(f"Общая ошибка при извлечении признаков для idx={idx}: {e}")
            # Возвращаем вектор нулей в случае ошибки
            return np.zeros(self.config.basic_feature_count + 
                        (len(self.config.extended_feature_functions) if hasattr(self.config, 'use_extended_features') 
                            and self.config.use_extended_features else 0))
    
    def _extract_state_features(self, prices, idx, threshold):
        """Извлекает признаки, связанные с текущим марковским состоянием"""
        current_state = self._get_state(prices, idx, threshold)
        if current_state is None:
            return np.zeros(self.config.state_feature_count + self.config.state_length)
        
        features = []
        
        # 1. Статистика по текущему состоянию
        start_idx = max(self.config.state_length, idx - self.config.window_size)
        transitions = defaultdict(lambda: {1: 0, 2: 0})
        
        for i in range(start_idx, idx - self.config.prediction_depth + 1):
            state = self._get_state(prices, i, threshold)
            if state != current_state:
                continue
            
            outcome = self._determine_outcome(prices, i, threshold)
            if outcome is None or outcome == 0:
                continue
            
            transitions[state][outcome] += 1
        
        # Вероятности переходов
        total = sum(transitions[current_state].values()) if current_state in transitions else 0
        up_prob = transitions[current_state].get(1, 0) / total if total > 0 else self.config.default_probability
        down_prob = transitions[current_state].get(2, 0) / total if total > 0 else self.config.default_probability
        
        features.append(up_prob)
        features.append(down_prob)
        features.append(total / self.config.window_size)  # Частота состояния
        
        # Дополнительные признаки состояния
        for feature_func in self.config.state_feature_functions:
            features.append(feature_func(current_state, transitions, total))
        
        # 2. Кодирование состояния
        for movement in current_state:
            features.append(float(movement) / self.config.movement_normalization)
        
        return np.array(features)
    
    def _collect_features(self, prices, idx, threshold):
        """Собирает все признаки для гибридной модели"""
        basic_features = self._extract_features(prices, idx)
        state_features = self._extract_state_features(prices, idx, threshold)
        return np.concatenate([basic_features, state_features])
    
    def _collect_features_optimized(self, prices, idx, threshold):
        """Собирает все признаки для гибридной модели с оптимизацией"""
        basic_features = self._extract_features_optimized(prices, idx)
        state_features = self._extract_state_features(prices, idx, threshold)
        return np.concatenate([basic_features, state_features])
    
    def _update_quantile_models(self, prices, results):
        """
        Обновляет модели квантильной регрессии на основе накопленных данных
        
        Параметры:
        prices (numpy.array): массив цен
        results (list): результаты предсказаний
        """
        # Группируем данные по состояниям
        state_data = defaultdict(lambda: {'X': [], 'y': []})
        
        for r in results:
            if 'state' not in r or r['state'] is None:
                continue
                    
            idx = r['index']
            state = r['state']
            
            # Пропускаем точки, для которых нет данных для проверки
            if idx + self.config.prediction_depth >= len(prices):
                continue
            
            # Собираем признаки
            threshold = self._calculate_dynamic_threshold(prices, idx)
            features = self._extract_features(prices, idx)
            
            # Определяем целевое значение (процентное изменение через prediction_depth точек)
            current_price = prices[idx]
            future_price = prices[idx + self.config.prediction_depth]
            if current_price != 0:
                pct_change = future_price / current_price - 1
                
                state_data[state]['X'].append(features)
                state_data[state]['y'].append(pct_change)
        
        # Отладочный вывод
        print(f"\nОбновление моделей: собрано образцов для {len(state_data)} состояний")
        
        # Обучаем модели для каждого состояния
        models_updated = 0
        for state, data in state_data.items():
            if len(data['X']) >= self.config.min_samples_for_regression:
                X = np.array(data['X'])
                y = np.array(data['y'])
                
                model = QuantileRegressionModel(
                    quantiles=self.config.quantiles,
                    alpha=self.config.regression_alpha,
                    min_samples=self.config.min_samples_for_regression,
                    solver=self.config.regression_solver
                )
                model.fit(X, y)
                self.quantile_models[state] = model
                models_updated += 1
        
        print(f"Обновлено {models_updated} моделей, всего моделей: {len(self.quantile_models)}")
        
        # Обучаем базовую модель на всех данных
        all_X = []
        all_y = []
        
        for data in state_data.values():
            all_X.extend(data['X'])
            all_y.extend(data['y'])
        
        if len(all_X) >= self.config.min_samples_for_regression:
            self.base_quantile_model.fit(np.array(all_X), np.array(all_y))
            print(f"Базовая модель обучена на {len(all_X)} образцах")

    def predict_at_point(self, prices, idx, max_predictions=None, current_predictions=0):
        """
        Делает предсказание для точки
        
        Параметры:
        prices (numpy.array): массив цен
        idx (int): индекс точки
        max_predictions (int): максимальное количество предсказаний
        current_predictions (int): текущее количество предсказаний
        
        Возвращает:
        dict: результат предсказания
        """
        # Базовые проверки
        if idx < self.config.window_size or idx < self.config.state_length:
            return {'prediction': 0, 'confidence': 0.0}
        
        if max_predictions is not None and current_predictions >= max_predictions:
            return {'prediction': 0, 'confidence': 0.0}
        
        # Вычисляем динамический порог
        dynamic_threshold = self._calculate_dynamic_threshold(prices, idx)
        
        # Получаем текущее состояние
        current_state = self._get_state(prices, idx, dynamic_threshold)
        if current_state is None:
            return {'prediction': 0, 'confidence': 0.0}
        
        # Отладочный вывод для выборочных точек
        if idx % 1000 == 0:
            print(f"Debug at idx={idx}: State={current_state}, Threshold={dynamic_threshold:.6f}")
        
        # Собираем признаки для модели
        features = self._extract_features(prices, idx)
        
        # Если у нас есть обученная модель для текущего состояния, используем ее
        if current_state in self.quantile_models:
            model = self.quantile_models[current_state]
            predictions = model.predict_single(features)
            if idx % 1000 == 0 and predictions:
                print(f"  Using state model, predictions: {predictions}")
        else:
            # Если модели нет, используем базовую модель
            if self.base_quantile_model.is_fitted:
                predictions = self.base_quantile_model.predict_single(features)
                if idx % 1000 == 0 and predictions:
                    print(f"  Using base model, predictions: {predictions}")
            else:
                if idx % 1000 == 0:
                    print(f"  No models available")
                return {'prediction': 0, 'confidence': 0.0}
        
        # Если не удалось получить предсказания, возвращаем пустой результат
        if predictions is None:
            if idx % 1000 == 0:
                print(f"  Predictions is None")
            return {'prediction': 0, 'confidence': 0.0}
        
        # Определяем квантили для расчета
        available_quantiles = sorted(predictions.keys())
        
        # Находим медиану и нужные квантили для расчета уверенности
        median_q = min(available_quantiles, key=lambda q: abs(q - 0.5))
        median = predictions[median_q]
        
        # Находим нижний и верхний квантили
        lower_q = min(available_quantiles, key=lambda q: abs(q - 0.1))
        lower = predictions[lower_q]
        
        upper_q = min(available_quantiles, key=lambda q: abs(q - 0.9))
        upper = predictions[upper_q]
        
        # Отладочная информация о квантилях
        if idx % 1000 == 0:
            print(f"  Median: {median:.6f}, Lower: {lower:.6f}, Upper: {upper:.6f}")
        
        # Определяем направление на основе медианы и порога
        prediction = 0
        if median > dynamic_threshold:
            prediction = 1  # рост
        elif median < -dynamic_threshold:
            prediction = 2  # падение
        
        # # Рассчитываем уверенность на основе распределения квантилей
        # confidence = 0.0
        # if prediction == 1:  # Рост
        #     # Уверенность тем выше, чем дальше нижний квантиль от нуля
        #     confidence = min(1.0, max(0, lower / dynamic_threshold + 0.5))
        # elif prediction == 2:  # Падение
        #     # Уверенность тем выше, чем дальше верхний квантиль от нуля (в отрицательную сторону)
        #     confidence = min(1.0, max(0, -upper / dynamic_threshold + 0.5))

        # Рассчитываем уверенность по-новому
        """
        Эта формула основана на абсолютном значении медианы, а не на нижнем или верхнем квантиле, 
        и дает значение уверенности не меньше 0.5 для любого предсказания, которое преодолевает порог.
        """
        if prediction == 1:  # Рост
            # Используем абсолютное значение и пропорцию от порога
            confidence = min(1.0, 0.5 + abs(median) / dynamic_threshold * 0.5)
        elif prediction == 2:  # Падение
            # Используем абсолютное значение и пропорцию от порога
            confidence = min(1.0, 0.5 + abs(median) / dynamic_threshold * 0.5)
        else:
            confidence = 0.0

        # Отладочная информация о решении
        if idx % 1000 == 0:
            print(f"  Decision: prediction={prediction}, confidence={confidence:.4f}, threshold={self.config.confidence_threshold}")
        
        # Применяем фильтр уверенности
        if confidence < self.config.confidence_threshold:
            prediction = 0
            confidence = 0.0
        
        # Формируем итоговый результат
        result = {
            'prediction': prediction,
            'confidence': confidence,
            'state': current_state,
            'quantile_predictions': {q: float(val) for q, val in predictions.items()} if predictions else {}
        }
        
        return result
            
    def run_on_data(self, prices, volumes=None, verbose=True, 
                    detect_plateau=True, plateau_window=None, 
                    min_predictions=None, min_success_rate=None):
        """
        Запускает модель на всем наборе данных с обнаружением плато
        
        Параметры:
        prices (numpy.array): массив цен
        volumes (numpy.array, optional): массив объемов торгов
        verbose (bool): выводить информацию о прогрессе
        detect_plateau (bool): включить обнаружение плато
        plateau_window (int): размер окна для обнаружения плато
        min_predictions (int): минимальное количество предсказаний для оценки плато
        min_success_rate (float): минимальная успешность для оценки плато
        
        Возвращает:
        list: результаты предсказаний
        """
        import numpy as np
        from tqdm import tqdm

        # Устанавливаем значения по умолчанию из конфигурации, если не указаны
        if plateau_window is None:
            plateau_window = self.config.plateau_window
        if min_predictions is None:
            min_predictions = self.config.min_predictions_for_plateau
        if min_success_rate is None:
            min_success_rate = self.config.min_success_rate_for_plateau
        
        # Сбрасываем состояние предиктора
        self.reset()
        
        # Проверяем, достаточно ли данных
        if len(prices) < self.config.window_size + 1:
            print("Недостаточно данных для обработки.")
            return []
        
        # Предварительно вычисляем изменения цен
        self._precompute_changes(prices)
        
        # Инициализируем кэши
        if not hasattr(self, '_state_cache'):
            self._state_cache = {}
        
        if not hasattr(self, '_threshold_cache'):
            self._threshold_cache = {}
        
        results = []
            
        # Начинаем с точки, где у нас достаточно данных для анализа
        min_idx = max(self.config.window_size, self.config.state_length)
        
        # Вычисляем максимальное количество предсказаний
        max_predictions = int(len(prices) * self.config.max_coverage)
        current_predictions = 0
        
        # Для обнаружения сходимости
        success_rate_history = []
        last_prediction_idx = None
        plateau_start_idx = None
        
        # Создаем диапазон индексов для итерации
        indices = range(min_idx, len(prices) - self.config.prediction_depth)
        
        # Инициализация прогресс-бара
        pbar = tqdm(total=len(indices), 
                    desc="Processing", 
                    mininterval=1.0,  # Обновляем прогресс-бар раз в секунду
                    disable=not verbose)
        
        # Проходим по всем точкам
        for idx in indices:
            # Делаем предсказание
            pred_result = self.predict_at_point(prices, idx, max_predictions, current_predictions)
            prediction = pred_result['prediction']
            
            # Получаем состояние для этой точки
            dynamic_threshold = self._calculate_dynamic_threshold(prices, idx)
            current_state = self._get_state(prices, idx, dynamic_threshold)
            
            # Если предсказание не "не знаю", проверяем результат
            if prediction != 0:
                current_predictions += 1
                actual_outcome = self._determine_outcome(prices, idx, dynamic_threshold)
                
                # Пропускаем проверку, если результат незначительное изменение (0)
                if actual_outcome is None or actual_outcome == 0:
                    pbar.update(1)
                    continue
                
                is_correct = (prediction == actual_outcome)
                
                # Обновляем статистику
                self.total_predictions += 1
                if is_correct:
                    self.correct_predictions += 1
                
                # Обновляем успешность
                self.success_rate = self.correct_predictions / self.total_predictions if self.total_predictions > 0 else 0
                success_rate_history.append(self.success_rate)
                
                # Запоминаем индекс последнего предсказания
                last_prediction_idx = idx
                
                # Сохраняем статистику для этой точки
                self.point_statistics[idx] = {
                    'correct': self.correct_predictions,
                    'total': self.total_predictions,
                    'success_rate': self.success_rate
                }
                
                # Обновляем статистику по этому состоянию
                if current_state:  # Проверяем, что состояние определено
                    self.state_statistics[current_state]['total'] += 1
                    if is_correct:
                        self.state_statistics[current_state]['correct'] += 1
                
                # Сохраняем результат с полной информацией
                result = {
                    'index': idx,
                    'price': prices[idx],
                    'prediction': prediction,
                    'actual': actual_outcome,
                    'is_correct': is_correct,
                    'confidence': pred_result['confidence'],
                    'success_rate': self.success_rate,
                    'correct_total': f"{self.correct_predictions}-{self.total_predictions}",
                    'state': current_state,
                    'quantile_predictions': pred_result.get('quantile_predictions', {})
                }
                
            #     # Проверка сходимости (заменяем старую логику плато)
            #     if detect_plateau and self.total_predictions >= min_predictions and self.success_rate >= min_success_rate:
            #         window = plateau_window
            #         threshold = 0.001  # Порог изменения успешности (0.1%)
                    
            #         if len(success_rate_history) >= window:
            #             recent_rates = success_rate_history[-window:]
            #             rate_changes = np.abs(np.diff(recent_rates))
            #             avg_change = np.mean(rate_changes)
                        
            #             if hasattr(self.config, 'debug_interval') and self.config.debug_interval > 0 and idx % self.config.debug_interval == 0:
            #                 print(f"Convergence check at index {idx}: avg_change={avg_change:.6f}, threshold={threshold}")
                        
            #             if avg_change < threshold:
            #                 if verbose:
            #                     print(f"\nСходимость достигнута на индексе {idx}, успешность: {self.success_rate * 100:.2f}%")
            #                     print(f"Среднее изменение успешности за последние {window} точек: {avg_change:.6f}")
            #                     print("Останавливаем обработку данных.")
            #                 break
            # else:
            #     # Если предсказание "не знаю"
            #     result = {
            #         'index': idx,
            #         'price': prices[idx],
            #         'prediction': 0,
            #         'confidence': pred_result.get('confidence', 0.0),
            #         'success_rate': self.success_rate if self.total_predictions > 0 else 0,
            #         'state': current_state
            #     }
            
            # results.append(result)

            # Проверка сходимости (супер новая логика плато)
            if detect_plateau and self.total_predictions >= min_predictions and self.success_rate >= min_success_rate:
                window = plateau_window
                # threshold = 0.001  # Убираем порог, так как проверяем точное равенство 0
                
                if len(success_rate_history) >= window:
                    recent_rates = success_rate_history[-window:]  # Берем последние plateau_window значений
                    rate_changes = np.abs(np.diff(recent_rates))  # Вычисляем абсолютные изменения
                    
                    # if hasattr(self.config, 'debug_interval') and self.config.debug_interval > 0 and idx % self.config.debug_interval == 0:
                    #     print(f"Convergence check at index {idx}: rate_changes={rate_changes}")
                    
                    # Проверяем, что все изменения равны 0
                    if np.all(rate_changes == 0):
                        if verbose:
                            print(f"\nСходимость достигнута на индексе {idx}, успешность: {self.success_rate * 100:.2f}%")
                            print(f"Все изменения успешности за последние {window} точек равны 0")
                            print("Останавливаем обработку данных.")
                        break
            else:
                # Если предсказание "не знаю"
                result = {
                    'index': idx,
                    'price': prices[idx],
                    'prediction': 0,
                    'confidence': pred_result.get('confidence', 0.0),
                    'success_rate': self.success_rate if self.total_predictions > 0 else 0,
                    'state': current_state
                }
                
            results.append(result)

            # Обучаем модель квантильной регрессии на основе этого результата
            if idx % self.config.model_update_interval == 0 and idx > min_idx + self.config.prediction_depth:
                self._update_quantile_models(prices, results)
            
            # Обновляем прогресс-бар
            pbar.update(1)
            if self.total_predictions > 0:
                pbar.set_postfix({
                    'Predictions': self.total_predictions,
                    'Success Rate': f"{self.success_rate*100:.2f}%"
                })
        
        # Финализация
        pbar.close()
        if verbose:
            print("\nВалидация завершена:")
            print(f"- Всего предсказаний: {self.total_predictions}")
            print(f"- Правильных предсказаний: {self.correct_predictions}")
            print(f"- Успешность: {self.success_rate * 100:.2f}%")
        
        # Сохраняем ссылку на массив цен для использования в generate_report
        self.prices = prices
        
        return results

    def get_state_statistics(self):
        """
        Возвращает статистику по состояниям
        
        Возвращает:
        pandas.DataFrame: статистика по состояниям
        """
        data = []
        for state, stats in self.state_statistics.items():
            success_rate = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
            data.append({
                'state': str(state),
                'total': stats['total'],
                'correct': stats['correct'],
                'success_rate': success_rate
            })
        
        # Проверка на пустой список
        if not data:
            # Возвращаем пустой DataFrame с нужными столбцами
            return pd.DataFrame(columns=['state', 'total', 'correct', 'success_rate'])
        
        df = pd.DataFrame(data)
        return df.sort_values('total', ascending=False)
    
    def visualize_results(self, prices, results, save_path=None):
        """
        Визуализирует результаты предсказаний, включая информацию от квантильной регрессии
        
        Параметры:
        prices (numpy.array): массив цен
        results (list): результаты предсказаний
        save_path (str): путь для сохранения графиков
        """
        # Создаем графики
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1]})
        
        # График цен
        ax1.plot(prices, color='blue', alpha=0.7, label='Цена')
        
        # Выделяем обучающий участок
        ax1.axvspan(0, self.config.window_size, color='lightgray', alpha=0.3, label='Начальное окно')
        
        # Отмечаем предсказания
        correct_up_indices = []
        correct_up_prices = []
        correct_down_indices = []
        correct_down_prices = []
        
        wrong_up_indices = []
        wrong_up_prices = []
        wrong_down_indices = []
        wrong_down_prices = []
        
        for r in results:
            idx = r['index']
            price = r['price']
            
            if 'is_correct' in r:
                if r['prediction'] == 1:  # Up
                    if r['is_correct']:
                        correct_up_indices.append(idx)
                        correct_up_prices.append(price)
                    else:
                        wrong_up_indices.append(idx)
                        wrong_up_prices.append(price)
                elif r['prediction'] == 2:  # Down
                    if r['is_correct']:
                        correct_down_indices.append(idx)
                        correct_down_prices.append(price)
                    else:
                        wrong_down_indices.append(idx)
                        wrong_down_prices.append(price)
        
        # Отмечаем предсказания на графике
        if correct_up_indices:
            ax1.scatter(correct_up_indices, correct_up_prices, color='green', marker='^', s=50, alpha=0.7, 
                        label='Верно (Рост)')
        if correct_down_indices:
            ax1.scatter(correct_down_indices, correct_down_prices, color='green', marker='v', s=50, alpha=0.7, 
                        label='Верно (Падение)')
        if wrong_up_indices:
            ax1.scatter(wrong_up_indices, wrong_up_prices, color='red', marker='^', s=50, alpha=0.7, 
                        label='Неверно (Рост)')
        if wrong_down_indices:
            ax1.scatter(wrong_down_indices, wrong_down_prices, color='red', marker='v', s=50, alpha=0.7, 
                        label='Неверно (Падение)')
        
        ax1.set_title('Цена и предсказания')
        ax1.set_ylabel('Цена')
        ax1.grid(alpha=0.3)
        ax1.legend()
        
        # График успешности предсказаний
        success_indices = []
        success_rates = []
        
        for idx, stats in sorted(self.point_statistics.items()):
            success_indices.append(idx)
            success_rates.append(stats['success_rate'] * 100)
        
        if success_indices:
            ax2.plot(success_indices, success_rates, 'g-', linewidth=2)
            ax2.axhline(y=50, color='r', linestyle='--', alpha=0.7)
            ax2.set_title('Динамика успешности предсказаний')
            ax2.set_xlabel('Индекс')
            ax2.set_ylabel('Успешность (%)')
            ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Сохраняем график, если указан путь
        if save_path:
            plt.savefig(save_path)
            print(f"График сохранен в {save_path}")
        
        plt.show()
        
        # График распределения состояний и их успешности
        if self.state_statistics:
            # Берем топ-10 самых распространенных состояний
            top_states = sorted(
                [(state, stats['total']) for state, stats in self.state_statistics.items()],
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            if top_states:  # Проверяем, что есть хотя бы одно состояние
                state_labels = [str(state) for state, _ in top_states]
                state_counts = [count for _, count in top_states]
                state_success_rates = [
                    self.state_statistics[state]['correct'] / self.state_statistics[state]['total'] * 100
                    if self.state_statistics[state]['total'] > 0 else 0
                    for state, _ in top_states
                ]
                
                # Создаем график для топ-10 состояний
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
                
                # График количества состояний
                ax1.bar(state_labels, state_counts, color='blue', alpha=0.7)
                ax1.set_title('Частота встречаемости топ-10 состояний')
                ax1.set_ylabel('Количество')
                ax1.set_xticklabels(state_labels, rotation=45)
                ax1.grid(alpha=0.3, axis='y')
                
                # График успешности по состояниям
                ax2.bar(state_labels, state_success_rates, color='green', alpha=0.7)
                ax2.axhline(y=50, color='r', linestyle='--', alpha=0.7)
                ax2.set_title('Успешность предсказаний по состояниям')
                ax2.set_ylabel('Успешность (%)')
                ax2.set_xticklabels(state_labels, rotation=45)
                ax2.grid(alpha=0.3, axis='y')
                
                plt.tight_layout()
                
                # Сохраняем график состояний, если указан путь
                if save_path:
                    states_path = save_path.replace('.png', '_states.png')
                    plt.savefig(states_path)
                    print(f"График состояний сохранен в {states_path}")
                
                plt.show()
        
        # Дополнительно создаем график с информацией от квантильной регрессии
        quantile_results = [r for r in results if 'quantile_predictions' in r]
        
        if quantile_results:
            # Создаем новый график
            plt.figure(figsize=(15, 8))
            
            # График цен
            plt.plot(prices, color='blue', alpha=0.5, label='Цена')
            
            # Собираем данные для отображения квантилей
            indices = []
            medians = []
            lower_bounds = []
            upper_bounds = []
            
            for r in quantile_results:
                if not r['quantile_predictions']:
                    continue
                    
                idx = r['index']
                indices.append(idx)
                
                # Получаем предсказания квантилей
                q_preds = r['quantile_predictions']
                
                # Находим медиану и границы
                median_q = min(q_preds.keys(), key=lambda q: abs(q - 0.5))
                lower_q = min(q_preds.keys(), key=lambda q: abs(q - self.config.lower_quantile))
                upper_q = min(q_preds.keys(), key=lambda q: abs(q - self.config.upper_quantile))
                
                # Преобразуем из процентного изменения в абсолютное значение цены
                price = prices[idx]
                medians.append(price * (1 + q_preds[median_q]))
                lower_bounds.append(price * (1 + q_preds[lower_q]))
                upper_bounds.append(price * (1 + q_preds[upper_q]))
            
            # Отображаем медианные предсказания
            plt.scatter(indices, medians, color='green', s=30, alpha=0.7, label='Медианное предсказание')
            
            # Отображаем интервалы предсказаний
            for i in range(len(indices)):
                plt.plot([indices[i], indices[i]], [lower_bounds[i], upper_bounds[i]], 
                        color='green', alpha=0.3)
                
            plt.title('Предсказания с квантильными интервалами')
            plt.xlabel('Индекс')
            plt.ylabel('Цена')
            plt.grid(alpha=0.3)
            plt.legend()
            
            # Сохраняем график, если указан путь
            if save_path:
                quantile_path = save_path.replace('.png', '_quantiles.png')
                plt.savefig(quantile_path)
                print(f"График квантилей сохранен в {quantile_path}")
            
            plt.show()
    
    def generate_report(self, results, save_path=None, prices=None):
        """
        Генерирует отчет о результатах предсказаний, включая информацию о квантильной регрессии
        
        Параметры:
        results (list): результаты предсказаний
        save_path (str): путь для сохранения отчета
        prices (numpy.array, optional): массив цен, используемый для анализа фактических изменений
        
        Возвращает:
        str: текст отчета
        """
        # Общая статистика
        total_predictions = self.total_predictions
        correct_predictions = self.correct_predictions
        success_rate = self.success_rate * 100
        
        # Распределение предсказаний
        up_count = sum(1 for r in results if r.get('prediction') == 1)
        down_count = sum(1 for r in results if r.get('prediction') == 2)
        neutral_count = sum(1 for r in results if r.get('prediction') == 0)
        
        # Успешность по типам предсказаний
        up_correct = sum(1 for r in results if r.get('prediction') == 1 and r.get('is_correct', False))
        down_correct = sum(1 for r in results if r.get('prediction') == 2 and r.get('is_correct', False))
        
        up_success_rate = up_correct / up_count * 100 if up_count > 0 else 0
        down_success_rate = down_correct / down_count * 100 if down_count > 0 else 0
        
        # Топ состояния
        state_stats = self.get_state_statistics()
        top_states = state_stats.head(10)
        
        # Формируем отчет
        report = f"""
# Отчет о работе предиктора

## Конфигурация
window_size={self.config.window_size}, prediction_depth={self.config.prediction_depth}, confidence_threshold={self.config.confidence_threshold}

## Общая статистика
- Всего предсказаний: {total_predictions}
- Правильных предсказаний: {correct_predictions}
- Успешность: {success_rate:.2f}%

## Распределение предсказаний
- Рост: {up_count} ({up_count/len(results)*100:.2f}%)
- Падение: {down_count} ({down_count/len(results)*100:.2f}%)
- Не знаю: {neutral_count} ({neutral_count/len(results)*100:.2f}%)

## Успешность по типам предсказаний
- Успешность предсказаний роста: {up_correct}/{up_count} ({up_success_rate:.2f}%)
- Успешность предсказаний падения: {down_correct}/{down_count} ({down_success_rate:.2f}%)

## Топ-10 состояний по частоте
{top_states.to_markdown(index=False) if not top_states.empty else "Нет данных о состояниях"}

## Покрытие предсказаний
- Общее покрытие: {(up_count + down_count) / len(results) * 100:.2f}%
"""
        
        # Добавляем информацию о квантильной регрессии, если доступны цены
        if prices is not None:
            quantile_results = [r for r in results if 'quantile_predictions' in r]
            
            if quantile_results:
                # Анализируем результаты квантильных предсказаний
                
                # 1. Точность квантильных предсказаний
                interval_coverage_90 = 0
                interval_coverage_50 = 0
                valid_quantile_count = 0
                
                for r in quantile_results:
                    idx = r['index']
                    if idx + self.config.prediction_depth >= len(prices):
                        continue
                    
                    valid_quantile_count += 1
                    
                    # Фактическое изменение
                    actual_change = prices[idx + self.config.prediction_depth] / prices[idx] - 1
                    
                    # Предсказанные квантили
                    q_preds = r['quantile_predictions']
                    
                    # Проверяем наличие нужных квантилей
                    lower_q = min(q_preds.keys(), key=lambda q: abs(q - self.config.lower_quantile))
                    upper_q = min(q_preds.keys(), key=lambda q: abs(q - self.config.upper_quantile))
                    
                    mid_lower_q = min(q_preds.keys(), key=lambda q: abs(q - self.config.mid_lower_quantile))
                    mid_upper_q = min(q_preds.keys(), key=lambda q: abs(q - self.config.mid_upper_quantile))
                    
                    # Проверяем попадание в интервалы
                    if q_preds[lower_q] <= actual_change <= q_preds[upper_q]:
                        interval_coverage_90 += 1
                    
                    if q_preds[mid_lower_q] <= actual_change <= q_preds[mid_upper_q]:
                        interval_coverage_50 += 1
                
                # Вычисляем проценты попаданий
                interval_90_coverage = interval_coverage_90 / valid_quantile_count * 100 if valid_quantile_count > 0 else 0
                interval_50_coverage = interval_coverage_50 / valid_quantile_count * 100 if valid_quantile_count > 0 else 0
                
                # 2. Средние предсказанные изменения для разных квантилей
                quantile_means = {}
                for q in self.config.quantiles:
                    valid_predictions = [r['quantile_predictions'][q] for r in quantile_results 
                                    if q in r['quantile_predictions']]
                    if valid_predictions:
                        quantile_means[f'q{int(q*100)}'] = np.mean(valid_predictions) * 100
                
                # Формируем дополнительный отчет
                quantile_report = f"""
## Статистика квантильной регрессии

### Общая информация
- Количество предсказаний с квантильной регрессией: {len(quantile_results)}
- Процент попаданий фактического значения в интервал [{self.config.lower_quantile*100:.0f}%, {self.config.upper_quantile*100:.0f}%]: {interval_90_coverage:.2f}%
- Процент попаданий фактического значения в интервал [{self.config.mid_lower_quantile*100:.0f}%, {self.config.mid_upper_quantile*100:.0f}%]: {interval_50_coverage:.2f}%

### Средние предсказанные изменения
"""
                # Добавляем средние значения по квантилям
                for q_name, q_value in quantile_means.items():
                    quantile_report += f"- Средний квантиль {q_name}: {q_value:.2f}%\n"
                
                quantile_report += f"""
### Квантильные модели
- Количество обученных моделей (по состояниям): {len(self.quantile_models)}
"""   
                # Объединяем отчеты
                report += quantile_report
        
        # Сохраняем отчет, если указан путь
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"Отчет сохранен в {save_path}")
        
        return report