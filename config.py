"""
config.py

Конфигурационный файл для настройки гибридного предиктора.
Содержит параметры по умолчанию и предустановленные конфигурации.
"""

import numpy as np

"""

PredictorConfig(window_size=750, prediction_depth=15, min_confidence=0.6, state_length=4, 
                significant_change_pct=0.4%, use_weighted_window=False, weight_decay=0.95, 
                recency_boost=1.5, quantiles=(0.1, 0.5, 0.9), min_samples_for_regression=10, 
                confidence_threshold=0.005, max_coverage=0.05)
"""



class PredictorConfig:
    """
    Конфигурация параметров гибридного предиктора.
    """
    def __init__(
        # Основные параметры
        self,
        window_size=750,               # Размер окна (количество точек для обучения)
        prediction_depth=15,           # Глубина предсказания (на сколько точек вперед)
        
        # Параметры состояний и порогов
        state_length=4,                # Длина последовательности состояний
        significant_change_pct=0.004,  # Порог значимого изменения (в долях)
        default_movement=1,            # Значение движения по умолчанию (1=рост, 2=падение)
        threshold_percentile=75,       # Процентиль для динамического порога
        min_samples_for_threshold=5,   # Минимум образцов для расчета порога
        
        # Параметры квантильной регрессии
        quantiles=(0.1, 0.5, 0.9),      # Квантили для регрессии
        regression_alpha=0.1,           # Параметр регуляризации для регрессии
        regression_solver='highs',      # Решатель для квантильной регрессии
        min_samples_for_regression=10,   # Минимум образцов для обучения регрессии
        
        # Параметры предсказаний
        min_confidence=0.6,             # Минимальная уверенность для предсказания
        confidence_threshold=0.005,      # Порог уверенности для фильтрации предсказаний
        confidence_offset=0.5,          # Смещение для расчета уверенности
        max_coverage=0.05,               # Максимальное покрытие (доля предсказаний)
        lower_quantile=0.1,             # Нижний квантиль для расчета уверенности
        upper_quantile=0.9,             # Верхний квантиль для расчета уверенности
        mid_lower_quantile=0.25,        # Средний нижний квантиль для отчетов
        mid_upper_quantile=0.75,        # Средний верхний квантиль для отчетов
        default_probability=0.5,        # Вероятность по умолчанию
        movement_normalization=2.0,     # Коэффициент нормализации движений
        
        # Параметры извлечения признаков
        basic_feature_count=5,          # Количество базовых признаков
        state_feature_count=3,          # Количество признаков состояния 
        min_window_for_features=3,      # Минимальный размер окна для признаков
        ma_period=20,                   # Период для скользящей средней
        speed_period=5,                 # Период для скорости изменения
        accel_period=10,                # Период для ускорения
        use_extended_features=False,    # Использовать расширенные признаки
        extended_feature_functions=None,# Функции для расширенных признаков
        state_feature_functions=None,   # Функции для признаков состояния
        
        # Параметры обработки и вывода
        debug_interval=1000,            # Интервал для отладочного вывода
        model_update_interval=100,      # Интервал обновления моделей
        
        # Параметры обнаружения плато
        plateau_window=1000,            # Размер окна для обнаружения плато
        min_predictions_for_plateau=50, # Минимум предсказаний для оценки плато
        min_success_rate_for_plateau=0.5# Минимальная успешность для оценки плато
    ):
        # Сохраняем все параметры
        self.window_size = window_size
        self.prediction_depth = prediction_depth
        
        self.state_length = state_length
        self.significant_change_pct = significant_change_pct
        self.default_movement = default_movement
        self.threshold_percentile = threshold_percentile
        self.min_samples_for_threshold = min_samples_for_threshold
        
        self.quantiles = quantiles
        self.regression_alpha = regression_alpha
        self.regression_solver = regression_solver
        self.min_samples_for_regression = min_samples_for_regression
        
        self.min_confidence = min_confidence
        self.confidence_threshold = confidence_threshold
        self.confidence_offset = confidence_offset
        self.max_coverage = max_coverage
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.mid_lower_quantile = mid_lower_quantile
        self.mid_upper_quantile = mid_upper_quantile
        self.default_probability = default_probability
        self.movement_normalization = movement_normalization
        
        self.basic_feature_count = basic_feature_count
        self.state_feature_count = state_feature_count
        self.min_window_for_features = min_window_for_features
        self.ma_period = ma_period
        self.speed_period = speed_period
        self.accel_period = accel_period
        self.use_extended_features = use_extended_features
        self.extended_feature_functions = extended_feature_functions or []
        self.state_feature_functions = state_feature_functions or []
        
        self.debug_interval = debug_interval
        self.model_update_interval = model_update_interval
        
        self.plateau_window = plateau_window
        self.min_predictions_for_plateau = min_predictions_for_plateau
        self.min_success_rate_for_plateau = min_success_rate_for_plateau
    
    def __str__(self):
        """Строковое представление конфигурации"""
        return (
            f"PredictorConfig(\n"
            f"  window_size={self.window_size}, "
            f"prediction_depth={self.prediction_depth}, "
            f"state_length={self.state_length},\n"
            f"  significant_change_pct={self.significant_change_pct*100:.2f}%, "
            f"quantiles={self.quantiles},\n"
            f"  confidence_threshold={self.confidence_threshold}, "
            f"max_coverage={self.max_coverage}\n"
            f")"
        )


# Предустановленные конфигурации для разных сценариев

def create_standard_config():
    """Стандартная сбалансированная конфигурация"""
    return PredictorConfig(
        window_size=750,
        prediction_depth=15,
        state_length=4,
        significant_change_pct=0.004,
        quantiles=(0.1, 0.5, 0.9),
        confidence_threshold=0.0055,
        max_coverage=0.1,
        plateau_window=500
    )

def create_high_precision_config():
    """Конфигурация для высокой точности предсказаний"""
    return PredictorConfig(
        window_size=750,
        prediction_depth=15,
        state_length=4,
        significant_change_pct=0.004,
        quantiles=(0.05, 0.25, 0.5, 0.75, 0.95),  # Расширенный набор квантилей
        confidence_threshold=0.0075,               # Высокий порог уверенности
        max_coverage=0.05                        # Низкое покрытие
    )

def create_high_coverage_config():
    """Конфигурация для максимального покрытия данных"""
    return PredictorConfig(
        window_size=750,
        prediction_depth=15,
        state_length=4,
        significant_change_pct=0.005,           # Меньший порог изменения
        quantiles=(0.1, 0.5, 0.9),
        confidence_threshold=0.0045,              # Низкий порог уверенности
        max_coverage=0.2                        # Высокое покрытие
    )

def create_high_volatility_config():
    """Конфигурация для рынков с высокой волатильностью"""
    return PredictorConfig(
        window_size=750,                       #  окно
        prediction_depth=15,                   # глубина предсказания
        state_length=7,
        significant_change_pct=0.002,           # порог изменения
        quantiles=(0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95),
        # quantiles=(0.1, 0.5, 0.9),
        confidence_threshold=0.0004,
        max_coverage=0.06
    )

def create_low_volatility_config():
    """Конфигурация для рынков с низкой волатильностью"""
    return PredictorConfig(
        window_size=1000,                      # Большее окно
        prediction_depth=20,                   # Большая глубина предсказания
        state_length=4,
        significant_change_pct=0.005,          # Меньший порог изменения
        quantiles=(0.1, 0.5, 0.9),
        confidence_threshold=0.005,
        max_coverage=0.1
    )

def create_optimized_success_rate_config():
    """Конфигурация, оптимизированная для успешности ~58%"""
    return PredictorConfig(
        window_size=750,
        prediction_depth=15,
        state_length=4,
        significant_change_pct=0.004,
        quantiles=(0.1, 0.5, 0.9),
        min_samples_for_regression=3,
        confidence_threshold=0.0058,
        max_coverage=0.1
    )
"""

PredictorConfig(window_size=750, prediction_depth=15, min_confidence=0.6, state_length=4, 
                significant_change_pct=0.4%, use_weighted_window=False, weight_decay=0.95, 
                recency_boost=1.5, quantiles=(0.1, 0.5, 0.9), min_samples_for_regression=10, 
                confidence_threshold=0.005, max_coverage=0.05)
"""

def create_quick_test_config():
    """Конфигурация для быстрого тестирования"""
    return PredictorConfig(
        window_size=500,
        prediction_depth=15,
        state_length=5,
        significant_change_pct=0.04, # не влияет 0.004 или 0.04
        quantiles=(0.05, 0.25, 0.5, 0.75, 0.95),
        min_samples_for_regression=3,
        confidence_threshold=0.003,  # не влияет 0.003 или 0.3
        max_coverage=0.8,
        plateau_window=500
    )

# Словарь с предустановками для удобного доступа
CONFIG_PRESETS = {
    'standard': create_standard_config,
    'high_precision': create_high_precision_config,
    'high_coverage': create_high_coverage_config,
    'high_volatility': create_high_volatility_config,
    'low_volatility': create_low_volatility_config,
    'optimized': create_optimized_success_rate_config,
    'quick_test': create_quick_test_config
}

def get_config(preset_name='standard'):
    """
    Получает конфигурацию по имени предустановки
    
    Параметры:
    preset_name (str): имя предустановки
    
    Возвращает:
    PredictorConfig: объект конфигурации
    """
    if preset_name not in CONFIG_PRESETS:
        raise ValueError(f"Неизвестная предустановка: {preset_name}. "
                        f"Допустимые значения: {', '.join(CONFIG_PRESETS.keys())}")
    
    return CONFIG_PRESETS[preset_name]()