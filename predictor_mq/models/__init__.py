"""
Модели прогнозирования временных рядов.
"""

from .hybrid_predictor import HybridPredictor
from .standardized_quantile_regression import QuantileRegressionModel

__all__ = ['HybridPredictor', 'QuantileRegressionModel']