"""
standardized_quantile_regression.py

Стандартизированная реализация квантильной регрессии, объединяющая функциональность
из hybrid_predictor.py и quantile_regression.py.
"""

import numpy as np
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler


class QuantileRegressionModel:
    """
    Унифицированная модель для предсказания квантилей будущего изменения цены.
    Объединяет функциональность из hybrid_predictor.py и quantile_regression.py.
    """
    
    def __init__(self, quantiles=(0.1, 0.5, 0.9), alpha=0.1, min_samples=5, solver='highs'):
        """
        Инициализация модели квантильной регрессии
        
        Параметры:
        quantiles (tuple): квантили для предсказания
        alpha (float): параметр регуляризации для модели
        min_samples (int): минимальное количество образцов для обучения
        solver (str): решатель для квантильной регрессии
        """
        self.quantiles = quantiles
        self.alpha = alpha
        self.min_samples = min_samples
        self.solver = solver
        self.models = {}  # Словарь с моделями для разных квантилей
        self.scaler = StandardScaler()  # Нормализация признаков
        self.is_fitted = False  # Флаг, указывающий, обучена ли модель
        self.stored_X = None  # для инкрементного обучения
        self.stored_y = None  # для инкрементного обучения
    
    def fit(self, X, y):
        """
        Обучает модель на исторических данных
        
        Параметры:
        X (numpy.array): признаки (каждая строка - вектор признаков для одного наблюдения)
        y (numpy.array): целевые значения (процентное изменение цены)
        
        Возвращает:
        self: обученная модель
        """
        # Проверяем, что данных достаточно для обучения
        if len(X) < self.min_samples:
            self.is_fitted = False
            return self
        
        # Обрабатываем случай, когда X - одномерный массив
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        # Нормализуем признаки
        X_scaled = self.scaler.fit_transform(X)
        
        # Обучаем модель для каждого квантиля
        for q in self.quantiles:
            try:
                model = QuantileRegressor(quantile=q, alpha=self.alpha, solver=self.solver)
                model.fit(X_scaled, y)
                self.models[q] = model
            except Exception as e:
                print(f"Ошибка при обучении модели для квантиля {q}: {e}")
                continue
        
        self.is_fitted = len(self.models) > 0
        return self
    
    def predict(self, X):
        """
        Делает предсказание для новых данных
        
        Параметры:
        X (numpy.array): признаки для предсказания
        
        Возвращает:
        dict: предсказания для разных квантилей
        """
        if not self.is_fitted:
            return None
        
        # Обрабатываем случай, когда X - одномерный массив
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        # Нормализуем признаки
        X_scaled = self.scaler.transform(X)
        
        # Делаем предсказания для каждого квантиля
        predictions = {}
        for q, model in self.models.items():
            predictions[q] = model.predict(X_scaled)
        
        return predictions
    
    def predict_single(self, X):
        """
        Делает предсказание для одного наблюдения
        
        Параметры:
        X (numpy.array): вектор признаков
        
        Возвращает:
        dict: предсказания для разных квантилей
        """
        if not self.is_fitted:
            return None
        
        # Проверяем формат входных данных
        X_reshaped = X.reshape(1, -1) if len(X.shape) == 1 else X
        
        # Проверяем размерность признаков
        if hasattr(self.scaler, 'n_features_in_') and X_reshaped.shape[1] != self.scaler.n_features_in_:
            return None
        
        # Нормализуем признаки
        X_scaled = self.scaler.transform(X_reshaped)
        
        # Делаем предсказания для каждого квантиля
        predictions = {}
        for q, model in self.models.items():
            predictions[q] = model.predict(X_scaled)[0]
        
        return predictions
    
    def incremental_fit(self, X, y, max_samples=10000):
        """
        Инкрементное обучение модели без полной переподготовки
        
        Параметры:
        X (numpy.array): новые признаки
        y (numpy.array): новые целевые значения
        max_samples (int): максимальное количество сэмплов для хранения
        
        Возвращает:
        self: обученная модель
        """
        # Если модель еще не обучена, обучаем с нуля
        if not self.is_fitted:
            return self.fit(X, y)
        
        # Если хранилище пусто, инициализируем его
        if self.stored_X is None:
            self.stored_X = X
            self.stored_y = y
        else:
            # Добавляем новые данные
            self.stored_X = np.vstack((self.stored_X, X))
            self.stored_y = np.append(self.stored_y, y)
            
            # Ограничиваем размер хранилища
            if len(self.stored_y) > max_samples:
                # Удаляем самые старые записи
                self.stored_X = self.stored_X[-max_samples:]
                self.stored_y = self.stored_y[-max_samples:]
        
        # Переобучаем модель на всех сохраненных данных
        return self.fit(self.stored_X, self.stored_y)