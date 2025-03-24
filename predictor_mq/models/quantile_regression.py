"""
Модуль с реализацией квантильной регрессии.
"""

import numpy as np
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler


class QuantileRegressionModel:
    """
    Модель для предсказания квантилей будущего изменения цены.
    """
    
    def __init__(self, quantiles=(0.1, 0.5, 0.9), alpha=0.1):
        """
        Инициализация модели квантильной регрессии
        
        Параметры:
        quantiles (tuple): квантили для предсказания
        alpha (float): параметр регуляризации для модели
        """
        self.quantiles = quantiles
        self.alpha = alpha
        self.models = {}  # Словарь с моделями для разных квантилей
        self.scaler = StandardScaler()  # Нормализация признаков
        self.is_fitted = False  # Флаг, указывающий, обучена ли модель
    
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
        if len(X) < 5:
            self.is_fitted = False
            return self
        
        # Нормализуем признаки
        X_scaled = self.scaler.fit_transform(X)
        
        # Обучаем модель для каждого квантиля
        for q in self.quantiles:
            model = QuantileRegressor(quantile=q, alpha=self.alpha, solver='highs')
            model.fit(X_scaled, y)
            self.models[q] = model
        
        self.is_fitted = True
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
        
        # Преобразуем вектор признаков в 2D массив
        X_reshaped = X.reshape(1, -1) if len(X.shape) == 1 else X
        
        # Нормализуем признаки
        X_scaled = self.scaler.transform(X_reshaped)
        
        # Делаем предсказания для каждого квантиля
        predictions = {}
        for q, model in self.models.items():
            predictions[q] = model.predict(X_scaled)[0]
        
        return predictions

class EnhancedQuantileRegressionModel(QuantileRegressionModel):
    """
    Улучшенная модель квантильной регрессии с поддержкой обогащенных признаков
    и инкрементного обучения.
    """
    
    def __init__(self, quantiles=(0.1, 0.5, 0.9), alpha=0.1, max_samples=10000):
        super().__init__(quantiles, alpha)
        self.max_samples = max_samples  # максимальное количество сэмплов для хранения
        self.stored_X = None  # для инкрементного обучения
        self.stored_y = None  # для инкрементного обучения
    
    def incremental_fit(self, X, y):
        """
        Инкрементное обучение модели без полной переподготовки
        
        Параметры:
        X (numpy.array): новые признаки
        y (numpy.array): новые целевые значения
        
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
            if len(self.stored_y) > self.max_samples:
                # Удаляем самые старые записи
                self.stored_X = self.stored_X[-self.max_samples:]
                self.stored_y = self.stored_y[-self.max_samples:]
        
        # Переобучаем модель на всех сохраненных данных
        return self.fit(self.stored_X, self.stored_y)
    
    def fit(self, X, y):
        """
        Обучает модель с расширенной поддержкой для разных размерностей признаков
        
        Параметры:
        X (numpy.array): признаки (каждая строка - вектор признаков для одного наблюдения)
        y (numpy.array): целевые значения (процентное изменение цены)
        
        Возвращает:
        self: обученная модель
        """
        # Проверяем, что данных достаточно для обучения
        if len(X) < 5:
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
                model = QuantileRegressor(quantile=q, alpha=self.alpha, solver='highs')
                model.fit(X_scaled, y)
                self.models[q] = model
            except Exception as e:
                print(f"Ошибка при обучении модели для квантиля {q}: {e}")
                continue
        
        self.is_fitted = len(self.models) > 0
        return self
    
    def predict_single(self, X):
        """
        Делает предсказание для одного наблюдения с поддержкой разных размерностей
        
        Параметры:
        X (numpy.array): вектор признаков
        
        Возвращает:
        dict: предсказания для разных квантилей
        """
        if not self.is_fitted:
            return None
        
        # Обрабатываем разные размерности входных данных
        if len(X.shape) == 1:
            X_reshaped = X.reshape(1, -1)
        else:
            X_reshaped = X
        
        # Проверяем размерность признаков
        if X_reshaped.shape[1] != self.scaler.n_features_in_:
            # Если количество признаков не совпадает, возвращаем None
            return None
        
        # Нормализуем признаки
        X_scaled = self.scaler.transform(X_reshaped)
        
        # Делаем предсказания для каждого квантиля
        predictions = {}
        for q, model in self.models.items():
            predictions[q] = model.predict(X_scaled)[0]
        
        return predictions    
    