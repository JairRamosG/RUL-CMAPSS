"""
Módulo de Selección de Características para RUL-CMAPSS.

Implementa un enfoque híbrido en dos etapas:
1. Filtro univariado (Mutual Information) para descartar ruido.
2. Selector embebido multivariado (Random Forest) para capturar interacciones.
Compatible con Scikit-learn Pipeline y protocolo anti-leakage.
"""

from typing import Any
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.validation import check_array, check_is_fitted

class MutualInfoSelector(BaseEstimator, TransformerMixin):
    """
    Selector de características basado en Información mútua univariada

    Calcula la dependencia no lineal entre cada característica y el target RUL
    usando mutual_info_regression y conserva el percentil superior configurado

    Args:
        percentile: Porcentaje de características a conservar
        random_state: Semilla aleatoria de reproducibilidad
    """

    def __init__(self, percentile: int = 60, random_state: int = 42) -> None:
        self.percentile = percentile
        self.random_state = random_state

    def fit(self, X: Any, y: Any = None) -> "MutualInfoSelector":
        """
        Aprende las puntuaciónes de información mutua sobre los X_train

        Args:
            X: Matriz o DataFrame de características
            y: Vector del target RUL porque es selección supervisada

        Raises:
            ValueError: Si y es un None
        """

        if y is None:
            raise ValueError(f"Se requiere un target 'y' porque es selección supervisada (target de entrenamiento solamente)")

        X_arr = check_array(X, accept_sparse = False)
        y_arr = np.asanyarray(y).ravel()

        self.n_features_in_ = X_arr.shape[1]

        # Calcular la información mútua para cada columna
        self.scores_ = mutual_info_regression(
            X_arr,
            y_arr,
            random_state = self.random_state
        ) 

        # Determinar cuántas características conservar según el percentil
        k = max(1, int(np.round(self.n_features_in_ * ((self.percentile / 100.0)))))
        k = min(k, self.n_features_in_)

        # Obtener los índices de los mayores puntajes
        top_indices = np.argsort(self.scores_)[-k:]
        self.selected_indices_ = np.sort(top_indices)

        # Crear una máscara booleana para identificarlos
        self.support_ = np.zeros(self.n_features_in_, dtype = bool)
        self.support_[self.selected_indices_] = True

    def transform(self, X:Any) -> np.ndarray:
        """
        Filtra y retorna únicamente las columnas seleccionadas en la mascara del fit

        Args:
            X: Matriz de características
        Returns:
            np.ndarray con las características seleccionadas solamente
        """

        check_is_fitted(self, ["support_", "selected_indices_"])
        X_arr = check_array(X, accept_sparse = False)

        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"Dimensiones imcompatibles: X tiene {X_arr.shape[1]} features",
                            f"pero el selector fue ajustado con {self.n_features_in_}")
        return X_arr[:, self.support_]

def get_support(self, indices: bool = False) -> np.ndarray:
        """Retorna la máscara booleana o los índices de las columnas elegidas."""
        check_is_fitted(self, ["support_", "selected_indices_"])
        if indices:
            return self.selected_indices_
        return self.support_

def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
    """Proyecta los nombres de las características que sobrevivieron al filtro."""
    check_is_fitted(self, ["support_", "selected_indices_"])
    if input_features is None:
        return np.array([f"x{i}" for i in self.selected_indices_])
    input_features = np.asarray(input_features)
    return input_features[self.support_]























