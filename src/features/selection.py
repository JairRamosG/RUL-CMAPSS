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
from sklearn.ensemble import RandomForestRegressor
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

        return self

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


class RFFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Selector de características embebido basado en el random forest

    Entrena un bosque de árboles de desición  para poder capturar dependencias no lineales
    e interacciónes multivariadas entre los sensores, seleccionando las 'k' características
    con mayor importancia para el RUL.

    Args:
        n_features_to_select: Cantidad exacta de características finales a conservar
        n_estimators: Número de arboles para le bosque
        max_depth: Profundidad máxima de los arboles
        random_state: Semilla para trazabilidad
        n_jobs: Número de núcleos del CPU
    """

    def __init__(
            self,
            n_features_to_select: int = 25,
            n_estimators: int = 120,
            max_depth: int | None = 15,
            random_state: int = 42,
            n_jobs: int = -1
            ) -> None:
        self.n_features_to_select = n_features_to_select
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X: Any, y: Any = None) -> "RFFeatureSelector":
        """
        Aprende la importancia de cada característica mientras entrena el Random FOrest

        Args:
            X: Matriz de características de entrenamiento, proviene del MutualInfo
            y: Vector con los valores originales del RUL (Supervisado)

        Raises:
            ValueError: Si Y es None porque es supervisado
        """

        if y is None:
            raise ValueError(f"se requiere de un vector de 'y' para el RUL")
        X_arr = check_array(X, accept_sparse= False)
        y_arr = np.asarray(y).ravel()

        self.n_features_in_ = X_arr.shape[1]

        # Instanciar y entrenar el modelo base
        rf = RandomForestRegressor(
            n_estimators = self.n_estimators,
            max_depth = self.max_depth,
            random_state = self.random_state,
            n_jobs = self.n_jobs
        )
        rf.fit(X_arr, y_arr)

        # Guardar las importancias
        self.feature_importances_ = rf.feature_importances_

        # Determinar las k mejores características
        k = min(self.n_features_to_select, self.n_features_in_)
        k = max(1, k)

        # Obtener los índices de mayor importancia
        top_indices = np.argsort(self.feature_importances_)[-k:]
        self.selected_indices_ = np.sort(top_indices)

        # Crear una mascara booleanda de soporte
        self.support_ = np.zeros(self.n_features_in_, dtype=bool)
        self.support_[self.selected_indices_] = True
        return self

    def transform(self, X: Any) -> np.ndarray:
        """
        Filtra y reotna únicamente las k columnas seleccionadas por el Random FOrest

        Args:
            X: Matriz de características
        
        Returns:
            np.ndarray con las k columnas seleccionadas
        """

        check_is_fitted(self, ["support_", "selected_indices_"])
        X_arr = check_array(X, accept_sparse = False)

        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"DImensiones incopatibles: X tiene {X_arr.shape[1]} features",
                            f"pero el selector fue ajustado con {self.n_features_in_}")

        return X_arr[:, self.support_]

    def get_support(self, indices: bool = False) -> np.ndarray:
        """
        Retorna la mascara booleana o los índices de las columnas elegidas
        """
        check_is_fitted(self,  ["support_", "selected_indices_"])
        if indices:
            return self. selected_indices_
        return self.support_

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """
        Proyecta los nombres de las características que sobrevivieron este segundo filtro
        """
        check_is_fitted(self, ["support_", "selected_indices_"])
        if input_features is None:
            return np.array([f"x{i}" for i in self.selected_indices_])
        input_features = np.asarray(input_features)
        return input_features[self.support_]
        

















