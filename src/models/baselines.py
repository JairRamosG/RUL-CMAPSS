# Baseline models (RF, XGBoost, SVR, MLP, CNN, LSTM)
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

class BaseModel(ABC):
    """
    Interfaz abstracta para todos los modelos del RUL

    Los modelos de ML, DL y la Memoria Asociativa implementan esta interfaz para
    poder garantizar compatiblidad en el pipeline con el que se va a evaluar.
    """

    @abstractmethod
    def fit(self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None
            )-> dict:
        """
        Función para el entrenamiento dle modelo

        Args:
            X_train: Features de entrenamiento (N, F) o (N, W, F)
            y_train: Target RUL de entrenamiento (N, )
            X_val: Features de validación para early stopping (puede ser opcional)
            y_val: Target RUL de validación (también puede ser opcional)
        
        Returns:
            dict con keys: train_loss, val_loss, epoch_trained, early_stopped
        """

    @abstractmethod
    def predict(self, X: np.ndarray)-> np.ndarray:
        """
        Función que predice el RUL para cada muestra

        Args:
            X: Features para predecir (N, F) o (N, W, F)
        
        Returns:
            Array con las predicciónes del RUL (N, )
        """

    @abstractmethod
    def get_params(self)-> dict:
        """
        Retorna los hiperparámetros ontenidos del modelo, los voy a ocupar con el MLflow
        """

class SKLearnModel(BaseModel):
    """
    Adaptador para los modelos que son de scikit-learn.

    Maneja el reshape automático de 3D a 2D y delega fit/predict al modelo que lo use.
    """

    def __init__(self, model, params: dict):
        """
        Inicializa el adaptador del modelo

        Args:
            model: Instancia que se use de scikit-learn
            params: Hiperparámetros a provar en el modelo        
        """
        self.model = model
        self.params = params

    def _flatten(self, X: np.ndarray)-> np.ndarray:
        """
        Aplana los datos si es necesario de 3D a 2D

        Args:
            X: Array 2D (N, F) o 3D (N, W, F).
        
        Returns:
            Array de forma 2D (N, F) o (N, W*F)
        """

        if X.ndim == 3:
            return X.reshape(X.shape[0], -1)
        return X
    
    def fit(
        self, 
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None)-> dict:
        """
        Función que entrena el modelo de scikit-learn

        Args:
            X_train: features de entrenamiento 2D (N, F) o 3D (N, W, F).,
            y_train: Target RUL de entrenameinto (N, ),
            X_val: No usado en scikit-learn para compatibilidad
            y_val: No usado en scikit-learn para compatibilidad
        
        Returns:
            dict con train_loss, val_loss, epoch_trained, early_stopped
        """
        X_flat = self._flatten(X_train)
        self.model.fit(X_flat, y_train)

        return {
            "train_loss" : 0.0,
            "val_loss"  : None,
            "epoch_trained" : 1,
            "early_stopped": False
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predice RUL usando el modelo scikit-learn.
        
        Args:
            X: Features a predecir (N, F) o (N, W, F).
            
        Returns:
            Array con predicciones RUL de forma (N,).
        """
        X_flat = self._flatten(X)
        return self.model.predict(X_flat)
    
    def get_params(self) -> dict:
        """Retorna hiperparámetros del modelo.
        
        Returns:
            dict con los hiperparámetros configurados
        """
