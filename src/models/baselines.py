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