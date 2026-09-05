from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class BaseModel(ABC):
    """Interfaz abstracta para todos los modelos de RUL.
    
    Todos los modelos (ML y DL) implementan esta interfaz para
    garantizar compatibilidad con el pipeline de evaluación.
    """
    
    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> dict:
        """Entrena el modelo.
        
        Args:
            X_train: Features de entrenamiento (N, F) o (N, W, F).
            y_train: Target RUL de entrenamiento (N,).
            X_val: Features de validación para early stopping (opcional).
            y_val: Target RUL de validación (opcional).
            
        Returns:
            dict con keys: train_loss, val_loss, epochs_trained, early_stopped
        """
        
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predice RUL para cada muestra.
        
        Args:
            X: Features a predecir (N, F) o (N, W, F).
            
        Returns:
            Array con predicciones RUL de forma (N,).
        """
        
    @abstractmethod
    def get_params(self) -> dict:
        """Retorna hiperparámetros del modelo.
        
        Returns:
            dict con los hiperparámetros configurados.
        """
