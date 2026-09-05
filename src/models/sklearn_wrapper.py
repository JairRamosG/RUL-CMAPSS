from typing import Optional, Dict, Any
import numpy as np
from src.models.base import BaseModel


class SKLearnModel(BaseModel):
    """Adapter para modelos scikit-learn.
    
    Maneja reshape automático de 3D a 2D y delega
    fit/predict al modelo subyacente.
    """
    
    def __init__(self, model, params: Optional[Dict[str, Any]] = None):
        """Inicializa el adapter.
        
        Args:
            model: Instancia de scikit-learn (SVR, RF, etc.)
            params: Hiperparámetros del modelo.
        """
        self.model = model
        self.params = params if params is not None else {}
    
    def _flatten(self, X: np.ndarray) -> np.ndarray:
        """Aplana input 3D a 2D si es necesario.
        
        Args:
            X: Array de forma (N, F) o (N, W, F).
            
        Returns:
            Array de forma (N, F) o (N, W*F).
        """
        if X.ndim == 3:
            return X.reshape(X.shape[0], -1)
        return X
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Entrena el modelo scikit-learn.
        
        Args:
            X_train: Features de entrenamiento (N, F) o (N, W, F).
            y_train: Target RUL de entrenamiento (N,).
            X_val: No utilizado en scikit-learn (para compatibilidad).
            y_val: No utilizado en scikit-learn (para compatibilidad).
            
        Returns:
            dict con train_loss, val_loss, epochs_trained, early_stopped.
        """
        X_flat = self._flatten(X_train)
        self.model.fit(X_flat, y_train)
        
        return {
            "train_loss": 0.0,
            "val_loss": None,
            "epochs_trained": 1,
            "early_stopped": False,
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
    
    def get_params(self) -> Dict[str, Any]:
        """Retorna hiperparámetros del modelo.
        
        Returns:
            dict con los hiperparámetros configurados.
        """
        return self.params.copy()
