# Baseline models (RF, XGBoost, SVR, MLP, CNN, LSTM)
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

# Obliga a que los que hereden de BaseModel, siempre tengan las funciónes de fit, predict y get_params -------
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

# Plantilla para los modelo de ML clásico-------------------------------------------------------
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
        return self.model.get_params()
    

# Plantilla para los modelo de DL ---------------------------------------------------------------
import copy
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def _get_device() -> torch.device:
    """Detecta GPU disponible, fallback a CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class PyTorchModel(BaseModel):
    """Base para modelos deep learning con PyTorch.

    Proporciona device management, early stopping y train loop.
    Los modelos hijos implementan la arquitectura específica.
    """

    def __init__(self, module: nn.Module, params: Optional[Dict[str, Any]] = None, lr: float = 1e-3):
        """Inicializa la plantilla adaptadora para PyTorch.

        Args:
            module: Módulo PyTorch con la arquitectura del modelo.
            params: Hiperparámetros del modelo.
            lr: Tasa de aprendizaje inicial.
        """
        self.module = module
        self.params = params if params is not None else {}
        self.lr = lr
        self.device = _get_device()
        self.module.to(self.device)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 64,
        patience: int = 10,
    ) -> Dict[str, Any]:
        """Entrena el modelo PyTorch con optimización de memoria VRAM y Early Stopping."""

        # Mantiene los tensores principales en CPU para no saturar la memoria VRAM
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Preparación de datos de validación
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.Adam(self.module.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        best_weights = None
        epochs_no_improve = 0
        early_stopped = False

        for epoch in range(epochs):
            # Fase de Entrenamiento
            self.module.train()
            train_loss = 0.0

            for X_batch, y_batch in loader:
                # Se envían únicamente los lotes activos al dispositivo
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                y_pred = self.module(X_batch).squeeze(-1)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * X_batch.size(0)

            train_loss /= len(loader.dataset)

            # Fase de Validación
            val_loss = None
            if val_loader is not None:
                self.module.eval()
                val_loss_sum = 0.0

                with torch.no_grad():
                    for X_vbatch, y_vbatch in val_loader:
                        X_vbatch = X_vbatch.to(self.device)
                        y_vbatch = y_vbatch.to(self.device)

                        y_vpred = self.module(X_vbatch).squeeze(-1)
                        v_loss = criterion(y_vpred, y_vbatch)
                        val_loss_sum += v_loss.item() * X_vbatch.size(0)

                val_loss = val_loss_sum / len(val_loader.dataset)

                # Control de Early Stopping y respaldo del mejor modelo
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    best_weights = copy.deepcopy(self.module.state_dict())
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        early_stopped = True
                        break

        # Cargar los mejores pesos encontrados si se usó Early Stopping
        if best_weights is not None:
            self.module.load_state_dict(best_weights)

        return {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epochs_trained": epoch + 1,
            "early_stopped": early_stopped,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predice RUL procesando la entrada por lotes para evitar sobrecarga de GPU."""
        self.module.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=256, shuffle=False)

        predictions = []
        with torch.no_grad():
            for (X_batch,) in loader:
                X_batch = X_batch.to(self.device)
                y_pred = self.module(X_batch).squeeze(-1)
                predictions.append(y_pred.cpu().numpy())

        return np.concatenate(predictions, axis=0).reshape(-1)

    def get_params(self) -> Dict[str, Any]:
        """Retorna hiperparámetros del modelo."""
        return self.params.copy()