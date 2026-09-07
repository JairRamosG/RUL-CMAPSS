from typing import List, Optional, Dict, Any
import torch
import torch.nn as nn
import numpy as np
from src.models.pytorch_wrapper import PyTorchModel


class _LSTMModule(nn.Module):
    """Módulo PyTorch con la arquitectura LSTM para estimación de RUL."""

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Capa recurrente LSTM
        # batch_first=True -> espera entrada (Batch, Window, Features)
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Capa de regularización
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Cabeza de regresión (capas densas finales)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape esperado: (Batch, Window_Size, Num_Features)
        
        # lstm_out shape: (Batch, Window_Size, Hidden_Dim)
        # _ contiene (h_n, c_n) estados ocultos y de celda finales
        lstm_out, _ = self.lstm(x)

        # Tomamos únicamente el último estado oculto de la secuencia temporal: (Batch, Hidden_Dim)
        last_time_step = lstm_out[:, -1, :]

        # Aplicamos dropout y pasamos por el regresor lineal
        x_out = self.dropout(last_time_step)
        return self.regressor(x_out)


class LSTMModel(PyTorchModel):
    """Wrapper para la red LSTM compatible con la interfaz BaseModel."""

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        lr: float = 1e-3,
    ):
        params = {
            "num_features": num_features,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "lr": lr,
        }

        arch_params = {
            "num_features": num_features,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
        }
        module = _LSTMModule(**arch_params)
        super().__init__(module=module, params=params, lr=lr)