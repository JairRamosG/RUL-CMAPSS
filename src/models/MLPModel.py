from typing import List, Optional, Dict, Any
import torch
import torch.nn as nn
import numpy as np
from src.models.pytorch_wrapper import PyTorchModel


class _MLPModule(nn.Module):
    """Módulo PyTorch con la arquitectura concreta del MLP."""

    def __init__(
        self,
        input_dim: int,
        hidden_units: List[int],
        dropout: float = 0.2,
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim

        for units in hidden_units:
            layers.append(nn.Linear(prev_dim, units))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = units

        # Capa de salida lineal para estimación de RUL (Regresión)
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Aplanado automático si las entradas vienen en 3D: (N, W, F) -> (N, W * F)
        if x.dim() == 3:
            x = x.view(x.size(0), -1)
        return self.network(x)


class MLPModel(PyTorchModel):
    """Wrapper para el Perceptrón Multicapa (MLP) compatible con la interfaz BaseModel."""

    def __init__(
        self,
        input_dim: int,
        hidden_units: Optional[List[int]] = None,
        dropout: float = 0.2,
        lr: float = 1e-3,
    ):
        if hidden_units is None:
            hidden_units = [128, 64]

        params = {
            "input_dim": input_dim,
            "hidden_units": hidden_units,
            "dropout": dropout,
            "lr": lr,
        }

        arch_params = {
            "input_dim": input_dim,
            "hidden_units": hidden_units,
            "dropout": dropout,
        }
        module = _MLPModule(**arch_params)
        super().__init__(module=module, params=params, lr=lr)