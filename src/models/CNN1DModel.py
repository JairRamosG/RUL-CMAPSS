from typing import List, Optional, Dict, Any
import torch
import torch.nn as nn
import numpy as np
from src.models.pytorch_wrapper import PyTorchModel


class _CNN1DModule(nn.Module):
    """Módulo PyTorch con la arquitectura convolucional 1D para RUL."""

    def __init__(
        self,
        num_features: int,
        window_size: int,
        conv_filters: List[int],
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        layers = []
        in_channels = num_features

        for out_channels in conv_filters:
            layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding="same",
                )
            )
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_channels = out_channels

        self.conv_net = nn.Sequential(*layers)
        
        # Reducción de la dimensión temporal a 1
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Regresor final denso
        self.regressor = nn.Sequential(
            nn.Linear(conv_filters[-1], 64),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Si la entrada entra como (N, W, F), la permutamos a (N, F, W) para Conv1d
        if x.dim() == 3 and x.size(1) != self.conv_net[0].in_channels:
            x = x.permute(0, 2, 1)

        x = self.conv_net(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return self.regressor(x)


class CNN1DModel(PyTorchModel):
    """Wrapper para la red CNN-1D compatible con la interfaz BaseModel."""

    def __init__(
        self,
        num_features: int,
        window_size: int,
        conv_filters: Optional[List[int]] = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
        lr: float = 1e-3,
    ):
        if conv_filters is None:
            conv_filters = [32, 64, 128]

        params = {
            "num_features": num_features,
            "window_size": window_size,
            "conv_filters": conv_filters,
            "kernel_size": kernel_size,
            "dropout": dropout,
            "lr": lr,
        }

        arch_params = {
            "num_features": num_features,
            "window_size": window_size,
            "conv_filters": conv_filters,
            "kernel_size": kernel_size,
            "dropout": dropout,
        }
        module = _CNN1DModule(**arch_params)
        super().__init__(module=module, params=params, lr=lr)