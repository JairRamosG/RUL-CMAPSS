"""
Módulo optimizador de Hiperparámetros con Optuna

    - suggest_from_space(trial, search_space)   YAML space -> optuna suggest_* calls
    - validate_optimization_config(opt_cfg)     ValueError on invalid optimization block
    - optimize_hyperparameters(...)             TPE tuning over leakage-free GroupKFold CV
    - export_best_params(results, out_path)     tuned JSON for train_eval --tuned-params
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import MinMaxScaler

from src.evaluation.metrics import nasa_score, rmse
from src.features.selection import create_feature_selector

logger = logging.getLogger(__name__)

VALID_OBJETIVES = ("rmse", "nasa_score")
VALID_DIRECTIONS = ("minimize", "maximize")

def suggest_from_space(trial: Any, search_space: dict) -> dict:
    """
    Mapea los espacios de busqueda del YAML para la llamada de Optuna suggest_*
    """

    params: dict[str, Any] = {}
    for name, spec in search_space.items():
        if "type" not in spec:
            raise ValueError(f"parameter: '{name}' no indica bien su tipo   ")

        ptype = spec["type"]
        if ptype == "categorical":
            choices = spec.get("choices") or []
            if not choices:
                raise ValueError(f"parametro categórico '{choices}' no puede estar vacío")
            params[name] = trial.suggest_categorical(name, tuple(choices))

        elif ptype in ('int', 'float'):
            low, high = spec.get("low"), spec.get("high")
            if low is None or high is None:
                raise ValueError(f"parameter {name}: {ptype} necesita limites inferiores o superiores")
            if low > high:
                raise ValueError(f"el parametro {low} no puede ser mayor que {high}")
            if ptype == "int":
                params[name] = trial.suggest_int(name, low, high, step = spec.get("step", 1))
            elif ptype == "float":
                params[name] = trial.suggest_float(name, low, high, log = bool(spec.get("log", False)))

        else:
            raise ValueError(f"parametro {name} no soportado")
    return params

def validate_optimization_config(opt_cfg: dict) -> dict:
    raise NotImplementedError("unit 2")


def optimize_hyperparameters(
    model_name: str,
    base_params: dict,
    search_space: dict,
    dataset,            # pd.DataFrame
    feature_cols: list[str],
    config: dict,
    model_factory=None,
    dry_run: bool = False,
    storage: str | None = None,
) -> dict:
    raise NotImplementedError("unit 4")


def export_best_params(results: dict, out_path) -> Path:
    raise NotImplementedError("unit 3")