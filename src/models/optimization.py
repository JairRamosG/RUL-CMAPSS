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

VALID_OBJECTIVES = ("rmse", "nasa_score")
VALID_DIRECTIONS = ("minimize", "maximize")

def validate_optimization_config(opt_cfg: dict) -> dict:
    """
    Valida que el bloque de configuración en el archivo YAML tenga un formato valido (Fail Early, Fail Fast)
    """
    if "models" not in opt_cfg:
        raise ValueError("Falta el bloque models en el archivo de configuración")
    if not opt_cfg["models"]:
        raise ValueError("optimization.models debe tener al menos un modelo")

    if int(opt_cfg.get("n_trials") or 0) < 20:
        raise ValueError(f"optimization.n_trials debe ser >= 20, se tiene: {opt_cfg.get('n_trials')}")

    if opt_cfg.get("objective") not in VALID_OBJECTIVES:
        raise ValueError(f"optimization.objective debe ser uno de los que están en {VALID_OBJECTIVES} y es {opt_cfg.get('objective')}")

    if opt_cfg.get("direction") not in VALID_DIRECTIONS:
        raise ValueError(f"optimization.direction debe ser uno de los que están en {VALID_DIRECTIONS} y es {opt_cfg.get('direction')}")

    if int(opt_cfg.get("cv_folds") or 0) < 2:
        raise ValueError(f"optimization.cv_folds debe ser >= 2 y se tiene {opt_cfg.get('cv_folds')}")

    return opt_cfg


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
    """
    Optimiza los hiperparámetros con Optuna TPE sobre un GroupKFold con grupos por motor donde el scaler y el selector solo ven train
    """

    if model_factory is None:
        raise ValueError(f"model_factoy con función lambda es obligatorio para optimizar los hiperparámetros")

    opt_cfg = config.get("optimization", {})
    n_trials = config.get("n_trials", 20)
    cv_folds = config.get("cv_folds", 2)
    objective = opt_cfg.get("objective", "rmse")
    direction = opt_cfg.get("direction", "minimize")
    seed = int(opt_cfg.get("seed", 42))

    if dry_run:
        n_trials = min(3, n_trials)
        cv_folds = min(2, cv_folds)

    metric_fn = {"rmse": rmse, "nasa_score": nasa_score}.get(objective)
    if metric_fn is None:
        raise ValueError(f"optimization.objective '{objective}' no soportado")

    groups = dataset["unit_number"].values
    gkf = GroupKFold(n_splits = cv_folds)
    splits = list(gkf.split(dataset, groups = groups))

    fs_cfg = config.get("feature_selection") or {}

    def evaluate(params: dict) -> float:
        """
        Obtiene la métrica promedio del objetivo con CV por motor, es la simulación del pipeline
        """
        fold_scores = []
        for train_idx, val_idx in splits:
            train_df = dataset.iloc[train_idx]
            val_df = dataset.iloc[val_idx]
            X_train, y_train = train_df[feature_cols], train_df["rul"]
            X_val, y_val = val_df[feature_cols], val_df["rul"]

            scaler = MinMaxScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)

            if fs_cfg.get("enabled"):
                selector = create_feature_selector(fs_cfg)
                X_train_s = selector.fit_transform(X_train_s, y_train)
                X_val_s = selector.transform(X_val_s)

            model = model_factory({**base_params, **params})
            model.fit(X_train_s, y_train)
            y_pred = np.asarray(model.predict(X_val_s), dtype=float)
            fold_scores.append(float(metric_fn(np.asarray(y_val, dtype=float), y_pred)))
        return float(np.mean(fold_scores))

    sampler = optuna.samplers.TPESampler(seed = seed)
    study = optuna.create_study(direction = direction, sampler = sampler, storage = storage)
    study.optimize(
        lambda trial : evaluate(suggest_from_space(trial, search_space)),
        n_trials = n_trials
    )

    baseline_value = evaluate(base_params)
    best_value = float(study.best_value)
    best_params = dict(study.best_params)

    baseline_beaten = (best_value < baseline_value if direction == "minimize" else best_value > baseline_value)
    if not baseline_beaten:
        logger.warning("%s: tuned (%.4f) no supera al baseline (%.4f) en '%s'; search space a revisar.",
            model_name, best_value, baseline_value, objective,
        )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    return{
        "best_params": best_params,
        "best_value": best_value,
        "baseline_value": baseline_value,
        "objective": objective,
        "n_trials_completed" : len(completed)
    }




def export_best_params(results: dict, out_path) -> Path:
    """
    Exporta los mejores hiperparametros a JSON para usar en train_eval --tuned-parms
    Crea los directorios por si faltan y sobreescribe los existentes
    """

    out = Path(out_path)
    out.parent.mkdir(parents = True, exist_ok = True)
    out.write_text(json.dumps(results, indent = 2), encoding = 'utf-8')
    return out

