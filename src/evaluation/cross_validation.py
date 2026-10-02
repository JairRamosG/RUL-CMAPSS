"""
Motor de validación cruzada agrupada (GroupKFold) y evaluación sobre test set.

Garantiza el diseño experimental pareado (blocking) perfilando tiempo de cómputo,
memoria RAM pico y latencia de inferencia por motor.
"""
import logging
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import MinMaxScaler

from src.features.selection import create_feature_selector, get_selected_feature_names
from src.features.windowing import scale_and_window_fold
from src.models.base import BaseModel
from src.models.factory import build_model
from src.models.pytorch_wrapper import PyTorchModel
from src.evaluation.metrics import rmse, mae, nasa_score, profile_resource_usage

logger = logging.getLogger(__name__)

def prepare_cv_feature_selection(
        train_enriched_df: pd.DataFrame,
        feature_cols: list[str],
        config: dict,
        n_folds: int = 10
) -> tuple [dict[int, list[str]], dict[str, int]]:
    """
    Aprende la selección de características dentro de cada fold cuidando la fuga de datos

    Returns:
        fold_feature_map: Siccionario {fold_index: lista_columnas_seleccionadas}
        feature_stability: DIccionario {columna: frecuencia_de_seleccion_en_folds}
    """
    fs_cfg = config.get("feature_selection", {})
    if not fs_cfg.get("enabled", False):
        logger.info("Selección de características DESACTIVADA: usando las 102 columnas.")
        return {f: list(feature_cols) for f in range(1, n_folds + 1)}, {c: n_folds for c in feature_cols}

    logger.info(f"Precomputando selección de características ({fs_cfg.get('method', 'hybrid')}) para {n_folds} folds...")
    gkf = GroupKFold(n_splits=n_folds)
    groups = train_enriched_df["unit_number"].values

    fold_feature_map = {}
    feature_counts: dict[str, int] = {col: 0 for col in feature_cols}

    for fold_idx, (train_idx, _) in enumerate(gkf.split(train_enriched_df, groups=groups), start=1):
        fold_train_df = train_enriched_df.iloc[train_idx]

        # Escalado local del fold para el selector
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(fold_train_df[feature_cols])
        y_train = fold_train_df["rul"].values

        # Ajuste del selector exclusivo en train_fold
        selector = create_feature_selector(fs_cfg)
        selector.fit(X_train_scaled, y_train)

        selected_cols = get_selected_feature_names(selector, feature_cols)
        fold_feature_map[fold_idx] = selected_cols

        for col in selected_cols:
            feature_counts[col] += 1

        logger.info(f"  -> Fold {fold_idx:02d}: {len(selected_cols)} features seleccionadas.")

    # Mostrar top 5 más estables para la consola/tesis
    top_stable = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    logger.info(f"Top 5 características más estables: {top_stable}")

    return fold_feature_map, feature_counts

def evaluate_model_cv(
    model_name: str,
    model_params: dict,
    train_enriched_df: pd.DataFrame,
    fold_feature_map: dict[int, list[str]],
    config: dict,
    dry_run: bool = False,
) -> dict:
    """Ejecuta la validación cruzada agrupada (GroupKFold) perfilando tiempo, RAM y métricas.

    Garantiza el diseño pareado (blocking) utilizando las características
    preseleccionadas de cada fold para todos los modelos por igual.

    Args:
        model_name: Identificador del modelo (ej. 'random_forest', 'lstm').
        model_params: Hiperparámetros base del modelo.
        train_enriched_df: DataFrame con características extraídas.
        fold_feature_map: Mapeo {fold_idx: lista_features_seleccionadas}.
        config: Diccionario con la configuración del experimento.
        dry_run: Si es True, reduce a 2 folds para pruebas rápidas.

    Returns:
        dict con métricas agregadas, resultados por fold y la última instancia
        entrenada (last_model/last_scaler/last_features). Estos últimos quedan
        sólo para trazabilidad: el test set se evalúa con el pipeline global
        ajustado con el 100% del train (ver prepare_global_datasets).
    """
    n_folds = 2 if dry_run else config.get("evaluation", {}).get("cv_folds", 10)
    w_size = config.get("data", {}).get("window_size", 30)

    gkf = GroupKFold(n_splits=n_folds)
    groups = train_enriched_df["unit_number"].values

    fold_metrics = []
    last_trained_model = None
    last_scaler = None
    last_features = None

    logger.info(f">>> Iniciando el CV de {model_name.upper()} con {n_folds} folds")

    for fold_indx, (train_idx, val_idx) in enumerate(gkf.split(train_enriched_df, groups=groups), start=1):
        fold_train_df = train_enriched_df.iloc[train_idx]
        fold_val_df = train_enriched_df.iloc[val_idx]
        n_val_engines = fold_val_df["unit_number"].nunique()

        # 1. Obtener las características seleccionadas para este fold
        current_features = fold_feature_map[fold_indx]
        input_shape = (w_size, len(current_features))

        # 2. Escalamiento y ventaneo sin data leakage
        X_train, y_train, X_val, y_val, fold_scaler = scale_and_window_fold(
            fold_train_df,
            fold_val_df,
            current_features,
            window_size=w_size,
        )

        # 3. Instanciar el modelo con las dimensiones reducidas
        model = build_model(model_name, model_params, input_shape)

        # Acelerar el dry-run si es PyTorch
        fit_kwargs = {}
        if isinstance(model, PyTorchModel) and dry_run:
            fit_kwargs = {"epochs": 2}

        # 4. Entrenamiento con perfilado de recursos
        with profile_resource_usage() as train_profiler:
            model.fit(X_train, y_train, **fit_kwargs)

        train_time = train_profiler.elapsed_time
        train_ram = train_profiler.peak_memory_mb

        # 5. Inferencia con perfilado de latencia
        with profile_resource_usage() as inf_profiler:
            y_pred = model.predict(X_val)

        inf_time = inf_profiler.elapsed_time
        latency_ms_per_engine = (inf_time * 1000.0) / max(n_val_engines, 1)

        # 6. Cálculo de métricas
        f_rmse = rmse(y_val, y_pred)
        f_mae = mae(y_val, y_pred)
        f_nasa = nasa_score(y_val, y_pred)

        fold_metrics.append({
            "fold": fold_indx,
            "rmse": f_rmse,
            "mae": f_mae,
            "nasa_score": f_nasa,
            "train_time_sec": train_time,
            "latency_ms_engine": latency_ms_per_engine,
            "peak_ram_mb": train_ram,
            "n_features": len(current_features),
        })

        last_trained_model = model
        last_scaler = fold_scaler
        last_features = current_features

        logger.info(
            "Fold %d/%d (%d features) - RMSE: %.2f | MAE: %.2f | NASA: %.2f | Train: %.2fs | RAM: %.1fMB",
            fold_indx, n_folds, len(current_features), f_rmse, f_mae, f_nasa, train_time, train_ram
        )

    # 7. Agregación estadística de los folds
    rmse_arr = np.array([f["rmse"] for f in fold_metrics])
    mae_arr = np.array([f["mae"] for f in fold_metrics])
    nasa_arr = np.array([f["nasa_score"] for f in fold_metrics])
    train_time_arr = np.array([f["train_time_sec"] for f in fold_metrics])
    latency_arr = np.array([f["latency_ms_engine"] for f in fold_metrics])
    ram_arr = np.array([f["peak_ram_mb"] for f in fold_metrics])

    summary = {
        "cv_rmse_mean": float(np.mean(rmse_arr)),
        "cv_rmse_std": float(np.std(rmse_arr)),
        "cv_mae_mean": float(np.mean(mae_arr)),
        "cv_mae_std": float(np.std(mae_arr)),
        "cv_nasa_mean": float(np.mean(nasa_arr)),
        "cv_nasa_std": float(np.std(nasa_arr)),
        "train_time_mean": float(np.mean(train_time_arr)),
        "latency_ms_mean": float(np.mean(latency_arr)),
        "peak_ram_mean": float(np.mean(ram_arr)),
        "n_features_selected": len(last_features),
    }

    logger.info(
        "=== Resumen CV %s: RMSE: %.2f ± %.2f | MAE: %.2f | NASA: %.2f (%d features) ===",
        model_name.upper(), summary["cv_rmse_mean"], summary["cv_rmse_std"],
        summary["cv_mae_mean"], summary["cv_nasa_mean"], summary["n_features_selected"]
    )

    return {
        "model_name": model_name,
        "fold_metrics": fold_metrics,
        "cv_rmse_scores": rmse_arr,
        "summary": summary,
        "last_model": last_trained_model,
        "last_scaler": last_scaler,
        "last_features": last_features,
    }

def evaluate_on_test_set(
        model: BaseModel,
        X_test: np.ndarray,
        y_test: np.ndarray
) -> dict:
    """
    Evalúa un modelo entrenado contra las etiquetas reales del Teset Set oficial.

    Args:
        model: INstancia entrenada que hereda del BaseModel
        X_test: Tensor 3D (N, W, F) con la última ventana de cada motor
        y_test: Vector 1D (N, ) con el RUL real final de la prueba

    Returns:
        dict con las medidas de deseméño de test: 'test_rmse', 'test_mae', 'test_nasa_score' y latencia
    """
    with profile_resource_usage() as prof:
        y_pred = model.predict(X_test)

    inf_time_sec = prof.elapsed_time
    latency_ms_per_engine = (inf_time_sec * 1000.0) / max(len(y_test), 1)

    t_rmse = float(rmse(y_test, y_pred))
    t_mae = float(mae(y_test, y_pred))
    t_nasa = float(nasa_score(y_test, y_pred))

    logger.info(f">>> TEST SET OFICIAL: RMSE {t_rmse:.2f} | MAE {t_mae:.2f} | NASA {t_nasa:.2f} | Latencia {latency_ms_per_engine:.2f} ms")
    return {
        "test_rmse": t_rmse,
        "test_mae": t_mae,
        "test_nasa_score": t_nasa,
        "test_latency_ms_engine": latency_ms_per_engine,
        "y_pred_test":  y_pred
    }


