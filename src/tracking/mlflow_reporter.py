"""
Reportes de experimentos en MLflow: tracking individual por modelo y
comparación estadística omnibus multimodelo.
"""
import logging
from typing import Dict, List, Optional

import mlflow
import numpy as np

from src.evaluation.statistical_tests import compare_multiple_models

logger = logging.getLogger(__name__)


def init_mlflow(config:dict) -> None:
    """
    Inicializa la conexión y el experimento en MLflow

    Args:
        config: DIccionario con la configuración del experimento
    """
    tracking_uri = config.get("experiment", {}).get("mlflow_tracking_uri", "sqlite:///mlflow.db")
    exp_name = config.get("experiment", {}).get("mlflow_experiment_name", "rul_cmapss_F001")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(exp_name)
    logger.info(f"MLflow conectado en: {tracking_uri} (experimento: {exp_name})")

def log_model_run_to_MLflow(
        model_name: str,
        cv_res: dict,
        test_res: dict,
        config: dict,
        n_features_selected: int,
        ) -> None:
    """
    Registra los resultados de un modelo individual en el MLflow.

    Args:
        model_name: Nombre del modelo
        cv_res: Diccionario con resultados de la validación cruzada
        test_res: Diccionario con los resultados del test set
        config: Archivo de configuración del experimento
        n_features_selected: Cantidad de features del tensor final (global).
            Si la selección está deshabilitada es el total de columnas.
    """
    models_dict = {m["name"]: m.get("params", {}) for m in config.get("models", [])}
    m_params = models_dict.get(model_name, {})

    with mlflow.start_run(run_name = model_name):

        # Tags
        mlflow.set_tags({
            "model_name": model_name,
            "subset": config.get('subset', 'FD001'),
            "stage": "baseline" 
        })

        # Parámetros globales y del modelo
        params_to_log = {
            "rul_max": config.get("data", {}).get("rul_max", 125),
            "window_size": config.get("data", {}).get("window_size", 30),
            "cv_folds": config.get("evaluation", {}).get("cv_folds", 10),
            "random_seed": config.get("experiment", {}).get("random_seed", {}),
            "n_features_selected": int(n_features_selected),
        }
        for k, v in m_params.items():
            params_to_log[f"model_{k}"] = str(v)
        mlflow.log_params(params_to_log)

        # Métricas obtenidas del CV y de recursos
        summary = cv_res.get("summary", {})
        mlflow.log_metrics({
            "cv_rmse_mean": summary.get("cv_rmse_mean", 0.0),
            "cv_rmse_std": summary.get("cv_rmse_std", 0.0),
            "cv_mae_mean": summary.get("cv_mae_mean", 0.0),
            "cv_mae_std": summary.get("cv_mae_std", 0.0),
            "cv_nasa_mean": summary.get("cv_nasa_mean", 0.0),
            "train_time_mean_sec": summary.get("train_time_mean", 0.0),
            "peak_ram_mean_mb": summary.get("peak_ram_mean", 0.0),
            "latency_mean_sec_engine": summary.get("latency_ms_mean", 0.0)
        })

        # Métricas sobre el test oficial
        mlflow.log_metrics({
            "test_rmse": test_res.get("test_rmse", 0.0),
            "test_mae": test_res.get("test_mae", 0.0),
            "test_nasa_score": test_res.get("test_nasa_score", 0.0),
            "test_latency_sec_engine": test_res.get("test_latency_ms_engine", 0.0)
        })
    logger.info(f"Resultados del modelo {model_name.upper()} registrados en MLflow")


def log_omnibus_comparission(
        cv_results_all: dict,
        test_results_all: dict,
        config: dict,
        fold_feature_map: dict[int, list[str]] | None = None,
        feature_counts: dict[str, int] | None = None,
) -> None:
    """
    Ejecuta la parte de la inferencia estadística multimodelo (Friedman + Nemenyi) y lo 
    registra en MLflow.

    Itera sobre las métricas declaradas en el YAML (precision, seguridad aeronáutica, tiempo de entrenamiento
    latencia de inferencia y consumo de RAM), registra los artefactos JSON correspondientes en un único RUN global
    en MLflow e imprime una tabla de resúmen en la consola

    Args:
        cv_results_all: Diccionario con los resultados de folds de CV de cada modelo
        test_results_all: Diccionario con los resultados del test set oficial
        config: Archivo de configuración de los experimentos
        fold_feature_map: Mapeo {fold_idx: lista_features_seleccionadas} para
            registrar la estabilidad de la selección de características.
        feature_counts: Frecuencia {feature: nº de folds que la seleccionaron}.
    """
    if len(cv_results_all) < 2:
        logger.info(f"Sólo se evaluó un modelo. Se omite la comparación estadística multimodelo")
        return

    stat_cfg = config.get("evaluation", {}).get("statistical_test", {})
    alpha = stat_cfg.get("alpha", 0.05)
    higher_is_better = stat_cfg.get("higher_is_better", False)
    force_test = stat_cfg.get("force_test", None)

    metrics_to_compare = stat_cfg.get("metrics_to_compare", [
        "rmse",
        "mae",
        "nasa_score",
        "latency_ms_engine",
        "peak_ram_mb",
        "train_time_sec"
    ])
    
    model_names = list(cv_results_all.keys())
    
    # Nombres para usar en la tabla de consola
    metric_labels = {
        "rmse": "Precisión (RMSE)",
        "mae": "Error Absoluto (MAE)",
        "nasa_score": "Seguridad (NASA Score)",
        "latency_ms_engine": "Latencia de Inferencia (ms)",
        "peak_ram_mb": "Memoria RAM Pico (MB)",
        "train_time_sec": "Tiempo de Entrenamiento (s)",
    }

    stat_summaries = []

    # 1. Un solo Run Omnibus en MLflow para todos los análisis
    with mlflow.start_run(run_name="Omnibus_statistical_comparison"):
        mlflow.set_tags({
            "type": "statistical_comparison",
            "subset": config.get("subset", "FD001"),
            "num_models": str(len(model_names)),
        })

        # Estabilidad de la selección de características (Issue #5, criterio 5)
        if fold_feature_map is not None and feature_counts is not None:
            mlflow.log_dict(
                {
                    "feature_counts": feature_counts,
                    "fold_feature_map": {str(k): v for k, v in fold_feature_map.items()},
                },
                "feature_stability.json",
            )

        # 2. Bucle para evaluar cada dimensión por separado 
        for metric_name in metrics_to_compare:
            # Extrae el vector pareado de esa métrica a través de los 10 folds para cada modelo
            scores_list = [
                np.array([f[metric_name] for f in cv_results_all[m]["fold_metrics"]])
                for m in model_names
            ]

            logger.info(f"Ejecutando prueba estadística para: {metric_name}")
            stat_res = compare_multiple_models(
                *scores_list,
                alpha=alpha,
                model_names=model_names,
                higher_is_better=higher_is_better, 
                force_test=force_test,
            )

            # Guarda el artefacto JSON específico en MLflow
            mlflow.log_dict(stat_res.to_dict(), f"statistical_analysis_{metric_name}.json")
            mlflow.log_metrics({f"stat_p_value_{metric_name}": float(stat_res.omnibus_p_value)})

            # Modelo ganador (Ranking 1.0)
            best_model = min(stat_res.rankings.items(), key=lambda x: x[1])[0]

            stat_summaries.append({
                "metric": metric_labels.get(metric_name, metric_name),
                "test_used": stat_res.test_used,
                "p_value": stat_res.omnibus_p_value,
                "significant": "Sí" if stat_res.significant else "No",
                "best_model": best_model,
            })

    # 3. Imprimir la Tabla Resumen Consolidada en Consola
    print("\n" + "=" * 95)
    print(f"{'RESUMEN DE TEST ESTADÍSTICOS CON GRUPOS PAREADOS':^95}")
    print("=" * 95)
    print(f"{'Dimension':<30} {'Prueba Utilizada':<24} {'Omnibus p-val':<16} {'Significante?':<10} {'Modelo #1':<15}")
    print("-" * 95)
    for row in stat_summaries:
        print(f"{row['metric']:<30} {row['test_used']:<24} {row['p_value']:<16.4e} {row['significant']:<10} {row['best_model']:<15}")
    print("=" * 95)

    # 4. Tabla de Métricas por Modelo
    print(f"{'TABLA RESUMEN DE RENDIMIENTO':^95}")
    print("-" * 95)
    print(f"{'Modelo':<18} {'CV RMSE (μ ± σ)':<22} {'Test RMSE':<12} {'RAM Pico':<14} {'Latencia (ms)':<14}")
    print("-" * 95)
    for m in model_names:
        cv_s = cv_results_all[m].get("summary", {})
        cv_str = f"{cv_s.get('cv_rmse_mean', 0.0):.2f} ± {cv_s.get('cv_rmse_std', 0.0):.2f}"
        t_rmse = test_results_all.get(m, {}).get("test_rmse", 0.0)
        ram = f"{cv_s.get('peak_ram_mean', 0.0):.1f} MB"
        lat = f"{cv_s.get('latency_ms_mean', 0.0):.2f} ms"
        print(f"{m:<18} {cv_str:<22} {t_rmse:<12.2f} {ram:<14} {lat:<14}")
    print("=" * 95 + "\n")