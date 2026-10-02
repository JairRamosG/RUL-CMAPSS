"""
Pipeline de entrenamiento, evaluación y tracking de RUL para el C-MAPSS.

Parte 1. Configuración, CLI y Reproducibilidad.

Uso:
    python scripts/train_eval.py --help
    python scripts/train_eval.py --help
    python scripts/train_eval.py --config configs/config_FD001.yaml
    python scripts/train_eval.py --config configs/config_FD001.yaml --models random_forest mlp
    python scripts/train_eval.py --config configs/config_FD001.yaml --dry-run
    python scripts/train_eval.py --models random_forest mlp lstm --dry-run

    train_eval consume ese único archivo con todos los modelos
    uv run python scripts/train_eval.py --config configs/config_FD001.yaml --tuned-params tuned/config_FD001_tuned.json

"""

import argparse
import json
import logging
from pathlib import Path
import random
import sys
from typing import Optional

import numpy as np
import torch
import yaml

from sklearn.preprocessing import MinMaxScaler
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_cmapss
from src.data.preprocessing import remove_constant_sensors, compute_piecewise_rul
from src.features.engineering import compute_rolling_stats, compute_trends, create_windows
from src.features.selection import create_feature_selector, get_selected_feature_names

from sklearn.model_selection import GroupKFold
from sklearn.base import BaseEstimator
from src.evaluation.metrics import rmse, mae, nasa_score, profile_resource_usage
from src.models.pytorch_wrapper import PyTorchModel

# -----------------------------------------
from src.models.base import BaseModel
from src.models.factory import build_model

from src.evaluation.cross_validation import(
    prepare_cv_feature_selection,
    evaluate_model_cv,
    evaluate_on_test_set
    )

from src.features.windowing import (scale_and_window_fold,
                                    scale_and_select_features,
                                    prepare_test_data,
                                    prepare_global_datasets)   

import mlflow
from src.evaluation.statistical_tests import compare_multiple_models

# Configuracción de los loggings
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_eval")

# Parsing para la línea de comandos
def parse_args() -> argparse.Namespace:
    """
    Parseo para la línea de comandos

    Returns:
        argparse.Namespace con los argumentos
    """
    parser = argparse.ArgumentParser(
        description= "Pipeline de entrenamiento, evaluación y tracking con MLflow para el C-Mapss",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type = str,
        default= "configs/config_FD001.yaml",
        help= "Ruta del archivo YAML de configuración del experimento"
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Lista de modelos específicos a evaluar"
            "Si no se especifican, se evalúan todos los modelos"
    )

    parser.add_argument(
        "--dry-run",
        action = "store_true",
        help="Modo de pruebas rápido con datos/folds reducidos para ahorro computacional"
    )

    parser.add_argument(
        "--tuned-params",
        type = str,
        default = None,
        help="Ruta al JSON de hiperparámetros afinados exportado por scripts/optimize.py "
            "(p.ej. tuned/config_FD001_tuned.json). Si se omite se usan los defaults del YAML"
    )

    return parser.parse_args()

# Cargar el archivo de configuración
def load_config(config_path:str | Path) -> dict:
    """
    Carga y valida los archivos de configuración YAML de los experimentos.

    Args:
        config_path: Ruta del archivo YAML

    Returns:
        dict con la información del experimento
    
    Raises:
        FileNotFoundError: Si el archivo no existe en el sistema de archivos
        ValueError: Le faltan secciónes al archivo de configuración
    """

    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de configuración en : {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Validación de las secciónes del archivo de configuración YAML
    required_sections = ["subset", "data", "models", "evaluation", "experiment"]
    missing = [sec for sec in required_sections if sec not in config]
    if missing:
        raise ValueError(f"El archivo {path.name} no es válido para los experimentos. Le falta: {missing}")
    
    logger.info(f"Configuración correcta cargada desde: {path.name}")
    return config

# Definir la semilla aleatoria
def set_seed(seed: int = 42) -> None:
    """
    Fija la semilla para ejecutar todos los experimentos con reproducibilidad

    Args:
        seed: Valor entero     
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Semilla determinística establecida en: {seed}")

def prepare_raw_data(config:dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carga los datos crudos, filtra los sensores constantes y etiqueta el valor de RUL

    Args:
        config: Diccionario del archivo de configuración del experimento

    Returns:
        tuple: (train_df, test_df, rul_test_df) preprocesados a nivel tabular base
    """
    subset = config.get("subset", "FD001")
    data_dir = config.get("data", {}).get("data_dir", "datos")
    rul_max = config.get("data", {}).get("rul_max", 125)
    sensors_to_remove = config.get("sensors", {}).get("remove", [])

    logger.info(f"Cargando el subset de: {subset} - {data_dir}")
    train_df, test_df, rul_test_df = load_cmapss(subset = subset, data_dir = data_dir)

    # FIltrar sensores de varianza 0 que se vieron en el EDA
    if sensors_to_remove:
        logger.info(f"Removiendo {len(sensors_to_remove)} sensores constantes: {sensors_to_remove}")
        train_df = remove_constant_sensors(train_df, sensors_to_remove)
        test_df = remove_constant_sensors(test_df, sensors_to_remove)

    # Etiquetar RUL en entrenamiento
    logger.info(f"Calculando la etiqueta de rul con el rul_max = {rul_max}")
    train_df = compute_piecewise_rul(train_df, rul_max = rul_max)

    return train_df, test_df, rul_test_df

def extract_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Aplica ingeniería de características temporales (rolling stats y trends)

    Args: 
        df: DataFrame ordenado por (unit_number, time)
        config: Diccionario de configuración del experimento

    Returns:
        DataFrame con todas las nuevas columnas
    """
    fe_cfg = config.get("feature_engineering", {})
    df_features = df.copy()

    # Calcular todas las rolling stats (time delay embedding)
    rolling_cfg = fe_cfg.get("rolling_stats", {})
    if rolling_cfg.get("enabled", False):
        w_size = rolling_cfg.get("window_size", 30)
        stat_types = rolling_cfg.get("stat_types", ["mean", "std", "min", "max"])
        logger.info(f"Calculando las rolling stats W:{w_size} - stats: {stat_types}")
        df_features = compute_rolling_stats(df_features, window_size = w_size, stat_types = stat_types)

    # Calcular tendencias y diferencias finitas
    trends_cfg = fe_cfg.get("trends", {})
    if trends_cfg.get("enabled", False):
        delta_steps = trends_cfg.get("delta_steps", [1])
        logger.info(f"Calculando las tendencias de cada sensor con deltas: {delta_steps}")
        df_features = compute_trends(df_features, delta_steps = delta_steps, base_features_only = True)
    return df_features

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

def apply_tuned_params(config: dict, tuned_path: str | Path) -> dict:
    """
    Fusiona los hiperparámetros afinados exportados por scripts/optimize.py sobre
    config["models"][i]["params"]: los defaults del YAML se conservan y en los
    keys en conflicto prevalece el valor afinado.

    Los modelos sin entrada en el JSON conservan los defaults del YAML.
    """
    tuned = json.loads(Path(tuned_path).read_text(encoding="utf-8"))
    for model in config.get("models", []):
        name = model.get("name")
        if name not in tuned:
            logger.info(f"'{name}' sin afinación en {tuned_path}: usando defaults del YAML")
            continue
        if not isinstance(tuned[name], dict):
            raise ValueError(
                f"tuned-params: la entrada de '{name}' debe ser un dict de parámetros, "
                f"es {type(tuned[name]).__name__}"
            )
        logger.info(f"Hiperparámetros afinados fusionados sobre defaults del YAML en {name}: {tuned[name]}")
        model["params"] = {**model.get("params", {}), **tuned[name]}
    return config

def main() -> None:
    """
    Función principal para ensamblar el pipeline
    """
    # Establecer las configuraciones del pipeline
    args = parse_args()
    logger.info(f"Iniciando el experimento con argumentos: {vars(args)}")

    config = load_config(args.config)
    if args.tuned_params:
        config = apply_tuned_params(config, args.tuned_params)
    seed = config.get("experiment", {}).get("random_seed", 42)
    set_seed(seed)

    # Filtrado opcional de modelos
    configured_models = [m["name"] for m in config.get("models", [])]
    if args.models:
        selected_models = [m for m in configured_models if m in args.models]
        logger.info(f"Modelos seleccionádos: {selected_models}")
    else:
        selected_models = configured_models
        logger.info(f"Se evaluarán todos los modelos")
    if args.dry_run:
        logger.warning(f"Entrenamiento en modo de pruebas rapido")

    # Carga y preparación de los datos
    train_df, test_df, rul_test_df = prepare_raw_data(config)
    if args.dry_run:
        train_df = train_df[train_df["unit_number"] <= 20].copy()
        test_df = test_df[test_df["unit_number"] <= 20].copy()
        rul_test_df = rul_test_df.iloc[:20].copy()
        logger.info(f"DRY-RUN reducido a {train_df['unit_number'].nunique()}")

    # Extraer las características
    train_enriched = extract_features(train_df, config)
    test_enriched = extract_features(test_df, config)

    # Identificar características numéricas que no me sirven
    exclude = {"unit_number", "time", "rul"}
    feature_cols = [col for col in train_enriched.columns if col not in exclude]
    logger.info(f"Total de características para modelado: {len(feature_cols)}")

    w_size = config.get("data", {}).get("window_size", 30)
    n_folds = 2 if args.dry_run else config.get("evaluation", {}).get("cv_folds", 10)

    # Selección de características por fold (una sola vez para todos los modelos)
    fold_feature_map, feature_stability = prepare_cv_feature_selection(
        train_enriched, feature_cols, config, n_folds=n_folds
    )

    # Pipeline global: scaler + selector ajustados con el 100% del train.
    # Compartido por TODOS los modelos (diseño pareado) y por el test set.
    X_train_full, y_train_full, X_test_final, global_features = prepare_global_datasets(
        train_enriched, test_enriched, feature_cols, config
    )

    rul_max = config.get("data", {}).get("rul_max", 125)
    y_test_official = np.minimum(rul_test_df["rul"].values, rul_max).astype(np.float32)

    init_mlflow(config)

    #----------------------------------------------------------------------------------------
    # Bucle de evaluación por modelo
    models_dict = {m["name"]: m.get("params", {}) for m in config.get("models", [])}
    cv_results_all = {}
    test_results_all = {}

    for m_name in selected_models:
        if m_name == "associative_memory":
            logger.warning(
                "Modelo 'associative_memory' omitido: pendiente de implementación (Issue #6). "
                "El experimento corre con los modelos restantes."
            )
            continue

        m_params = models_dict.get(m_name, {})

        # Validación cruzada
        cv_res = evaluate_model_cv(
            model_name=m_name,
            model_params=m_params,
            train_enriched_df=train_enriched,
            fold_feature_map=fold_feature_map,
            config=config,
            dry_run=args.dry_run,
        )
        cv_results_all[m_name] = cv_res

        # Reentrenamiento final desde cero con los MISMOS params sobre el 100% del train.
        # Los modelos de CV (last_model/last_scaler/last_features) quedan sólo para trazabilidad.
        final_model = build_model(m_name, m_params, (w_size, len(global_features)))
        fit_kwargs = {}
        if isinstance(final_model, PyTorchModel) and args.dry_run:
            fit_kwargs = {"epochs": 2}
        final_model.fit(X_train_full, y_train_full, **fit_kwargs)

        test_res = evaluate_on_test_set(
            model=final_model,
            X_test=X_test_final,
            y_test=y_test_official,
        )
        test_results_all[m_name] = test_res

        # Registrar Run en MLflow
        log_model_run_to_MLflow(
            model_name=m_name,
            cv_res=cv_res,
            test_res=test_res,
            config=config,
            n_features_selected=len(global_features),
        )

    # Inferencia estadística multimodelo
    log_omnibus_comparission(
        cv_results_all,
        test_results_all,
        config,
        fold_feature_map=fold_feature_map,
        feature_counts=feature_stability,
    )


if __name__ == "__main__":
    main()
























