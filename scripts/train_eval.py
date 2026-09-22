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

"""

import argparse
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

from src.models.base import BaseModel
from src.models.RFModel import RFModel
from src.models.XGBoostModel import XGBoostModel
from src.models.LightGBMModel import LightGBMModel
from src.models.SVRModel import SVRModel
from src.models.MLPModel import MLPModel
from src.models.CNN1DModel import CNN1DModel
from src.models.LSTMModel import LSTMModel

from sklearn.model_selection import GroupKFold
from src.evaluation.metrics import rmse, mae, nasa_score, profile_resource_usage
from src.models.pytorch_wrapper import PyTorchModel

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

def scale_and_window_fold(
    train_fold_df: pd.DataFrame,
    val_fold_df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Escala las características y estructura en ventanas temporales 3D por fold.

    PROTOCOLO ANTI-LEAKAGE:
        El MinMaxScaler se ajusta (fit) EXCLUSIVAMENTE con los datos de entrenamiento
        del fold actual (train_fold_df). La partición de validación (val_fold_df)
        se transforma usando únicamente la media y escala aprendidas de entrenamiento.

    Args:
        train_fold_df: DataFrame con los motores de entrenamiento del fold.
        val_fold_df: DataFrame con los motores de validación del fold.
        feature_cols: Lista de columnas correspondientes a características numéricas.
        window_size: Tamaño de la ventana deslizante (ciclos de tiempo).

    Returns:
        tuple (X_train, y_train, X_val, y_val):
            - X_train: Tensor 3D (N_train, W, F) float32
            - y_train: Vector 1D (N_train,) float32
            - X_val: Tensor 3D (N_val, W, F) float32
            - y_val: Vector 1D (N_val,) float32
    """
    train_scaled = train_fold_df.copy()
    val_scaled = val_fold_df.copy()

    # Ajuste anti-leakage
    scaler = MinMaxScaler()
    train_scaled[feature_cols] = scaler.fit_transform(train_fold_df[feature_cols])
    val_scaled[feature_cols] = scaler.transform(val_fold_df[feature_cols])

    # Ventanas temporales tridimensionales (N, W, F)
    cols_to_keep = ["unit_number", "time", "rul"] + feature_cols
    X_train, y_train = create_windows(train_scaled[cols_to_keep], window_size=window_size, pad_strategy="edge")
    X_val, y_val = create_windows(val_scaled[cols_to_keep], window_size=window_size, pad_strategy="edge")

    return X_train, y_train, X_val, y_val, scaler

def build_model(
        model_name = str,
        model_params = dict,
        input_shape = tuple[int, int],
) -> BaseModel:
    """
    Instancia dinámicamente los modelos de ML Clásico y DL.
    Conecta hiperparámetros del YAML y la forma de entrada temporal con el constructor
    específico para cada algorítmo de las categorías.

    Args: 
        model_name: Identificador del modelo
        model_params: Diccionario de hiperparámetros de la configuración
        input_shape: TUpla (window_size, num_features) para el tensor 3D
    
    Returns:
        Instancia configurada que hereda del BaseModel (fit, predict, get_params)

    Raises:
        ValueError: Si el nombre del modelo no se reconoce
        NotImplementedError: Para la memoria asociativa
    """

    window_size, num_features = input_shape
    input_dim_flattened = window_size * num_features
    params = model_params.copy() if model_params else {}

    # Modelos basados en ML Clásico (Arboles y Kernel)
    if model_name in ("random_forest", "rf"):
        return RFModel(**params)

    elif model_name == "xgboost":
        return XGBoostModel(**params)

    elif model_name == "lightgbm":
        return LightGBMModel(**params)

    elif model_name == "svr":
        return SVRModel(**params)

    # Modelos basados en DL
    elif model_name == "mlp":
        return MLPModel(input_dim= input_dim_flattened, **params)

    elif model_name == "cnn1d":
        return CNN1DModel(num_features=num_features, window_size=window_size, **params)

    elif model_name == "lstm":
        return LSTMModel(num_features=num_features, **params)

    # Memorias asociativas
    elif model_name == "asociative_memory":
        raise NotImplementedError("Hay que programar la memoria asociativa")

    else:
        raise ValueError(f"Modelo no soportado: {model_name}")

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
        dict con métricas agregadas, resultados por fold y última instancia entrenada.
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

def prepare_test_data(
    test_enriched_df: pd.DataFrame,
    scaler: MinMaxScaler,
    feature_cols: list[str],
    window_size: int = 30,
) -> np.ndarray:

    """
    Prepara el tensor de 3D del test extrayendo únicamente la última ventana de cada motor

    Aplica el escalador ajustado previamente con las estadísticas de los motores de entrenamiento
    y maneja motores con pocos ciclos aplicando un padding inicial idéntico al entrenamiento.

    Args: 
        test_enriched_df: Dataframe de pruebas con características extraidas
        scaler: MinMax scaler ya ajustado con los datos de entrenamiento
        feature_cols: Listacon los nombresde las características
        window_size: Longitud de la secuencia temporal

    Returns:
        np.ndarray: Tensor 3D (N_motores_test, W, F) con la última ventana de cada motor
    """

    test_scaled = test_enriched_df.copy()
    test_scaled[feature_cols] = scaler.transform(test_enriched_df[feature_cols])

    grouped = test_scaled.groupby('unit_number', sort = False)
    last_windows = []

    for unit_id, group in grouped:
        values = group[feature_cols].values
        T = len(values)

        if T >= window_size:
            # Se toman solamente los ultimos W ciclos observados
            window = values[-window_size:]
        else:
            # Si el motor tiene menos de W ciclos, se usa un padding al inicio repitiendo el ciclo 1
            pad_len = window_size - T
            pad_block = np.tile(values[0:1], (pad_len, 1))
            window = np.concatenate([pad_block, values], axis = 0)

        last_windows.append(window)

    X_test_last = np.array(last_windows, dtype = np.float32)
    logger.info(f"Tensor de Test Set preparado: Shape {X_test_last.shape} (última ventana por motor)")
    return X_test_last

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
        config: dict
        ) -> None:
    """
    Registra los resultados de un modelo individual en el MLflow.

    Args:
        model_name: Nombre del modelo
        cv_res: Diccionario con resultados de la validación cruzada
        test_res: Diccionario con los resultados del test set
        config: Archivo de configuración del experimento
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
            "random_seed": config.get("experiment", {}).get("random_seed", {})
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
            "test_latency_sec_engine": test_res.get("test_latency_sec_engine", 0.0)
        })
    logger.info(f"Resultados del modelo {model_name.upper()} registrados en MLflow")

def log_omnibus_comparission(
        cv_results_all: dict,
        test_results_all: dict,
        config: dict
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
                higher_is_better=False, 
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
    

def main() -> None:
    """
    Función principal para ensamblar el pipeline
    """
    # Establecer las configuraciones del pipeline
    args = parse_args()
    logger.info(f"Iniciando el experimento con argumentos: {vars(args)}")

    config = load_config(args.config)
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

        # Test set con el MISMO escalador y features del último fold
        X_test_last = prepare_test_data(
            test_enriched, cv_res["last_scaler"], cv_res["last_features"], w_size
        )
        test_res = evaluate_on_test_set(
            model=cv_res["last_model"],
            X_test=X_test_last,
            y_test=y_test_official,
        )
        test_results_all[m_name] = test_res

        # Registrar Run en MLflow
        log_model_run_to_MLflow(
            model_name=m_name,
            cv_res=cv_res,
            test_res=test_res,
            config=config,
        )

    # Inferencia estadística multimodelo
    log_omnibus_comparission(cv_results_all, test_results_all, config)


if __name__ == "__main__":
    main()
























