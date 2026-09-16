"""
Pipeline de entrenamiento, evaluación y tracking de RUL para el C-MAPSS.

Parte 1. Configuración, CLI y Reproducibilidad.

Uso:
    python scripts/train_eval.py --help
    python scripts/train_eval.py --help
    python scripts/train_eval.py --config configs/config_FD001.yaml
    python scripts/train_eval.py --config configs/config_FD001.yaml --models random_forest mlp
    python scripts/train_eval.py --config configs/config_FD001.yaml --dry-run

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
    X_train, y_train = create_windows(train_scaled, window_size=window_size, pad_strategy="edge")
    X_val, y_val = create_windows(val_scaled, window_size=window_size, pad_strategy="edge")

    return X_train, y_train, X_val, y_val


# Función principal
def main() -> None:
    """
    Función principal para ensamblar el pipeline
    """
    # Estabelcer las configuraciónes del pipeline
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
        logger.info(f"DRY-RUN reducido a {train_df['unit_number'].nunique()}")

    # Extraer las características
    train_enriched = extract_features(train_df, config)

    # Identificar características numéricas que no me sirven
    exclude = {"unit_number", "time", "rul"}
    feature_cols = [col for col in train_enriched.columns if col not in exclude]
    logger.info(f"Total de características para modelado: {len(feature_cols)}")

    # Prueba rápida de la función anti-leakage con una partición simple (ej. motores 1..10 val, resto train)
    val_units = [1, 2] if args.dry_run else list(range(1, 11))
    f_val = train_enriched[train_enriched["unit_number"].isin(val_units)]
    f_train = train_enriched[~train_enriched["unit_number"].isin(val_units)]
    
    w_size = config.get("data", {}).get("window_size", 30)
    X_tr, y_tr, X_va, y_va = scale_and_window_fold(f_train, f_val, feature_cols, window_size=w_size)
    logger.info("Shapes generados -> X_train: %s, y_train: %s | X_val: %s, y_val: %s", 
                X_tr.shape, y_tr.shape, X_va.shape, y_va.shape)


if __name__ == "__main__":
    main()





























