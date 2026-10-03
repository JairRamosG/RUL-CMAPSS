"""Pasos del pipeline de datos: carga cruda y feature engineering temporal."""
import logging

import pandas as pd

from src.data.loader import load_cmapss
from src.data.preprocessing import remove_constant_sensors, compute_piecewise_rul
from src.features.engineering import compute_rolling_stats, compute_trends

logger = logging.getLogger(__name__)


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
