"""
Módulo de ventaneo temporal 3D, escalamiento y preparación de tensores.

Implementa el protocolo anti-fuga de datos (Anti-Data-Leakage) tanto
para la validación cruzada por motor (GroupKFold) como para el pipeline global.
"""
import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import MinMaxScaler

from src.features.engineering import create_windows
from src.features.selection import create_feature_selector, get_selected_feature_names

logger = logging.getLogger(__name__)


def scale_and_window_fold(
    train_fold_df: pd.DataFrame,
    val_fold_df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """Scale features and build the 3D sliding windows for a single CV fold.

    ANTI-LEAKAGE PROTOCOL:
        The MinMaxScaler is fitted EXCLUSIVELY on the training partition of the
        current fold (train_fold_df). The validation partition (val_fold_df) is
        transformed using only the mean and scale learned from training.

    Args:
        train_fold_df: DataFrame with the training engines of the fold.
        val_fold_df: DataFrame with the validation engines of the fold.
        feature_cols: List with the names of the numeric feature columns.
        window_size: Sliding window size (time cycles).

    Returns:
        tuple (X_train, y_train, X_val, y_val, scaler):
            - X_train: 3D tensor (N_train, W, F) float32
            - y_train: 1D vector (N_train,) float32
            - X_val: 3D tensor (N_val, W, F) float32
            - y_val: 1D vector (N_val,) float32
            - scaler: MinMaxScaler fitted on train_fold_df only (kept for traceability)
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

def scale_and_select_features(
    df: pd.DataFrame,
    scaler: MinMaxScaler,
    selector: BaseEstimator,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Applies a globally fitted scaler and feature selector to a dataset.

    ANTI-LEAKAGE PROTOCOL:
        Both artifacts must be fitted on 100% of the training set beforehand;
        here they are only used in transform mode, so the test set is never fitted.

    Args:
        df: Enriched DataFrame with 'unit_number', 'time' (and 'rul' for train).
        scaler: MinMaxScaler fitted on the full training set.
        selector: Fitted selector, or passthrough transformer when FS is disabled.
        feature_cols: Original names of the feature columns fed to the scaler.

    Returns:
        DataFrame with the identifier columns plus the selected feature columns,
        scaled and reduced in the same order for every dataset.
    """
    selected_names = get_selected_feature_names(selector, feature_cols)
    X_scaled = scaler.transform(df[feature_cols])
    X_selected = selector.transform(X_scaled)

    prepared = df[["unit_number", "time"]].copy()
    if "rul" in df.columns:
        prepared["rul"] = df["rul"].to_numpy()
    for idx, name in enumerate(selected_names):
        prepared[name] = X_selected[:, idx]

    return prepared


def prepare_test_data(
    test_prepared_df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int = 30,
) -> np.ndarray:

    """
    Builds the 3D test tensor keeping only the last window of each engine.

    The input frame must already be scaled and selected with the artifacts fitted
    on the full training set (see prepare_global_datasets / scale_and_select_features),
    so this helper only performs the windowing. Engines with fewer cycles than
    window_size get the same initial padding used during training.

    Args: 
        test_prepared_df: Test DataFrame already scaled and feature-selected,
            containing 'unit_number' and the selected feature columns.
        feature_cols: Names of the selected feature columns, in training order.
        window_size: Length of the temporal sequence.

    Returns:
        np.ndarray: Tensor 3D (N_motores_test, W, F) con la última ventana de cada motor
    """

    grouped = test_prepared_df.groupby('unit_number', sort = False)
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

def prepare_global_datasets(
    train_enriched_df: pd.DataFrame,
    test_enriched_df: pd.DataFrame,
    feature_cols: list[str],
    config: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Fits the scaler and the selector on 100% of train and builds the final tensors.

    ANTI-LEAKAGE PROTOCOL:
        The scaler and the feature selector are fitted once on the complete
        training set and then reused to transform both train and test. The test
        set is only ever transformed, never fitted. The resulting tensors are
        shared by every model, preserving the paired (blocking) design.

    Args:
        train_enriched_df: Training DataFrame with all engineered features.
        test_enriched_df: Test DataFrame with all engineered features.
        feature_cols: Original feature column names.
        config: Dictionary with the experiment configuration.

    Returns:
        tuple (X_train_full, y_train_full, X_test_final, global_features):
            - X_train_full: 3D tensor (N_train, W, F_global) float32
            - y_train_full: 1D vector (N_train,) float32
            - X_test_final: 3D tensor (N_engines_test, W, F_global) float32
            - global_features: Names of the features selected on the full train
    """
    w_size = config.get("data", {}).get("window_size", 30)
    fs_cfg = config.get("feature_selection", {})
    fs_enabled = fs_cfg.get("enabled", False)

    # 1. Global scaler fitted on 100% of the training set
    global_scaler = MinMaxScaler()
    global_scaler.fit(train_enriched_df[feature_cols])

    # 2. Global selector fitted on the scaled full training set (same recipe as the folds)
    global_selector = create_feature_selector(fs_cfg)
    global_selector.fit(
        global_scaler.transform(train_enriched_df[feature_cols]),
        train_enriched_df["rul"].to_numpy(),
    )
    global_features = get_selected_feature_names(global_selector, feature_cols)
    logger.info(
        f"Selector global ajustado con el 100% del train (enabled={fs_enabled}): "
        f"{len(global_features)}/{len(feature_cols)} features"
    )

    # 3. Full training tensors for the final retrain
    train_prepared = scale_and_select_features(
        train_enriched_df, global_scaler, global_selector, feature_cols
    )
    train_cols = ["unit_number", "time", "rul"] + global_features
    X_train_full, y_train_full = create_windows(
        train_prepared[train_cols], window_size=w_size, pad_strategy="edge"
    )

    # 4. Test tensor transformed with the very same scaler and selector
    test_prepared = scale_and_select_features(
        test_enriched_df, global_scaler, global_selector, feature_cols
    )
    X_test_final = prepare_test_data(test_prepared, global_features, window_size=w_size)

    logger.info(
        f"Tensores globales listos: train {X_train_full.shape} | test {X_test_final.shape}"
    )
    return X_train_full, y_train_full, X_test_final, global_features

