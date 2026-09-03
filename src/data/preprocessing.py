"""
Preprocessing modules for RUL-CMAPSS pipeline.

Provides data preparation functions with strict data-leakage prevention:
GroupKFold splitting, sensor filtering, Piecewise Linear RUL, and
per-fold scaling.
"""

import numpy as np
import pandas as pd


def create_groups(df: pd.DataFrame) -> np.ndarray:
    """Extae unit para for GroupKFold splitting.

    Retorna un arreglo de forma (n_samples,) donde cada elemento es el
    unit_number de la fila correspondiente. Este arreglo es adecuado para
    pasarlo a sklearn.model_selection.GroupKFold.split().

    Args:
        df: DataFrame con una columna ``unit_number``.

    Returns:
        numpy array de la unidad IDs alineado con los df rows.

    Raises:
        ValueError: si no se tiene un ``unit_number`` en el df.
    """
    if "unit_number" not in df.columns:
        raise ValueError("DataFrame must contain a 'unit_number' column")

    return df["unit_number"].values


def remove_constant_sensors(
    df: pd.DataFrame, sensors_to_remove: list[int]
) -> pd.DataFrame:
    """Remove constant-variance sensors from the DataFrame.

    Drops columns ``sensor_<id>`` for each id in ``sensors_to_remove``.
    Columns that do not exist in df are silently ignored.

    Args:
        df: DataFrame with sensor columns named ``sensor_<id>``.
        sensors_to_remove: List of sensor IDs (ints) to drop.

    Returns:
        DataFrame without the specified sensor columns.
    """
    cols_to_drop = [
        f"sensor_{sid}" for sid in sensors_to_remove
        if f"sensor_{sid}" in df.columns
    ]
    return df.drop(columns=cols_to_drop)
