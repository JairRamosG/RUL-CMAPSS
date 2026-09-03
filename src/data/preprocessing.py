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


def remove_constant_sensors(df: pd.DataFrame, sensors_to_remove: list[int]) -> pd.DataFrame:
    """Remueve sensores con varianza constante del DataFrame.

    Elimina las columnas ``sensor_<id>`` por cada id en la lista ``sensors_to_remove``.
    Las columnas que no existen en el DataFrame van a ser ignoradas.

    Args:
        df: DataFrame con columnas de sensor llamadas ``sensor_<id>``.
        sensors_to_remove: Lista de IDs de sensores (ints) para eliminar.

    Returns:
        DataFrame sin las columnas especificadas.
    """
    cols_to_drop = [f"sensor_{sid}" for sid in sensors_to_remove if f"sensor_{sid}" in df.columns]

    return df.drop(columns=cols_to_drop)

def compute_piecewise_rul(df: pd.DataFrame, rul_max: int = 125) -> pd.DataFrame:
    """
    Calcula el RUL de manera piecewise linear para cada unidad de sensores en el DataFrame.

    Para cada unidad, el RUL empieza en un rul_max y va decreciendo linealmente a 0
    hasta llegar a la falla del ciclo de operación.

    RUL(t) = min(T_failure - t, rul_max)

    Esto previene penalizar el modelo durante la etapa saludable cuando inicia su operación. (Heimes, 2008).

    Args:
        df: DataFrame con columnas 'unit_number' y 'time'
        threshold: valor maximo del RUL (default: 125)

    Returns:
        DataFrame con una nueva columna 'RUL' que contiene el RUL calculado para cada fila.

    Raises:
        ValueError: si no se tienen las columnas requeridas en el DataFrame.
    """

    required = {'unit_number', 'time'}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"DataFrame debe contener las columnas: {missing}")

    result = df.copy()

    # Ciclo de falla
    failure_cycle = result.groupby('unit_number')['time'].transform('max')

    #Piecewise RUL
    raw_rul = failure_cycle - result['time']
    result['rul'] = raw_rul.clip(upper=rul_max)

    return result   


if __name__ == "__main__":

    import pandas as pd

    df = pd.DataFrame({
        'unit_number': [1, 1, 1, 1, 1],
        'time': [1, 2, 3, 4, 5],
    })
    result = compute_piecewise_rul(df)
    print(result)
    print()
    print('RUL:', list(result['rul']))
    print('Expected: [4, 3, 2, 1, 0]')

