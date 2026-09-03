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

def preprocess_fold(X_train: np.ndarray, X_val: np.ndarray, scaler_type: str = 'minmax') -> tuple[np.ndarray, np.ndarray, object]:
    """
    Aplicación de función de escalamiento únicamente en los datos de entrenaiento, y transformar tambien el test.
    Previene de la fuga de datos asegurandose que las estadísticas de validación nunca se usan en el entrenamiento.

    Args:
        X_train: Datos de entrenamiento, shape (n_train, n_features)
        X_val: Datos de validación, shape (n_val, n_features)
        scaler_type: "minmax" para MinMaxScaler, "zscore" para StandardScaler (default: "minmax")
    
    Returns:
        tupla (X_train_scaled, X_val_scaled, scaler) donde:
            - X_train_scaled: Datos de entrenamiento escalados
            - X_val_scaled: Datos de validación escalados
            - scaler: objeto del escalador usado para transformar los datos
    Raises:
        ValueError: si el tipo de escalador no es reconocido.
    """
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'zscore':
        scaler = StandardScaler()
    else:
        raise ValueError(f"scaler_type debe ser 'minmax' o 'zscore', no '{scaler_type}'")

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    return X_train_scaled, X_val_scaled, scaler



if __name__ == "__main__":

    import pandas as pd

    X_train = pd.DataFrame({
        'feature': [np.random.randint(0, 100) for _ in range(100)]
    })

    X_val = pd.DataFrame({
        'feature': [np.random.randint(0, 100) for _ in range(100)]
    })
    result = preprocess_fold(X_train.values, X_val.values, scaler_type='minmax')

    print(f"X_train_scaled: {result[0][:5]}")
    print(f"X_val_scaled: {result[1][:5]}") 
    print(f"Scaler: {result[2]}")
