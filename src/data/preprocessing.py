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

def full_preprocessing(subset: str = "FD001", config_path: str = "configs/config_FD001.yaml") -> dict:
    """
    Ejecuta el pipeline de preprocesamiento completo de preprocesamiento para el subset especificado de CMAPSS.

    1. Carga los datos
    2. Elimina los sensores constantes
    3. Calcula el RUL piecewise lineal
    4. Escala los datos por fold usando GroupKFold para prevenir la fuga de datos.

    pipeline:
        - load_data
        - remove_constant_sensors
        - compute_piecewise_rul
    
    Args:
        subset: CMPASS subset de los 4 disponibles (FD001, FD002, FD003, FD004)
        config_path: ruta al archivo de configuración YAML de los 4 disponibles
    
    Returns:
        Diccionario con las keys:
            - "train" : Datos de entrenaiento limpios con RUL calculado y sensores constantes removidos
            - "test"  : Datos de validación limpios con RUL calculado y sensores constantes removidos
            - "rul"   : RUL del conjunto de validación
            - "config": configuración cargada desde el archivo YAML
    Raises:
        FileNotFoundError: si el archivo de configuraciónno existe
        ValueError: si el subset del archivo de configuración no existe
    """
    import yaml
    from src.data.loader import load_cmapss

    # Cargar la configuración
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Archivo de configuración no encontrado: {config_path}")

    if config.get("subset") != subset:
        raise ValueError(f"Subset no encontrado en el archivo de configuración: {subset}")

    # Cargar los datos crudos
    train_df, test_df, rul = load_cmapss(subset)

    # Remover los sensores constantes definidos en el experimento
    sensors_to_remove = config['sensors']['remove']
    train = remove_constant_sensors(train_df, sensors_to_remove)
    test = remove_constant_sensors(test_df, sensors_to_remove)

    # Calcular el RUL piecewise
    rul_max = config['data']['rul_max']
    train = compute_piecewise_rul(train, rul_max=rul_max)

    # Regresar resultado procesado
    return{
        'train' : train,
        'test' : test,
        'rul' : rul,
        'config' : config
    }



if __name__ == "__main__":

    result = full_preprocessing("FD001", "configs/config_FD001.yml")
    print("Train shape:", result["train"].shape)
    print("Test shape:", result["test"].shape)
    print("RUL shape:", result["rul"].shape)
    print("RUL:", result['rul'])
    print()
    print("Train columns:", list(result["train"].columns))
    print()
    print("RUL max:", result["train"]["rul"].max())
    print("RUL min:", result["train"]["rul"].min())
    print()
    print(result["train"].head())
