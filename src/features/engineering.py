# Feature engineering modules

import pandas as pd
import numpy as np

def create_windows(df: pd.DataFrame, window_size: int = 30, pad_strategy: str = 'edge') -> tuple[np.ndarray, np.ndarray]:
    """
    Construye la representación final de un DataFrame
    Extrae un DataFrame que contiene variables temporales (los lags, rolling stats, EWMA, tendencias), crea las ventanas
    por característica y regresa el DataFrame listo para usar en un modelo.

    Padding: Para los motores que tienen medidas menores a el tamaño de la ventana, se va a repetir el primer valor para
    para evitar que el modelo capture una degradación artificial desde un inicio, porque eso hace que no tome en cuenta
    la etapa en la que los motores funcionan normalmente antes de que aparezca el fenomeno de degradación en los sensores.

    Args:
        df : DataFrame
        window_size: Numero de ciclos por cada ventana (bibliografía usan 30)
        pad_strategy: edge para repetir el valor del inicio

    Returns:
        Tuple : (X, y)
            - X : np.ndarray (n_samples, window_size, n_features)
            - y : np.ndarray (n_samples, )
    Raises:
        ValueError: si hay columnas faltantes o no existe el padding especificado
    """

    required = {'unit_number', 'time'}
    if  not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f'El dataset no tiene las columnas: {missing}')

    if window_size < 1:
        raise ValueError(f'El valor de la ventana debe ser mayor a 1, se tiene: {window_size}')

    if pad_strategy != 'edge':
        raise ValueError(f'pad_strategy: {pad_strategy} no permitida')

    # Solo las columnas que se usan en un entrenamiento
    exclude_columns = {'unit_number', 'time', 'rul'}
    feature_cols = [col for col in df.columns if col not in exclude_columns]
    if not feature_cols:
        raise ValueError(f'No se tienen columnas para entrenar en el DataFrame')

    # Asegurar el orden correcto
    df = df.sort_values(['unit_number', 'time']).reset_index(drop=True)

    # Calcular el RUL si no lo trae
    if "rul" in df.columns:
        rul_series = df["rul"]
    else:
        failure_cycle = df.groupby("unit_number")["time"].transform("max")
        rul_series = (failure_cycle - df["time"]).clip(upper=125)

    grouped = df.groupby("unit_number", sort=False)
    windows: list[np.ndarray] = []
    targets: list[float] = []

    for _, group in grouped:
        values = group[feature_cols].values       # (T, F)
        rul_vals = rul_series.loc[group.index].values  # (T,)
        T = len(values)

        if T == 0:
            continue

        if T >= window_size:
            # Enough history — tomar estadísticas validas
            for start in range(T - window_size + 1):
                end = start + window_size
                windows.append(values[start:end])
                targets.append(float(rul_vals[end - 1]))
        else:
            # Edge padding: repetir el primer valor
            pad_len = window_size - T
            pad_block = np.tile(values[0:1], (pad_len, 1))
            padded = np.concatenate([pad_block, values], axis=0)
            # Single window covering the full motor history
            windows.append(padded)
            targets.append(float(rul_vals[T - 1]))

    X = np.array(windows)
    y = np.array(targets)

    return X, y

def compute_rolling_stats(df: pd.DataFrame, window_size : int = 30, stat_types : list[str] | None = None) -> pd.DataFrame:
    """
    Computo de las características que vienen en el libro, las de tipo Time Delay (lags, rolling_windows, seasonal_rolling_windows y EWMA)
    Para cada sensor calcula las estadísticas de ciclos del pasado y los agrega a nuevas columnas para el DataFrame.
    Las estadísticas están calculadas por 'unit_number', la primera es 'window_size-1', van a haber NaN si no alcanzan valores

    Args:
        df: DataFrame limpio desde create_window
        window_size: numero de ciclos pasados a considerar (30 de literatura)
        stat_types: lista de funciónes de agregación ['mean', 'std', 'min', 'max']

    Returns:
        pd.DateFrame: Nuevo DataFrame con todas las nuevas características temporales calculadas
    
    Raises:
        ValueError: Si hay campos faltantes para procesar el DataFrame
    """

    if stat_types is None:
        stat_types = ['mean', 'std', 'min', 'max']

    if not stat_types:
        raise ValueError('No existen etadísticos para las variables Time Delay Embedding')

    required = {'unit_number', 'time'}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f'El DataFrame debe contener: {missing}')

    exclude = {'unit_number', 'time', 'rul'}
    feature_cols = [col for col in df.columns if col not in exclude]
    if not feature_cols:
        raise ValueError(f'No hay columnas para calcular en el DataFrame')

    result = df.copy()
    result = result.sort_values(['unit_number', 'time']).reset_index(drop=True)

    grouped = result.groupby('unit_number', sort = False)

    for col in feature_cols:
        for stat in stat_types:
            col_name = f"{col}_{stat}"
            result[col_name] = grouped[col].transform(lambda x: getattr(x.rolling(window_size, min_periods=1), stat)())

    return result




if __name__ == '__main__':

    from src.data.preprocessing import full_preprocessing

    result = full_preprocessing("FD001", "configs/config_FD001.yaml")
    print("Train shape:", result["train"].shape)
    print("Test shape:", result["test"].shape)
    print("RUL shape:", result["rul"].shape)
    print()
    print("Train columns:", list(result["train"].columns))
    print()
    print("RUL max:", result["train"]["rul"].max())
    print("RUL min:", result["train"]["rul"].min())
    print()

    #compute_rolling_stats---------------------------------------------------------------
    X = result['train']
    X_fe = compute_rolling_stats(X)
    print(f'X_fe shape: {X_fe.shape}')

    #create_windows---------------------------------------------------------------
    X_win, y = create_windows(X_fe, window_size = 30, pad_strategy = 'edge')
    print(f"\nX shape: {X_win.shape}")   # (n_windows, 30, n_features)
    print(f"y shape: {y.shape}")     # (n_windows,)
    print(f"y primeros 5: {y[:5]}")
    print(f"y min: {y.min()}, y max: {y.max()}")


