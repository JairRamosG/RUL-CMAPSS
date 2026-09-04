# Feature engineering modules

import pandas as pd
import numpy as np

def compute_rolling_stats(df: pd.DataFrame, window_size: int = 30, stat_types: list[str] | None = None) -> pd.DataFrame:
    """
    Calcula estadísticas móviles temporales (Time Delay Embeddings) por cada motor.

    Para cada variable/sensor del DataFrame, calcula métricas acumulativas en una
    ventana deslizante hacia el pasado para capturar la tendencia de degradación
    del sistema sin incurrir en Data Leakage (fuga de datos del futuro).

    Args:
        df (pd.DataFrame): DataFrame ordenado que contiene las variables temporales
            y las columnas obligatorias 'unit_number' y 'time'.
        window_size (int, optional): Tamaño de la ventana retrospectiva en ciclos.
            Por defecto es 30 (estándar en la literatura CMAPSS).
        stat_types (list[str] | None, optional): Lista de funciones de agregación a
            calcular. Opciones soportadas: 'mean', 'std', 'min', 'max'.
            Por defecto es ['mean', 'std', 'min', 'max'].

    Returns:
        pd.DataFrame: Nuevo DataFrame enriquecido que conserva todas las columnas
            originales junto con las nuevas columnas calculadas bajo la nomenclatura
            `<sensor>_<stat_type>` (ej. `sensor_2_mean`).

    Raises:
        ValueError: Si `stat_types` está vacío o no es válido.
        ValueError: Si al DataFrame le faltan las columnas 'unit_number' o 'time'.
        ValueError: Si no existen columnas numéricas de sensores/características para procesar.
    """
    if stat_types is None:
        stat_types = ['mean', 'std', 'min', 'max']

    if not stat_types:
        raise ValueError('No existen estadísticos para las variables Time Delay Embedding')

    required = {'unit_number', 'time'}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f'El DataFrame debe contener: {missing}')

    exclude = {'unit_number', 'time', 'rul'}
    feature_cols = [col for col in df.columns if col not in exclude]
    if not feature_cols:
        raise ValueError('No hay columnas para calcular en el DataFrame')

    result = df.copy().sort_values(['unit_number', 'time']).reset_index(drop=True)
    
    # Acumular en un diccionario las estadísticas SOLO de los sensores base
    new_cols = {}
    grouped = result.groupby('unit_number', sort=False)
    
    for col in feature_cols:
        rolled = grouped[col].rolling(window_size, min_periods=1)
        for stat in stat_types:
            col_name = f"{col}_{stat}"
            # Extraer y alinear limpiamente
            new_cols[col_name] = getattr(rolled, stat)().reset_index(level=0, drop=True)

    new_features_df = pd.DataFrame(new_cols, index=result.index)
    return pd.concat([result, new_features_df], axis=1)


def compute_trends(df: pd.DataFrame, delta_steps: list[int] | None = None, base_features_only: bool = True) -> pd.DataFrame:
    """
    Calcula las diferencias finitas (deltas/tendencias) de orden $k$ por cada motor.

    Aplica la fórmula de derivación discreta $\Delta S_t = S_t - S_{t-k}$ para capturar
    la velocidad de cambio e instabilidad en los sensores conforme el motor se degrada.

    Args:
        df (pd.DataFrame): DataFrame ordenado con columnas de sensores y los
            identificadores 'unit_number' y 'time'.
        delta_steps (list[int] | None, optional): Pasos discretos hacia el pasado ($k$)
            para calcular las diferencias. Por defecto es [1].
        base_features_only (bool, optional): Si es True, calcula los deltas únicamente
            sobre las señales base de los sensores, ignorando las columnas estadísticas
            generadas por rolling stats. Evita la explosión de dimensionalidad.
            Por defecto es True.

    Returns:
        pd.DataFrame: DataFrame enriquecido que incluye las columnas originales y las
            nuevas características con el sufijo `<sensor>_delta<k>` (ej. `sensor_2_delta1`).

    Raises:
        ValueError: Si al DataFrame le faltan las columnas 'unit_number' o 'time'.
        ValueError: Si algún valor $k$ en `delta_steps` es menor o igual a 0.
        ValueError: Si no existen columnas válidas para calcular las tendencias.
    """
    if delta_steps is None:
        delta_steps = [1]

    required = {'unit_number', 'time'}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f'Le faltan las columnas {missing} al DataFrame')

    if any(k <= 0 for k in delta_steps):
        raise ValueError('Los valores K deben ser enteros positivos')

    exclude = {'unit_number', 'time', 'rul'}
    
    # Filtrar para calcular deltas SOLO de los sensores originales y no de las rolling stats
    if base_features_only:
        feature_cols = [col for col in df.columns if col not in exclude and not any(col.endswith(f"_{s}") for s in ['mean', 'std', 'min', 'max'])]
    else:
        feature_cols = [col for col in df.columns if col not in exclude]

    if not feature_cols:
        raise ValueError('No hay valores de sensores en el DataFrame')

    result = df.copy().sort_values(['unit_number', 'time']).reset_index(drop=True)
    grouped = result.groupby('unit_number', sort=False)

    new_cols = {}
    for col in feature_cols:
        for k in delta_steps:
            col_name = f'{col}_delta{k}'
            new_cols[col_name] = grouped[col].diff(periods=k).fillna(0.0)

    new_features_df = pd.DataFrame(new_cols, index=result.index)
    return pd.concat([result, new_features_df], axis=1)


def create_windows(df: pd.DataFrame, window_size: int = 30, pad_strategy: str = 'edge') -> tuple[np.ndarray, np.ndarray]:
    """
    Transforma un DataFrame tabular en tensores tridimensionales para aprendizaje profundo.

    Estructura secuencias temporales en ventanas deslizantes por cada unidad de motor.
    Si un motor registra un historial menor al tamaño de ventana ($T < W$), aplica un
    relleno (padding) repitiendo la primera observación para prevenir artefactos de degradación.

    Args:
        df (pd.DataFrame): DataFrame enriquecido con las características seleccionadas.
            Debe contener las columnas 'unit_number' y 'time'.
        window_size (int, optional): Longitud de la secuencia temporal (pasos de tiempo)
            para cada ventana $W$. Por defecto es 30.
        pad_strategy (str, optional): Estrategia de rellenado para motores con pocos ciclos.
            Actualmente solo admite 'edge' (repite el estado operacional inicial).
            Por defecto es 'edge'.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - X (np.ndarray): Tensor 3D de entradas con forma `(N_muestras, window_size, N_características)`
                en formato `float32`.
            - y (np.ndarray): Vector 1D de etiquetas RUL (Remaining Useful Life) objetivo
                con forma `(N_muestras,)` en formato `float32`.

    Raises:
        ValueError: Si faltan las columnas 'unit_number' o 'time'.
        ValueError: Si `window_size` es menor a 1.
        ValueError: Si `pad_strategy` no es una estrategia permitida ('edge').
        ValueError: Si no existen columnas de características (sensores) para construir la ventana.
    """
    required = {'unit_number', 'time'}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f'El dataset no tiene las columnas: {missing}')

    if window_size < 1:
        raise ValueError(f'El valor de la ventana debe ser mayor a 1, se tiene: {window_size}')

    if pad_strategy != 'edge':
        raise ValueError(f'pad_strategy: {pad_strategy} no permitida')

    exclude_columns = {'unit_number', 'time', 'rul'}
    feature_cols = [col for col in df.columns if col not in exclude_columns]
    if not feature_cols:
        raise ValueError('No se tienen columnas para entrenar en el DataFrame')

    # Asegurar orden e índice limpio
    df_sorted = df.sort_values(['unit_number', 'time']).reset_index(drop=True)

    # Calcular RUL si no viene precalculado
    if "rul" in df_sorted.columns:
        df_sorted["target_rul"] = df_sorted["rul"]
    else:
        failure_cycle = df_sorted.groupby("unit_number")["time"].transform("max")
        df_sorted["target_rul"] = (failure_cycle - df_sorted["time"]).clip(upper=125)

    grouped = df_sorted.groupby("unit_number", sort=False)
    windows: list[np.ndarray] = []
    targets: list[float] = []

    for _, group in grouped:
        values = group[feature_cols].values       # Shape: (T, F)
        rul_vals = group["target_rul"].values      # Shape: (T,)
        T = len(values)

        if T == 0:
            continue

        if T >= window_size:
            # Ventana deslizante para motores con suficientes ciclos
            for start in range(T - window_size + 1):
                end = start + window_size
                windows.append(values[start:end])
                targets.append(float(rul_vals[end - 1]))
        else:
            # Padding al inicio si el motor tiene menos ciclos que el window_size
            pad_len = window_size - T
            pad_block = np.tile(values[0:1], (pad_len, 1))
            padded = np.concatenate([pad_block, values], axis=0)
            windows.append(padded)
            targets.append(float(rul_vals[-1]))

    X = np.array(windows, dtype=np.float32)
    y = np.array(targets, dtype=np.float32)

    return X, y


if __name__ == '__main__':
    from src.data.preprocessing import full_preprocessing

    result = full_preprocessing("FD001", "configs/config_FD001.yaml")
    X = result['train']

    # 1. Aplicar FE controlado
    X_fe = compute_rolling_stats(X, window_size=30)
    X_fe = compute_trends(X_fe, delta_steps=[1, 2, 3], base_features_only=True)

    print(f"Columnas totales tras FE: {X_fe.shape[1]}")

    # 2. Generar ventanas 3D
    X_win, y = create_windows(X_fe, window_size=30, pad_strategy='edge')

    print(f"X_win shape (N, W, F): {X_win.shape}")
    print(f"y shape: {y.shape}")
