"""
Preprocessing modules for RUL-CMAPSS pipeline.

Provides data preparation functions with strict data-leakage prevention:
GroupKFold splitting, sensor filtering, Piecewise Linear RUL, and
per-fold scaling.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


class OperatingRegimeNormalizer(BaseEstimator, TransformerMixin):
    """
    Estandarización condicionada por Régimen Operativo (ORN)

    Elimina la varianza exógena de variables operativas (altitud, Mach, TRA) de los sensores
    restando la media y dividiendo por la desviación estándar de cada regimen operativo para que
    solo quede el patrón de degradación.

    Args:
        n_regimenes: Número de clusters en KMeans (6 de los YAML)
        random_state: 42 para reproducibilidad
        n_init: Inicializaciónes de los KMeans
        eps: Espsilon para evitar divisiones con 0
        settings_mode: Qúe se hace con los settings despues del cluster
            "drop" = eliminarlos
            "onehot" = reemplazarlos por columnas
    """

    SETTINGS_COLS = ["setting_1", "setting_2", "setting_3"]

    def __init__(
            self, 
            n_regimenes: int = 6, 
            random_state: int = 42,
            n_init: int = 10,
            eps: float = 1e-6,
            settings_mode: str = "drop"
    ) -> None:
        
        if settings_mode not in ["drop", "onehot"]:
            raise ValueError(f"settings mode solo es drop o onehot, se tiene: {settings_mode}")

        self.n_regimenes = n_regimenes
        self.random_state = random_state
        self.n_init = n_init
        self.eps = eps
        self.settings_mode = settings_mode
        self._fitted = False

    def fit(self, df: pd.DataFrame, y = None) -> "OperatingRegimeNormalizer":
        """
        Aprende del artefacto ORN solo con los datos de entrenamiento

        Ajusta: Escalador de settings, KMeans y tablas con la media y desviación estándar por (regimen, sensor)

        Args:
            df: pd.DataFrame de TRAIN con settings, sensores y columnas base
        
        Returns:
            self para poder ejecutar un fit_transform

        Raises:
            ValueError si faltan columnas requeridas
        """

        required = {"unit_number", "time", *self.SETTINGS_COLS} # es porque es una variable global?
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"ORN: faltan columnas requeridas: {missing}")

        self.sensor_cols_ = sorted(c for c in df.columns if c.startswith("sensor_"))
        if not self.sensor_cols_:
            raise ValueError(f"No se tienen columnas de sensor que normalizar")

        # Paso 1: escalar los settings
        self.settings_scaler_ = StandardScaler()
        settings_scaled = self.settings_scaler_.fit_transform(df[self.SETTINGS_COLS])

        # Paso 2: KMeans para etiquetar el régimen del ciclo
        self.kmeans_ = KMeans(
            n_clusters = self.n_regimenes,
            random_state = self.random_state,
            n_init = self.n_init
        )
        labels = self.kmeans_.fit_predict(settings_scaled)

        # Paso 3: mu y sigma por cada regimen
        sensors_df = df[self.sensor_cols_].copy()
        sensors_df["regimen"] = labels
        self.regime_means_ = sensors_df.groupby("regimen").mean()                         
        self.regime_stds_ = sensors_df.groupby("regimen").std().fillna(self.eps).clip(lower=self.eps)

        self._fitted = True
        return self

    def transform(self, df : pd.DataFrame):
        """
        Aplicar el artefacto aprendido en fit() sin reajustar nada

        LOs settings del DataFrame se asugnan al cluster más cercano con kmeans.predict() y los
        sensores se estandarizan con las tablas mu/sigma del train.

        Args:
            df: pd.DataFrame(train ya ajustado o test nunca visto)
        
        Returns:
            Nuevo DataFrame con los sensores normalizados y los settings gestionados por settings_mode

        Raises:
            RuntimeError si se llama fit antes que train
            ValueError si faltan sensores en el fit
        """

        if not self._fitted:
            raise RuntimeError("No se puede ajustar sin antes entrenar")

        missing = [c for c in self.sensor_cols_ if c not in df.columns]
        if missing:
            raise ValueError(f"ORN: faltan sensores vistos en fit: {missing}")

        out = df.copy()

        # Asignación de régimen del test con el MISMO escalador y K-Means
        settings_scaled = self.settings_scaler_.transform(out[self.SETTINGS_COLS])
        labels = self.kmeans_.predict(settings_scaled)

        # z-score condicional: (x - mu_del_régimen) / sigma_del_régimen
        # .loc[labels] trae la fila mu/sigma de CADA fila según su régimen
        mu = self.regime_means_.loc[labels][self.sensor_cols_].to_numpy()
        sigma = self.regime_stds_.loc[labels][self.sensor_cols_].to_numpy()
        out[self.sensor_cols_] = (out[self.sensor_cols_].to_numpy() - mu) / sigma

        # Gestión de los settings según settings_mode
        if self.settings_mode == "drop":
            out = out.drop(columns=self.SETTINGS_COLS)
        elif self.settings_mode == "onehot":
            regime_dummies = pd.get_dummies(
                pd.Categorical(labels, categories=range(self.n_regimenes)),
                prefix="regime", dtype=float,
            )
            regime_dummies.index = out.index
            out = out.drop(columns=self.SETTINGS_COLS)
            out = pd.concat([out, regime_dummies], axis=1)

        return out


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
