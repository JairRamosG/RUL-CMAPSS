from typing import Tuple, Dict, Any

from src.models.base import BaseModel
from src.models.RFModel import RFModel
from src.models.XGBoostModel import XGBoostModel
from src.models.LightGBMModel import LightGBMModel
from src.models.SVRModel import SVRModel
from src.models.MLPModel import MLPModel
from src.models.CNN1DModel import CNN1DModel
from src.models.LSTMModel import LSTMModel


def build_model(
        model_name : str,
        model_params : dict[str, Any],
        input_shape : tuple[int, int],
) -> BaseModel:
    """
    Instancia dinámicamente los modelos de ML Clásico y DL.
    Conecta hiperparámetros del YAML y la forma de entrada temporal con el constructor
    específico para cada algorítmo de las categorías.

    Args: 
        model_name: Identificador del modelo
        model_params: Diccionario de hiperparámetros de la configuración
        input_shape: TUpla (window_size, num_features) para el tensor 3D
    
    Returns:
        Instancia configurada que hereda del BaseModel (fit, predict, get_params)

    Raises:
        ValueError: Si el nombre del modelo no se reconoce
        NotImplementedError: Para la memoria asociativa
    """

    window_size, num_features = input_shape
    input_dim_flattened = window_size * num_features
    params = model_params.copy() if model_params else {}

    # Modelos basados en ML Clásico (Arboles y Kernel)
    if model_name in ("random_forest", "rf"):
        return RFModel(**params)

    elif model_name == "xgboost":
        return XGBoostModel(**params)

    elif model_name == "lightgbm":
        return LightGBMModel(**params)

    elif model_name == "svr":
        return SVRModel(**params)

    # Modelos basados en DL
    elif model_name == "mlp":
        return MLPModel(input_dim= input_dim_flattened, **params)

    elif model_name == "cnn1d":
        return CNN1DModel(num_features=num_features, window_size=window_size, **params)

    elif model_name == "lstm":
        return LSTMModel(num_features=num_features, **params)

    # Memorias asociativas
    elif model_name == "asociative_memory":
        raise NotImplementedError("Hay que programar la memoria asociativa")

    else:
        raise ValueError(f"Modelo no soportado: {model_name}")