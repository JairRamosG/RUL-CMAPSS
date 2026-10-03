"""Carga, validación y fusión de archivos de configuración de experimentos."""
import json
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


# Cargar el archivo de configuración
def load_config(config_path:str | Path) -> dict:
    """
    Carga y valida los archivos de configuración YAML de los experimentos.

    Args:
        config_path: Ruta del archivo YAML

    Returns:
        dict con la información del experimento
    
    Raises:
        FileNotFoundError: Si el archivo no existe en el sistema de archivos
        ValueError: Le faltan secciónes al archivo de configuración
    """

    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de configuración en : {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Validación de las secciónes del archivo de configuración YAML
    required_sections = ["subset", "data", "models", "evaluation", "experiment"]
    missing = [sec for sec in required_sections if sec not in config]
    if missing:
        raise ValueError(f"El archivo {path.name} no es válido para los experimentos. Le falta: {missing}")
    
    logger.info(f"Configuración correcta cargada desde: {path.name}")
    return config

def apply_tuned_params(config: dict, tuned_path: str | Path) -> dict:
    """
    Fusiona los hiperparámetros afinados exportados por scripts/optimize.py sobre
    config["models"][i]["params"]: los defaults del YAML se conservan y en los
    keys en conflicto prevalece el valor afinado.

    Los modelos sin entrada en el JSON conservan los defaults del YAML.
    """
    tuned = json.loads(Path(tuned_path).read_text(encoding="utf-8"))
    for model in config.get("models", []):
        name = model.get("name")
        if name not in tuned:
            logger.info(f"'{name}' sin afinación en {tuned_path}: usando defaults del YAML")
            continue
        if not isinstance(tuned[name], dict):
            raise ValueError(
                f"tuned-params: la entrada de '{name}' debe ser un dict de parámetros, "
                f"es {type(tuned[name]).__name__}"
            )
        logger.info(f"Hiperparámetros afinados fusionados sobre defaults del YAML en {name}: {tuned[name]}")
        model["params"] = {**model.get("params", {}), **tuned[name]}
    return config