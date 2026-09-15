"""
Pipeline de entrenamiento, evaluación y tracking de RUL para el C-MAPSS.

Parte 1. Configuración, CLI y Reproducibilidad.

Uso:
    python scripts/train_eval.py --help
    python scripts/train_eval.py --help
    python scripts/train_eval.py --config configs/config_FD001.yaml
    python scripts/train_eval.py --config configs/config_FD001.yaml --models random_forest mlp
    python scripts/train_eval.py --config configs/config_FD001.yaml --dry-run

"""

import argparse
import logging
from pathlib import Path
import random
import sys
from typing import Optional

import numpy as np
import torch
import yaml

# Configuracción de los loggings
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_eval")

# Parsing para la línea de comandos
def parse_args() -> argparse.Namespace:
    """
    Parseo para la línea de comandos

    Returns:
        argparse.Namespace con los argumentos
    """
    parser = argparse.ArgumentParser(
        description= "Pipeline de entrenamiento, evaluación y tracking con MLflow para el C-Mapss",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type = str,
        default= "configs/config_FD001.yaml",
        help= "Ruta del archivo YAML de configuración del experimento"
    )

    parser.add_argument(
        "--models",
        args="+",
        default=None,
        help="Lista de modelos específicos a evaluar"
            "Si no se especifican, se evalúan todos los modelos"
    )

    parser.add_argument(
        "--dry-run",
        action = "store_true",
        help="Modo de pruebas rápido con datos/folds reducidos para ahorro computacional"
    )

    return parser.parse_args()