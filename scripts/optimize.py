"""
CLI de la optimización de hiperparámetros con Optuna

Uso:
    uv run python scripts/optimize.py --config configs/config_FD001.yaml \
        [--models random_forest xgboost] [--dry-run] \
        [--tuned-params-out tuned/config_FD001_tuned.json]
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlflow  
import optuna

from scripts.train_eval import(
    build_model,
    extract_features,
    init_mlflow,
    load_config,
    prepare_raw_data,
    set_seed
)

from src.models.optimization import(
    export_best_params,
    optimize_hyperparameters,
    validate_optimization_config
)

logging.baseConfig(
    level = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S"   
)

OPTUNA_STORAGE = "sqlite:///optuna.db"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = "Pipeline de optimización de hiperparámetros con Optuna y tracking con MLflow",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--config",
        type = str,
        default = "configs/config_FD001.yaml",
        help = "Ruta del archivo YAML para el experimento"
    )

    parser.add_argument(
        "--models",
        nargs = "+",
        default = None,
        help = "Subset de modelos del bloque de optimization.models para optimizar (default: todos)",
    )

    parser.add_argument(
        "--dry-run",
        action = "store_true",
        help = "Modo de pruebas rapido para ejecutar solo una parte de los datos con 3 trials y 2 folds",
    )

    parser.add_argument(
        "tuned-params-out",
        type = str,
        default = None,
        help = "Ruta de salida para el JSON con los mejores hiperparámetros encontrados (default: tuned/<subset>_tuned.json)"
    )

    return parser.parse_args()


def _existing_trials(study_name: str) -> int:
    """
    Trials que ya existen registrados en el Study de Optuna (0 si no existe aún).
    """
    try:
        return len(optuna.load_study(study_name = study_name, storage = OPTUNA_STORAGE).trials)
    except KeyError:
        return 0

    
