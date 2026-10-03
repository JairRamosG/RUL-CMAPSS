"""
Pipeline de entrenamiento, evaluación y tracking de RUL para el C-MAPSS.

Parte 1. Configuración, CLI y Reproducibilidad.

Uso:
    python scripts/train_eval.py --help
    python scripts/train_eval.py --help
    python scripts/train_eval.py --config configs/config_FD001.yaml
    python scripts/train_eval.py --config configs/config_FD001.yaml --models random_forest mlp
    python scripts/train_eval.py --config configs/config_FD001.yaml --dry-run
    python scripts/train_eval.py --models random_forest mlp lstm --dry-run

    train_eval consume ese único archivo con todos los modelos
    uv run python scripts/train_eval.py --config configs/config_FD001.yaml --tuned-params tuned/config_FD001_tuned.json

"""

import argparse
import logging
from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_cmapss
from src.data.preprocessing import remove_constant_sensors, compute_piecewise_rul
from src.features.engineering import compute_rolling_stats, compute_trends, create_windows
from src.features.selection import create_feature_selector, get_selected_feature_names

from sklearn.model_selection import GroupKFold
from sklearn.base import BaseEstimator
from src.evaluation.metrics import rmse, mae, nasa_score, profile_resource_usage
from src.models.pytorch_wrapper import PyTorchModel

# -----------------------------------------
from src.models.base import BaseModel
from src.models.factory import build_model

from src.evaluation.cross_validation import(
    prepare_cv_feature_selection,
    evaluate_model_cv,
    evaluate_on_test_set
    )

from src.features.windowing import (scale_and_window_fold,
                                    scale_and_select_features,
                                    prepare_test_data,
                                    prepare_global_datasets)   

from src.tracking.mlflow_reporter import (
    init_mlflow,
    log_model_run_to_MLflow,
    log_omnibus_comparission
)

from src.utils.config import load_config, apply_tuned_params
from src.data.pipeline import prepare_raw_data, extract_features
from src.utils.reproducibility import set_seed


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
        nargs="+",
        default=None,
        help="Lista de modelos específicos a evaluar"
            "Si no se especifican, se evalúan todos los modelos"
    )

    parser.add_argument(
        "--dry-run",
        action = "store_true",
        help="Modo de pruebas rápido con datos/folds reducidos para ahorro computacional"
    )

    parser.add_argument(
        "--tuned-params",
        type = str,
        default = None,
        help="Ruta al JSON de hiperparámetros afinados exportado por scripts/optimize.py "
            "(p.ej. tuned/config_FD001_tuned.json). Si se omite se usan los defaults del YAML"
    )

    return parser.parse_args()

def main() -> None:
    """
    Función principal para ensamblar el pipeline
    """
    # Establecer las configuraciones del pipeline
    args = parse_args()
    logger.info(f"Iniciando el experimento con argumentos: {vars(args)}")

    config = load_config(args.config)
    if args.tuned_params:
        config = apply_tuned_params(config, args.tuned_params)
    seed = config.get("experiment", {}).get("random_seed", 42)
    set_seed(seed)

    # Filtrado opcional de modelos
    configured_models = [m["name"] for m in config.get("models", [])]
    if args.models:
        selected_models = [m for m in configured_models if m in args.models]
        logger.info(f"Modelos seleccionádos: {selected_models}")
    else:
        selected_models = configured_models
        logger.info(f"Se evaluarán todos los modelos")
    if args.dry_run:
        logger.warning(f"Entrenamiento en modo de pruebas rapido")

    # Carga y preparación de los datos
    train_df, test_df, rul_test_df = prepare_raw_data(config)
    if args.dry_run:
        train_df = train_df[train_df["unit_number"] <= 20].copy()
        test_df = test_df[test_df["unit_number"] <= 20].copy()
        rul_test_df = rul_test_df.iloc[:20].copy()
        logger.info(f"DRY-RUN reducido a {train_df['unit_number'].nunique()}")

    # Extraer las características
    train_enriched = extract_features(train_df, config)
    test_enriched = extract_features(test_df, config)

    # Identificar características numéricas que no me sirven
    exclude = {"unit_number", "time", "rul"}
    feature_cols = [col for col in train_enriched.columns if col not in exclude]
    logger.info(f"Total de características para modelado: {len(feature_cols)}")

    w_size = config.get("data", {}).get("window_size", 30)
    n_folds = 2 if args.dry_run else config.get("evaluation", {}).get("cv_folds", 10)

    # Selección de características por fold (una sola vez para todos los modelos)
    fold_feature_map, feature_stability = prepare_cv_feature_selection(
        train_enriched, feature_cols, config, n_folds=n_folds
    )

    # Pipeline global: scaler + selector ajustados con el 100% del train.
    # Compartido por TODOS los modelos (diseño pareado) y por el test set.
    X_train_full, y_train_full, X_test_final, global_features = prepare_global_datasets(
        train_enriched, test_enriched, feature_cols, config
    )

    rul_max = config.get("data", {}).get("rul_max", 125)
    y_test_official = np.minimum(rul_test_df["rul"].values, rul_max).astype(np.float32)

    init_mlflow(config)

    #----------------------------------------------------------------------------------------
    # Bucle de evaluación por modelo
    models_dict = {m["name"]: m.get("params", {}) for m in config.get("models", [])}
    cv_results_all = {}
    test_results_all = {}

    for m_name in selected_models:
        if m_name == "associative_memory":
            logger.warning(
                "Modelo 'associative_memory' omitido: pendiente de implementación (Issue #6). "
                "El experimento corre con los modelos restantes."
            )
            continue

        m_params = models_dict.get(m_name, {})

        # Validación cruzada
        cv_res = evaluate_model_cv(
            model_name=m_name,
            model_params=m_params,
            train_enriched_df=train_enriched,
            fold_feature_map=fold_feature_map,
            config=config,
            dry_run=args.dry_run,
        )
        cv_results_all[m_name] = cv_res

        # Reentrenamiento final desde cero con los MISMOS params sobre el 100% del train.
        # Los modelos de CV (last_model/last_scaler/last_features) quedan sólo para trazabilidad.
        final_model = build_model(m_name, m_params, (w_size, len(global_features)))
        fit_kwargs = {}
        if isinstance(final_model, PyTorchModel) and args.dry_run:
            fit_kwargs = {"epochs": 2}
        final_model.fit(X_train_full, y_train_full, **fit_kwargs)

        test_res = evaluate_on_test_set(
            model=final_model,
            X_test=X_test_final,
            y_test=y_test_official,
        )
        test_results_all[m_name] = test_res

        # Registrar Run en MLflow
        log_model_run_to_MLflow(
            model_name=m_name,
            cv_res=cv_res,
            test_res=test_res,
            config=config,
            n_features_selected=len(global_features),
        )

    # Inferencia estadística multimodelo
    log_omnibus_comparission(
        cv_results_all,
        test_results_all,
        config,
        fold_feature_map=fold_feature_map,
        feature_counts=feature_stability,
    )

if __name__ == "__main__":
    main()
