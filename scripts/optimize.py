"""
CLI de la optimización de hiperparámetros con Optuna

Uso:
    uv run python scripts/optimize.py --config configs/config_FD001.yaml --models random_forest
    uv run python scripts/optimize.py --config configs/config_FD001.yaml --models random_forest --dry-run --tuned-params-out tuned/smoke_rf.json
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

logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S"   
)

logger = logging.getLogger("optimize")

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
        "--tuned-params-out",
        type = str,
        default = None,
        help = "Ruta de salida para el JSON con los mejores hiperparámetros encontrados (default: tuned/config_<subset>_tuned.json)"
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


def _log_trials_to_mlflow(study_name: str, model_name: str, subset: str, objective: str) -> None:
    """
    Registra cada trial del study como un run anidado bajo el run padre en el MLflow.
    """

    study = optuna.load_study(study_name = study_name, storage = OPTUNA_STORAGE)
    for trial in study.trials:
        with mlflow.start_run(run_name = f"trial_{trial.number}", nested = True):
            mlflow.set_tags({
                "model_name" : model_name,
                "subset" : subset,
                "stage": "optimization",
                "trial_state": str(trial.state)
            })
            if trial.params:
                mlflow.log_params(trial.params)
            if trial.value is not None:
                mlflow.log_metrics({f"{objective}": float(trial.value)})

def main() -> None:
    """
    Función principal para lanzar la búsqueda de los parametros con OPtuna y registrarlos en MLflow
    """
    args = parse_args()
    config = load_config(args.config)
    subset = config["subset"]
    opt_cfg = config.get("optimization")

    if not opt_cfg or not opt_cfg.get("enabled"):
        logger.info(f"optimization.enabled = false en {args.config}: nada que optimizar.")
        return

    validate_optimization_config(opt_cfg)
    set_seed(int(opt_cfg.get("seed", 42)))

    # carga de datos + feature engineering como el pipeline de entrenamiento
    train_df, _, _ = prepare_raw_data(config)
    if args.dry_run:
        train_df = train_df[train_df["unit_number"] <= 20]

    train_enriched = extract_features(train_df, config)
    feature_cols = [c for c in train_enriched.columns if c not in {"unit_number", "time", "rul"}]

    # solo se optimizan los modelos definidos en el bloque optimization.models
    opt_space = opt_cfg.get("models", {})
    selected = [m for m in opt_space if args.models is None or m in args.models]

    if args.models:
        skipped = sorted(set(args.models) - set(opt_space))
        if skipped:
            logger.warning(f"Modelos omitidos si bloque optimization.models: {skipped}")

    if not selected:
        logger.warning(f"Ningún modelo valido para optimizar con los filtros proporcionados")
        return

    baseline_params = {m["name"]: m.get("params", {}) for m in config.get("models", [])}
    input_shape = (config.get("data", {}).get("window_size", 30), len(feature_cols))
    out_path = Path(args.tuned_params_out) if args.tuned_params_out else Path(f"tuned/config_{subset}_tuned.json")

    init_mlflow(config)
    tuned_results: dict[str, dict] = {}

    for model_name in selected:
        study_name = f"{subset}_{model_name}"
        done = _existing_trials(study_name)

        target_trials = 3 if args.dry_run else int(opt_cfg.get("n_trials", 20))
        remaining = max(0, target_trials - done)

        if remaining == 0:
            logger.info(f"Ya existen {done} trials para {model_name} en {study_name}, no se ejecuta optimización")
            continue

        cfg_run = {**config, "optimization": {**opt_cfg, "n_trials": remaining}}

        def model_factory(params : dict, _name: str = model_name):
            return build_model(_name, params, input_shape)

        logger.info(
            "Optimizando %s: ejecutando %s trials faltantes (de %s ya completados) | dry_run=%s",
            model_name, remaining, done, args.dry_run,
        )

        with mlflow.start_run(run_name=model_name):
            mlflow.set_tags(
                {"model_name": model_name, "subset": subset, "stage": "optimization"}
            )
            
            results = optimize_hyperparameters(
                model_name=model_name,
                base_params=baseline_params.get(model_name, {}),
                search_space=opt_space[model_name],
                dataset=train_enriched,
                feature_cols=feature_cols,
                config=cfg_run,
                model_factory=model_factory,
                dry_run=args.dry_run,
                storage=OPTUNA_STORAGE,
                study_name=study_name,
            )
            
            tuned_results[model_name] = results["best_params"]
            
            # Registrar en MLflow
            mlflow.log_params({f"tuned_{k}": str(v) for k, v in results["best_params"].items()})
            mlflow.log_metrics(
                {
                    "best_value": float(results["best_value"]),
                    "baseline_value": float(results["baseline_value"]),
                    "n_trials_completed": float(results["n_trials_completed"]),
                }
            )
            _log_trials_to_mlflow(study_name, model_name, subset, results["objective"])

    if tuned_results:
        path = export_best_params(tuned_results, out_path)
        logger.info("Mejores hiperparámetros exportados exitosamente a %s", path)


if __name__ == "__main__":
    main()





