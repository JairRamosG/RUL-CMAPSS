"""
Módulo optimizador de Hiperparámetros con Optuna

    - suggest_from_space(trial, search_space)   YAML space -> optuna suggest_* calls
    - validate_optimization_config(opt_cfg)     ValueError on invalid optimization block
    - optimize_hyperparameters(...)             TPE tuning over leakage-free GroupKFold CV
    - export_best_params(results, out_path)     tuned JSON for train_eval --tuned-params
"""

