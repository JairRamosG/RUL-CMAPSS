"""Tests del CLI scripts/optimize.py (Issue #11)."""

import json
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import optuna
import pandas as pd
import pytest

import scripts.optimize as cli


# ---------------------------------------------------------------------------
# Fixtures and test doubles
# ---------------------------------------------------------------------------

RF_SPACE = {"n_estimators": {"type": "int", "low": 100, "high": 500}}
LGBM_SPACE = {"learning_rate": {"type": "float", "low": 0.01, "high": 0.3}}


def make_cli_config(opt_models: dict | None = None) -> dict:
    """Config stub de main(): subset FD001 con bloque optimization valido."""
    if opt_models is None:
        opt_models = {"random_forest": RF_SPACE}
    return {
        "subset": "FD001",
        "optimization": {
            "enabled": True,
            "n_trials": 20,
            "cv_folds": 2,
            "objective": "rmse",
            "direction": "minimize",
            "seed": 42,
            "models": opt_models,
        },
        "models": [{"name": "random_forest", "params": {"n_estimators": 100}}],
    }


def make_raw_data() -> pd.DataFrame:
    """Data stub de prepare_raw_data: 3 motores x 3 ciclos, columnas base."""
    rows = [
        {"unit_number": unit, "time": t, "rul": float(11 - t)}
        for unit in (1, 2, 3)
        for t in (1, 2, 3)
    ]
    return pd.DataFrame(rows)


def make_enriched_data() -> pd.DataFrame:
    """Data stub de extract_features: la base mas una feature f1."""
    df = make_raw_data()
    df["f1"] = (df["unit_number"] + df["time"]).astype(float)
    return df


class RecordingMlflow:
    """Doble de mlflow que solo registra las llamadas (sin servidor real)."""

    def __init__(self):
        self.runs: list[dict] = []
        self.tags: list[dict] = []
        self.params: list[dict] = []
        self.metrics: list[dict] = []

    @contextmanager
    def start_run(self, run_name=None, nested=False, **kwargs):
        self.runs.append({"run_name": run_name, "nested": nested})
        yield

    def set_tags(self, tags):
        self.tags.append(dict(tags))

    def log_params(self, params):
        self.params.append(dict(params))

    def log_metrics(self, metrics):
        self.metrics.append(dict(metrics))

    def install(self, monkeypatch):
        monkeypatch.setattr(cli.mlflow, "start_run", self.start_run)
        monkeypatch.setattr(cli.mlflow, "set_tags", self.set_tags)
        monkeypatch.setattr(cli.mlflow, "log_params", self.log_params)
        monkeypatch.setattr(cli.mlflow, "log_metrics", self.log_metrics)


class _DummyRun:
    """Context manager de reemplazo para mlflow.start_run (sin run real)."""

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


@pytest.fixture
def tmp_storage(tmp_path, monkeypatch):
    """Storage sqlite Optuna en tmp_path; parchea cli.OPTUNA_STORAGE."""
    uri = f"sqlite:///{tmp_path}/optuna.db"
    monkeypatch.setattr(cli, "OPTUNA_STORAGE", uri)
    return uri


@pytest.fixture
def main_env(monkeypatch):
    """Stub de todas las dependencias pesadas de main() + grabadores.

    start_run devuelve un context manager dummy y el logging lateral de
    mlflow (set_tags/log_params/log_metrics) queda como no-op: los tests
    no deben tocar ningun servidor ni almacen real de MLflow.
    """
    env = SimpleNamespace(
        config=make_cli_config(),
        config_path=None,
        seed_calls=[],
        optimize_calls=[],
        existing_calls=[],
        logged_calls=[],
        build_calls=[],
        init_mlflow_calls=0,
    )

    def fake_load_config(config_path):
        env.config_path = config_path
        return env.config

    def fake_prepare_raw_data(config):
        return (make_raw_data(), make_raw_data(), make_raw_data())

    def fake_extract_features(df, config):
        return make_enriched_data()

    def fake_build_model(model_name, params, input_shape):
        env.build_calls.append((model_name, params, input_shape))
        return object()

    def fake_set_seed(seed=None):
        env.seed_calls.append(seed)

    def fake_init_mlflow(config):
        env.init_mlflow_calls += 1

    def fake_optimize_hyperparameters(*args, **kwargs):
        env.optimize_calls.append(kwargs)
        return {
            "best_params": {"n_estimators": 10},
            "best_value": 1.0,
            "baseline_value": 2.0,
            "objective": "rmse",
            "n_trials_completed": 2,
        }

    def fake_existing_trials(study_name):
        env.existing_calls.append(study_name)
        return 0

    def fake_log_trials_to_mlflow(study_name, model_name, subset, objective):
        env.logged_calls.append((study_name, model_name, subset, objective))

    monkeypatch.setattr(cli, "load_config", fake_load_config)
    monkeypatch.setattr(cli, "prepare_raw_data", fake_prepare_raw_data)
    monkeypatch.setattr(cli, "extract_features", fake_extract_features)
    monkeypatch.setattr(cli, "build_model", fake_build_model)
    monkeypatch.setattr(cli, "set_seed", fake_set_seed)
    monkeypatch.setattr(cli, "init_mlflow", fake_init_mlflow)
    monkeypatch.setattr(cli, "optimize_hyperparameters", fake_optimize_hyperparameters)
    monkeypatch.setattr(cli, "_existing_trials", fake_existing_trials)
    monkeypatch.setattr(cli, "_log_trials_to_mlflow", fake_log_trials_to_mlflow)
    monkeypatch.setattr(cli.mlflow, "start_run", lambda *args, **kwargs: _DummyRun())
    monkeypatch.setattr(cli.mlflow, "set_tags", lambda tags: None)
    monkeypatch.setattr(cli.mlflow, "log_params", lambda params: None)
    monkeypatch.setattr(cli.mlflow, "log_metrics", lambda metrics: None)
    monkeypatch.setattr(sys, "argv", ["optimize.py"])

    return env


# ---------------------------------------------------------------------------
# 1. parse_args: banderas del CLI
# ---------------------------------------------------------------------------

class TestParseArgs:
    def test_defaults(self, monkeypatch):
        """Sin banderas: defaults del parser."""
        monkeypatch.setattr(sys, "argv", ["optimize.py"])
        args = cli.parse_args()

        assert args.config == "configs/config_FD001.yaml"
        assert args.models is None
        assert args.dry_run is False
        assert args.tuned_params_out is None

    def test_config_flag(self, monkeypatch):
        """--config pisa el default."""
        monkeypatch.setattr(
            sys, "argv", ["optimize.py", "--config", "configs/config_FD004.yaml"]
        )
        args = cli.parse_args()

        assert args.config == "configs/config_FD004.yaml"

    def test_models_subset(self, monkeypatch):
        """--models acepta un subset de uno o mas modelos."""
        monkeypatch.setattr(
            sys, "argv", ["optimize.py", "--models", "random_forest", "xgboost"]
        )
        args = cli.parse_args()

        assert args.models == ["random_forest", "xgboost"]

    def test_dry_run_flag(self, monkeypatch):
        """--dry-run activa el modo de pruebas rapido."""
        monkeypatch.setattr(sys, "argv", ["optimize.py", "--dry-run"])
        args = cli.parse_args()

        assert args.dry_run is True

    def test_tuned_params_out_flag(self, monkeypatch):
        """--tuned-params-out fija la ruta del JSON de mejores hiperparametros."""
        monkeypatch.setattr(
            sys, "argv", ["optimize.py", "--tuned-params-out", "tuned/x.json"]
        )
        args = cli.parse_args()

        assert args.tuned_params_out == "tuned/x.json"


# ---------------------------------------------------------------------------
# 2. _existing_trials: conteo reanudable de trials en el storage
# ---------------------------------------------------------------------------

class TestExistingTrials:
    def test_estudio_inexistente_devuelve_0(self, tmp_storage):
        """Estudio que nunca corrio: 0 trials y sin excepcion."""
        assert cli._existing_trials("FD001_fake") == 0

    def test_cuenta_trials_registrados(self, tmp_storage):
        """Cuenta los trials ya registrados en el storage."""
        study = optuna.create_study(study_name="s1", storage=tmp_storage)
        study.optimize(lambda trial: 1.0, n_trials=3)

        assert cli._existing_trials("s1") == 3


# ---------------------------------------------------------------------------
# 3. _log_trials_to_mlflow: cada trial como run anidado (fakes, sin servidor)
# ---------------------------------------------------------------------------

class TestLogTrialsToMlflow:
    def test_registra_trials_como_runs_anidados(self, tmp_storage, monkeypatch):
        """Cada trial queda como run trial_N con tags, params y metrics."""
        study = optuna.create_study(study_name="FD001_rf", storage=tmp_storage)
        study.optimize(
            lambda trial: trial.suggest_float("alpha", 0.0, 1.0), n_trials=2
        )
        trials = list(study.trials)

        recorder = RecordingMlflow()
        recorder.install(monkeypatch)

        cli._log_trials_to_mlflow("FD001_rf", "random_forest", "FD001", "rmse")

        assert [run["run_name"] for run in recorder.runs] == ["trial_0", "trial_1"]
        assert all(run["nested"] is True for run in recorder.runs)

        assert len(recorder.tags) == 2
        for tags in recorder.tags:
            assert tags["stage"] == "optimization"
            assert tags["model_name"] == "random_forest"
            assert tags["subset"] == "FD001"

        assert recorder.params == [trial.params for trial in trials]
        assert recorder.metrics == [
            {"rmse": float(trial.value)} for trial in trials
        ]


# ---------------------------------------------------------------------------
# 4. main(): contrato del CLI completo (dependencias todas stubbed)
# ---------------------------------------------------------------------------

class TestMainCli:
    def test_main_dry_run_exporta_tuned_json(self, tmp_path, monkeypatch, main_env):
        """--dry-run exporta el tuned JSON y optimiza con n_trials restantes."""
        out = tmp_path / "tuned.json"
        monkeypatch.setattr(
            sys,
            "argv",
            ["optimize.py", "--dry-run", "--tuned-params-out", str(out)],
        )

        cli.main()

        assert out.exists()
        assert json.loads(out.read_text(encoding="utf-8")) == {
            "random_forest": {"n_estimators": 10}
        }

        assert len(main_env.optimize_calls) == 1
        call = main_env.optimize_calls[0]
        assert call["dry_run"] is True
        assert call["storage"] == cli.OPTUNA_STORAGE
        assert call["study_name"] == "FD001_random_forest"
        # dry-run: main capa el target a 3 trials; existing stubbed a 0 -> restantes 3
        assert call["config"]["optimization"]["n_trials"] == 3
        assert main_env.seed_calls == [42]

    def test_main_enabled_false_no_hace_nada(self, tmp_path, monkeypatch, main_env):
        """optimization.enabled=false: no optimiza ni crea el JSON de salida."""
        main_env.config["optimization"]["enabled"] = False
        out = tmp_path / "tuned.json"
        monkeypatch.setattr(
            sys, "argv", ["optimize.py", "--tuned-params-out", str(out)]
        )

        cli.main()

        assert main_env.optimize_calls == []
        assert not out.exists()

    def test_main_filtro_models(self, tmp_path, monkeypatch, main_env):
        """--models lightgbm: solo se optimiza lightgbm del espacio."""
        main_env.config = make_cli_config(
            {"random_forest": RF_SPACE, "lightgbm": LGBM_SPACE}
        )
        out = tmp_path / "tuned.json"
        monkeypatch.setattr(
            sys,
            "argv",
            ["optimize.py", "--models", "lightgbm", "--tuned-params-out", str(out)],
        )

        cli.main()

        assert [call["model_name"] for call in main_env.optimize_calls] == [
            "lightgbm"
        ]

    def test_main_sin_flag_usa_ruta_default(self, tmp_path, monkeypatch, main_env):
        """Sin --tuned-params-out exporta en tuned/config_<subset>_tuned.json (ruta default)."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["optimize.py"])

        cli.main()

        out = tmp_path / "tuned" / "config_FD001_tuned.json"
        assert out.exists()
        assert json.loads(out.read_text(encoding="utf-8")) == {
            "random_forest": {"n_estimators": 10}
        }
