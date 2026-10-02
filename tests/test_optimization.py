"""TDD test suite for Issue #11 — Optuna hyperparameter optimization.

Phase RED: these tests define the contract for src/models/optimization.py
(module does not exist yet). Run:

    uv run pytest tests/test_optimization.py -q

Contract under test:
    - suggest_from_space(trial, search_space)     -> YAML space -> suggest_* calls
    - validate_optimization_config(opt_cfg)       -> ValueError on invalid config
    - optimize_hyperparameters(...)               -> best/baseline results dict
    - export_best_params(results, out_path)       -> tuned JSON for train_eval
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.model_selection import GroupKFold

from src.models.optimization import (
    export_best_params,
    optimize_hyperparameters,
    suggest_from_space,
    validate_optimization_config,
)


# ---------------------------------------------------------------------------
# Fixtures and test doubles
# ---------------------------------------------------------------------------

@pytest.fixture
def dataset() -> pd.DataFrame:
    """Tiny synthetic enriched dataset: 4 engines x 6 cycles, 5 features."""
    rows = []
    for unit in range(1, 5):
        for t in range(1, 7):
            rows.append(
                {
                    "unit_number": unit,
                    "time": t,
                    "rul": float(11 - t),
                    **{f"f{i}": float(unit * 10 + t + i) for i in range(1, 6)},
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def feature_cols() -> list[str]:
    return [f"f{i}" for i in range(1, 6)]


class RecordingTrial:
    """Doubles optuna.Trial just to record suggest_* calls."""

    def __init__(self):
        self.calls: list[tuple] = []

    def suggest_int(self, name, low, high, step=1, **kwargs):
        self.calls.append(("int", name, low, high, step))
        return low

    def suggest_float(self, name, low, high, **kwargs):
        self.calls.append(("float", name, low, high, kwargs.get("log", False)))
        return low

    def suggest_categorical(self, name, choices, **kwargs):
        self.calls.append(("categorical", name, tuple(choices)))
        return choices[0]


class StubRegressor:
    """Constant-predictor model so objective math is deterministic.

    RMSE(y, bias) is minimized when bias is closest to mean(y),
    which lets tests force either 'tuned beats baseline' or the reverse.
    """

    def __init__(self, params: dict):
        self.params = params

    def fit(self, X, y):
        self._bias = float(self.params.get("bias", 0.0))
        return self

    def predict(self, X):
        return np.full(len(X), self._bias, dtype=np.float64)


def stub_factory(params: dict) -> StubRegressor:
    return StubRegressor(params)


def make_config(opt_overrides: dict | None = None, fs_enabled: bool = False) -> dict:
    optimization = {
        "enabled": True,
        "n_trials": 20,
        "cv_folds": 2,
        "objective": "rmse",
        "direction": "minimize",
        "seed": 42,
        "models": {"stub_model": {"bias": {"type": "float", "low": 0.0, "high": 10.0}}},
    }
    if opt_overrides:
        optimization.update(opt_overrides)
    return {
        "optimization": optimization,
        "feature_selection": {
            "enabled": fs_enabled,
            "method": "hybrid",
            "mutual_info": {"percentile": 60, "random_state": 42},
            "rf_importance": {
                "n_features_to_select": 3,
                "n_estimators": 10,
                "max_depth": 4,
                "random_state": 42,
                "n_jobs": 1,
            },
        },
    }


BASE_PARAMS = {"bias": 5.5}          # mean(rul) -> strong baseline for the stub
SPACE_BIAS_FAR = {"bias": {"type": "float", "low": 50.0, "high": 90.0}}
SPACE_BIAS_NEAR = {"bias": {"type": "float", "low": 0.0, "high": 11.0}}


# ---------------------------------------------------------------------------
# 1. YAML search space -> suggest_* mapping (criteria 1 & 2)
# ---------------------------------------------------------------------------

class TestSuggestFromSpace:
    def test_int_spec_maps_to_suggest_int(self):
        trial = RecordingTrial()
        suggest_from_space(trial, {"n": {"type": "int", "low": 100, "high": 800, "step": 100}})
        assert trial.calls == [("int", "n", 100, 800, 100)]

    def test_int_spec_without_step_defaults_to_step_one(self):
        trial = RecordingTrial()
        suggest_from_space(trial, {"n": {"type": "int", "low": 5, "high": 30}})
        assert trial.calls == [("int", "n", 5, 30, 1)]

    def test_float_spec_maps_to_suggest_float(self):
        trial = RecordingTrial()
        suggest_from_space(trial, {"lr": {"type": "float", "low": 0.01, "high": 0.3}})
        assert trial.calls == [("float", "lr", 0.01, 0.3, False)]

    def test_float_log_true_maps_to_log_uniform(self):
        trial = RecordingTrial()
        suggest_from_space(trial, {"lr": {"type": "float", "low": 0.001, "high": 0.1, "log": True}})
        assert trial.calls == [("float", "lr", 0.001, 0.1, True)]

    def test_categorical_spec_maps_to_suggest_categorical(self):
        trial = RecordingTrial()
        suggest_from_space(
            trial, {"c": {"type": "categorical", "choices": ["sqrt", "log2"]}}
        )
        assert trial.calls == [("categorical", "c", ("sqrt", "log2"))]

    def test_multiple_params_are_all_suggested(self):
        trial = RecordingTrial()
        suggest_from_space(
            trial,
            {
                "n": {"type": "int", "low": 1, "high": 10},
                "lr": {"type": "float", "low": 0.1, "high": 0.9},
                "m": {"type": "categorical", "choices": ["a", "b"]},
            },
        )
        assert len(trial.calls) == 3

    def test_unknown_type_raises_informative_error(self):
        trial = RecordingTrial()
        with pytest.raises(ValueError, match="no soportado"):
            suggest_from_space(trial, {"n": {"type": "string", "low": 1, "high": 2}})

    def test_missing_type_raises_informative_error(self):
        trial = RecordingTrial()
        with pytest.raises(ValueError, match="tipo"):
            suggest_from_space(trial, {"n": {"low": 1, "high": 2}})

    def test_low_greater_than_high_raises(self):
        trial = RecordingTrial()
        with pytest.raises(ValueError, match="no puede ser mayor"):
            suggest_from_space(trial, {"n": {"type": "int", "low": 50, "high": 10}})

    def test_missing_bounds_for_int_raises(self):
        trial = RecordingTrial()
        with pytest.raises(ValueError):
            suggest_from_space(trial, {"n": {"type": "int"}})

    def test_empty_choices_for_categorical_raises(self):
        trial = RecordingTrial()
        with pytest.raises(ValueError, match="vacío"):
            suggest_from_space(trial, {"c": {"type": "categorical", "choices": []}})


# ---------------------------------------------------------------------------
# 2. Optimization config validation (criteria 2 & 6)
# ---------------------------------------------------------------------------

class TestValidateOptimizationConfig:
    def test_valid_config_passes_through(self):
        cfg = make_config()["optimization"]
        assert validate_optimization_config(cfg) == cfg

    def test_n_trials_below_20_raises(self):
        cfg = make_config({"n_trials": 19})["optimization"]
        with pytest.raises(ValueError, match="n_trials"):
            validate_optimization_config(cfg)

    def test_invalid_objective_raises(self):
        cfg = make_config({"objective": "mae"})["optimization"]
        with pytest.raises(ValueError, match="objective"):
            validate_optimization_config(cfg)

    def test_valid_objectives_accepted(self):
        for objective in ("rmse", "nasa_score"):
            cfg = make_config({"objective": objective})["optimization"]
            assert validate_optimization_config(cfg)["objective"] == objective

    def test_invalid_direction_raises(self):
        cfg = make_config({"direction": "sideways"})["optimization"]
        with pytest.raises(ValueError, match="direction"):
            validate_optimization_config(cfg)

    def test_missing_models_block_raises(self):
        cfg = make_config()
        cfg["optimization"].pop("models")
        with pytest.raises(ValueError, match="models"):
            validate_optimization_config(cfg["optimization"])

    def test_empty_models_block_raises(self):
        cfg = make_config({"models": {}})["optimization"]
        with pytest.raises(ValueError, match="models"):
            validate_optimization_config(cfg)

    def test_cv_folds_below_2_raises(self):
        cfg = make_config({"cv_folds": 1})["optimization"]
        with pytest.raises(ValueError, match="cv_folds"):
            validate_optimization_config(cfg)


# ---------------------------------------------------------------------------
# 3. optimize_hyperparameters end-to-end with stub model (criteria 1, 5, 6)
# ---------------------------------------------------------------------------

class TestOptimizeHyperparameters:
    def test_returns_result_dict_with_required_keys(self, dataset, feature_cols):
        result = optimize_hyperparameters(
            model_name="stub_model",
            base_params=BASE_PARAMS,
            search_space=SPACE_BIAS_NEAR,
            dataset=dataset,
            feature_cols=feature_cols,
            config=make_config(),
            model_factory=stub_factory,
        )
        assert set(result) >= {"best_params", "best_value", "baseline_value", "objective"}
        assert isinstance(result["best_params"], dict)
        assert np.isfinite(result["best_value"])
        assert np.isfinite(result["baseline_value"])

    def test_best_params_come_from_search_space(self, dataset, feature_cols):
        result = optimize_hyperparameters(
            model_name="stub_model",
            base_params=BASE_PARAMS,
            search_space=SPACE_BIAS_NEAR,
            dataset=dataset,
            feature_cols=feature_cols,
            config=make_config(),
            model_factory=stub_factory,
        )
        assert set(result["best_params"]) == set(SPACE_BIAS_NEAR)
        assert SPACE_BIAS_NEAR["bias"]["low"] <= result["best_params"]["bias"] <= SPACE_BIAS_NEAR["bias"]["high"]

    def test_dry_run_caps_trials_at_three_and_folds_at_two(self, dataset, feature_cols):
        result = optimize_hyperparameters(
            model_name="stub_model",
            base_params=BASE_PARAMS,
            search_space=SPACE_BIAS_NEAR,
            dataset=dataset,
            feature_cols=feature_cols,
            config=make_config({"n_trials": 50, "cv_folds": 5}),
            model_factory=stub_factory,
            dry_run=True,
        )
        assert result["n_trials_completed"] <= 3

    def test_same_seed_produces_same_best_params(self, dataset, feature_cols):
        runs = [
            optimize_hyperparameters(
                model_name="stub_model",
                base_params=BASE_PARAMS,
                search_space=SPACE_BIAS_NEAR,
                dataset=dataset,
                feature_cols=feature_cols,
                config=make_config({"seed": 42, "n_trials": 5}),
                model_factory=stub_factory,
            )
            for _ in range(2)
        ]
        assert runs[0]["best_params"] == runs[1]["best_params"]
        assert runs[0]["best_value"] == runs[1]["best_value"]

    def test_baseline_not_beaten_logs_warning(self, dataset, feature_cols, caplog):
        # baseline bias 5.5 is near mean(rul); every trial lands at 50..90 -> baseline wins
        with caplog.at_level(logging.WARNING):
            optimize_hyperparameters(
                model_name="stub_model",
                base_params=BASE_PARAMS,
                search_space=SPACE_BIAS_FAR,
                dataset=dataset,
                feature_cols=feature_cols,
                config=make_config(),
                model_factory=stub_factory,
                dry_run=True,
            )
        assert "baseline" in caplog.text.lower()

    def test_tuned_beating_baseline_does_not_warn(self, dataset, feature_cols, caplog):
        # baseline bias 999 is terrible; trials land near mean(rul) -> tuned wins
        with caplog.at_level(logging.WARNING):
            optimize_hyperparameters(
                model_name="stub_model",
                base_params={"bias": 999.0},
                search_space=SPACE_BIAS_NEAR,
                dataset=dataset,
                feature_cols=feature_cols,
                config=make_config(),
                model_factory=stub_factory,
                dry_run=True,
            )
        assert "baseline" not in caplog.text.lower()

    def test_objective_nasa_score_is_recorded(self, dataset, feature_cols):
        result = optimize_hyperparameters(
            model_name="stub_model",
            base_params=BASE_PARAMS,
            search_space=SPACE_BIAS_NEAR,
            dataset=dataset,
            feature_cols=feature_cols,
            config=make_config({"objective": "nasa_score"}),
            model_factory=stub_factory,
            dry_run=True,
        )
        assert result["objective"] == "nasa_score"

    def test_missing_model_factory_raises_informative_error(self, dataset, feature_cols):
        with pytest.raises(ValueError, match="model_factory"):
            optimize_hyperparameters(
                model_name="stub_model",
                base_params=BASE_PARAMS,
                search_space=SPACE_BIAS_NEAR,
                dataset=dataset,
                feature_cols=feature_cols,
                config=make_config(),
                model_factory=None,
            )


# ---------------------------------------------------------------------------
# 4. Anti-leakage of the internal CV (criteria 3 & 8)
# ---------------------------------------------------------------------------

class TestInternalCVAntiLeakage:
    def test_selector_is_fit_only_on_train_fold_rows(self, dataset, feature_cols, monkeypatch):
        """create_feature_selector must be fit ONLY on train_fold rows of each
        GroupKFold split (never on val_fold nor on the full dataset)."""
        fit_row_counts: list[int] = []

        class SpySelector:
            def fit(self, X, y):
                fit_row_counts.append(len(X))
                return self

            def transform(self, X):
                return X

            def fit_transform(self, X, y):
                return self.fit(X, y).transform(X)

            def get_support(self, indices=False):
                return np.ones(X.shape[1], dtype=bool) if False else np.ones(1, dtype=bool)

        import src.models.optimization as opt_module

        monkeypatch.setattr(
            opt_module, "create_feature_selector", lambda cfg: SpySelector(), raising=True
        )

        optimize_hyperparameters(
            model_name="stub_model",
            base_params=BASE_PARAMS,
            search_space=SPACE_BIAS_NEAR,
            dataset=dataset,
            feature_cols=feature_cols,
            config=make_config(fs_enabled=True),
            model_factory=stub_factory,
            dry_run=True,  # <= 3 trials, 2 folds
        )

        total_rows = len(dataset)
        n_folds = 2

        # A fit never saw the full dataset nor a val_fold in disguise
        assert fit_row_counts, "selector was never fit -> selection is not inside the CV"
        assert all(count < total_rows for count in fit_row_counts)

        # Expected train sizes: each fold trains on 2 of 4 engines (6 rows each)
        gkf = GroupKFold(n_splits=n_folds)
        groups = dataset["unit_number"].values
        expected_train = sorted(
            len(dataset.iloc[train_idx]) for _, train_idx in gkf.split(dataset, groups=groups)
        )

        # Baseline + every trial performs exactly one fit per fold
        assert len(fit_row_counts) % n_folds == 0
        observed_train = sorted(fit_row_counts)
        repeats = len(observed_train) // n_folds
        assert observed_train == sorted(expected_train * repeats)


# ---------------------------------------------------------------------------
# 5. JSON export for train_eval --tuned-params (criteria 4 & 8)
# ---------------------------------------------------------------------------

class TestExportBestParams:
    def test_writes_valid_json_with_model_structure(self, tmp_path):
        out = tmp_path / "tuned" / "config_FD001_tuned.json"
        results = {"random_forest": {"n_estimators": 500, "max_depth": 18}}
        path = export_best_params(results, out)

        assert path.exists()
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded == {"random_forest": {"n_estimators": 500, "max_depth": 18}}

    def test_returns_path_object(self, tmp_path):
        path = export_best_params({"xgboost": {"max_depth": 6}}, tmp_path / "t.json")
        assert isinstance(path, Path)

    def test_creates_missing_parent_directories(self, tmp_path):
        out = tmp_path / "a" / "b" / "c" / "tuned.json"
        export_best_params({"m": {"p": 1}}, out)
        assert out.exists()

    def test_overwrites_existing_file(self, tmp_path):
        out = tmp_path / "tuned.json"
        export_best_params({"m": {"p": 1}}, out)
        export_best_params({"m": {"p": 2}}, out)
        assert json.loads(out.read_text())["m"]["p"] == 2

    def test_multiple_models_round_trip(self, tmp_path):
        results = {
            "random_forest": {"n_estimators": 300, "max_depth": 12},
            "xgboost": {"learning_rate": 0.05, "max_depth": 8},
        }
        path = export_best_params(results, tmp_path / "t.json")
        assert json.loads(path.read_text()) == results


# ---------------------------------------------------------------------------
# 6. YAML integration: configs ship a valid optimization block (criteria 2 & 7)
# ---------------------------------------------------------------------------

class TestConfigsOptimizationBlock:
    @pytest.mark.parametrize(
        "config_name", ["config_FD001.yaml", "config_FD002.yaml", "config_FD003.yaml", "config_FD004.yaml"]
    )
    def test_config_has_valid_optimization_block(self, config_name):
        with open(Path("configs") / config_name, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

        assert "optimization" in raw, f"{config_name} must define an optimization block"
        opt = raw["optimization"]
        validate_optimization_config(opt)  # n_trials>=20, objective, direction, models...

    def test_optimization_block_exposes_standard_keys(self):
        with open(Path("configs") / "config_FD001.yaml", encoding="utf-8") as fh:
            opt = yaml.safe_load(fh)["optimization"]
        assert set(opt) >= {"enabled", "n_trials", "cv_folds", "objective", "direction", "seed", "models"}


# ---------------------------------------------------------------------------
# 7. Reproducibilidad por trial: set_seed antes de cada evaluación (Issue #11, AC §97)
# ---------------------------------------------------------------------------

class TestSeedPorTrial:
    def test_set_seed_se_invoca_una_vez_por_trial_con_la_semilla_del_config(
        self, dataset, feature_cols, monkeypatch
    ):
        """Cada trial y el baseline llaman a set_seed con la seed del config."""
        import src.models.optimization as opt_module

        llamadas: list[int] = []
        monkeypatch.setattr(opt_module, "set_seed", lambda s: llamadas.append(s))

        n_trials = 4
        optimize_hyperparameters(
            model_name="stub_model",
            base_params=BASE_PARAMS,
            search_space=SPACE_BIAS_NEAR,
            dataset=dataset,
            feature_cols=feature_cols,
            config=make_config({"seed": 7, "n_trials": n_trials}),
            model_factory=stub_factory,
        )

        # n_trials semillas por trial + 1 semilla para el baseline
        assert llamadas == [7] * (n_trials + 1)

    def test_set_seed_se_ejecuta_antes_de_evaluar_cada_trial(
        self, dataset, feature_cols, monkeypatch
    ):
        """El orden dentro de la corrida es: set_seed y recién después los fits del CV."""
        import src.models.optimization as opt_module

        eventos: list[str] = []
        monkeypatch.setattr(opt_module, "set_seed", lambda s: eventos.append("seed"))

        def factory_spy(params: dict):
            eventos.append("fit")
            return stub_factory(params)

        n_trials = 3
        cv_folds = 2
        optimize_hyperparameters(
            model_name="stub_model",
            base_params=BASE_PARAMS,
            search_space=SPACE_BIAS_NEAR,
            dataset=dataset,
            feature_cols=feature_cols,
            config=make_config({"n_trials": n_trials, "cv_folds": cv_folds}),
            model_factory=factory_spy,
        )

        indices_seed = [i for i, ev in enumerate(eventos) if ev == "seed"]
        # nada se evalúa sin sembrar primero
        assert eventos[0] == "seed"
        assert len(indices_seed) == n_trials + 1  # trials + baseline
        # entre semillas consecutivas: exactamente un evaluate de cv_folds fits
        for a, b in zip(indices_seed, indices_seed[1:]):
            assert eventos[a + 1 : b] == ["fit"] * cv_folds
        # el evaluate final (baseline) también va precedido de su semilla
        assert eventos[indices_seed[-1] + 1 :] == ["fit"] * cv_folds

    def test_misma_semilla_dos_corridas_mismo_resultado(
        self, dataset, feature_cols
    ):
        """Dos corridas con la misma config y seed dan exactamente el mismo
        resultado aunque el modelo consuma el RNG global (AC §97)."""

        class StubRuidoso(StubRegressor):
            def predict(self, X):
                return super().predict(X) + np.random.normal(0.0, 0.05, len(X))

        def factory_ruidoso(params: dict) -> StubRuidoso:
            return StubRuidoso(params)

        corridas = [
            optimize_hyperparameters(
                model_name="stub_model",
                base_params=BASE_PARAMS,
                search_space=SPACE_BIAS_NEAR,
                dataset=dataset,
                feature_cols=feature_cols,
                config=make_config({"seed": 123, "n_trials": 5}),
                model_factory=factory_ruidoso,
            )
            for _ in range(2)
        ]

        assert corridas[0]["best_value"] == corridas[1]["best_value"]
        assert corridas[0]["baseline_value"] == corridas[1]["baseline_value"]
        assert corridas[0]["best_params"] == corridas[1]["best_params"]
