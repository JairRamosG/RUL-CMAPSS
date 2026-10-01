"""Tests de apply_tuned_params y del flag --tuned-params de scripts/train_eval.py."""

import json
import sys

import pytest

import scripts.train_eval as te


def make_config() -> dict:
    """Config stub: random_forest y svr con params de ejemplo."""
    return {
        "subset": "FD001",
        "models": [
            {"name": "random_forest", "params": {"n_estimators": 100, "max_depth": 5, "min_samples_split": 2}},
            {"name": "svr", "params": {"C": 1.0}},
        ],
    }


class TestApplyTunedParams:
    """Tests de apply_tuned_params(config, tuned_path)."""

    def test_sobreescribe_params_del_modelo_afinado(self, tmp_path):
        """El modelo afinado pisa sus keys con los del JSON (los base-only se conservan)."""
        tuned = {"n_estimators": 500, "max_depth": 9}
        path = tmp_path / "tuned.json"
        path.write_text(json.dumps({"random_forest": tuned}), encoding="utf-8")

        config = te.apply_tuned_params(make_config(), path)

        expected = {**make_config()["models"][0]["params"], **tuned}
        assert config["models"][0]["params"] == expected
        by_name = {m["name"]: m.get("params", {}) for m in config["models"]}
        assert by_name["random_forest"] == expected

    def test_tuned_hace_merge_sobre_defaults(self, tmp_path):
        """Tuned pisa los keys solapados y conserva los keys base-only del YAML."""
        path = tmp_path / "tuned.json"
        path.write_text(json.dumps({"random_forest": {"n_estimators": 500}}), encoding="utf-8")

        config = te.apply_tuned_params(make_config(), path)

        params = config["models"][0]["params"]
        assert params["n_estimators"] == 500
        assert params["min_samples_split"] == 2
        assert params == {"n_estimators": 500, "max_depth": 5, "min_samples_split": 2}

    def test_modelo_sin_entrada_conserva_defaults(self, tmp_path):
        """Los modelos ausentes del JSON conservan los defaults del YAML."""
        path = tmp_path / "tuned.json"
        path.write_text(json.dumps({"random_forest": {"n_estimators": 500}}), encoding="utf-8")

        config = te.apply_tuned_params(make_config(), path)

        assert config["models"][1]["params"] == {"C": 1.0}

    def test_mutacion_in_place(self, tmp_path):
        """El mismo objeto config se muta: es lo que garantiza que el logger de MLflow lo vea."""
        tuned = {"n_estimators": 500, "max_depth": 9}
        path = tmp_path / "tuned.json"
        path.write_text(json.dumps({"random_forest": tuned}), encoding="utf-8")

        config = make_config()
        result = te.apply_tuned_params(config, path)

        assert result is config
        assert config["models"][0]["params"] == {
            "n_estimators": 500, "max_depth": 9, "min_samples_split": 2
        }

    def test_entrada_no_dict_lanza_value_error(self, tmp_path):
        """Una entrada que no es dict de parámetros levanta ValueError."""
        path = tmp_path / "tuned.json"
        path.write_text(json.dumps({"random_forest": [1, 2, 3]}), encoding="utf-8")

        with pytest.raises(ValueError, match="dict"):
            te.apply_tuned_params(make_config(), path)

    def test_config_sin_models_no_explode(self, tmp_path):
        """config sin clave models (o vacía) devuelve sin error."""
        path = tmp_path / "tuned.json"
        path.write_text(json.dumps({"random_forest": {"n_estimators": 1}}), encoding="utf-8")

        assert te.apply_tuned_params({"models": []}, path) == {"models": []}
        assert te.apply_tuned_params({}, path) == {}


class TestParseArgsTunedParams:
    """Tests del flag --tuned-params en parse_args()."""

    def test_sin_flag_es_none(self, monkeypatch):
        """Sin el flag, args.tuned_params es None (se usan los defaults del YAML)."""
        monkeypatch.setattr(sys, "argv", ["train_eval.py"])

        args = te.parse_args()

        assert args.tuned_params is None

    def test_con_flag(self, monkeypatch):
        """Con el flag, args.tuned_params es la ruta al JSON."""
        monkeypatch.setattr(sys, "argv", ["train_eval.py", "--tuned-params", "tuned/x.json"])

        args = te.parse_args()

        assert args.tuned_params == "tuned/x.json"
