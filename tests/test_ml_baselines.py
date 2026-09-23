"""Tests for ML baseline models: RandomForest, XGBoost, LightGBM."""
import numpy as np
from src.models.RFModel import RFModel
from src.models.XGBoostModel import XGBoostModel
from src.models.LightGBMModel import LightGBMModel
from src.models.sklearn_wrapper import SKLearnModel


# =============================================================================
# RFModel
# =============================================================================
class TestRFModel:
    """Tests for RFModel."""

    def test_default_params(self):
        model = RFModel()
        params = model.get_params()

        assert params["n_estimators"] == 100
        assert params["max_depth"] is None

    def test_custom_params(self):
        model = RFModel(n_estimators=50, max_depth=10)
        params = model.get_params()

        assert params["n_estimators"] == 50
        assert params["max_depth"] == 10

    def test_fit_returns_dict(self, X_2d, y):
        model = RFModel(n_estimators=10)
        result = model.fit(X_2d, y)

        assert isinstance(result, dict)
        assert "train_loss" in result
        assert "epoch_trained" in result

    def test_predict_returns_array(self, X_2d, y):
        model = RFModel(n_estimators=10)
        model.fit(X_2d, y)
        preds = model.predict(X_2d)

        assert isinstance(preds, np.ndarray)
        assert preds.shape == (100,)

    def test_fit_predict_3d(self, X_3d, y):
        model = RFModel(n_estimators=10)
        model.fit(X_3d, y)
        preds = model.predict(X_3d)

        assert preds.shape == (100,)

    def test_inherits_sklearn(self):
        model = RFModel()
        assert isinstance(model, SKLearnModel)


# =============================================================================
# XGBoostModel
# =============================================================================
class TestXGBoostModel:
    """Tests for XGBoostModel."""

    def test_default_params(self):
        model = XGBoostModel()
        params = model.get_params()

        assert params["n_estimators"] == 100
        assert params["max_depth"] == 6
        assert params["learning_rate"] == 0.1

    def test_custom_params(self):
        model = XGBoostModel(n_estimators=50, max_depth=3, learning_rate=0.05)
        params = model.get_params()

        assert params["n_estimators"] == 50
        assert params["max_depth"] == 3
        assert params["learning_rate"] == 0.05

    def test_fit_returns_dict(self, X_2d, y):
        model = XGBoostModel(n_estimators=10)
        result = model.fit(X_2d, y)

        assert isinstance(result, dict)
        assert "train_loss" in result

    def test_predict_returns_array(self, X_2d, y):
        model = XGBoostModel(n_estimators=10)
        model.fit(X_2d, y)
        preds = model.predict(X_2d)

        assert isinstance(preds, np.ndarray)
        assert preds.shape == (100,)

    def test_fit_predict_3d(self, X_3d, y):
        model = XGBoostModel(n_estimators=10)
        model.fit(X_3d, y)
        preds = model.predict(X_3d)

        assert preds.shape == (100,)

    def test_inherits_sklearn(self):
        model = XGBoostModel()
        assert isinstance(model, SKLearnModel)


# =============================================================================
# LightGBMModel
# =============================================================================
class TestLightGBMModel:
    """Tests for LightGBMModel."""

    def test_default_params(self):
        model = LightGBMModel()
        params = model.get_params()

        assert params["n_estimators"] == 100
        assert params["num_leaves"] == 31
        assert params["learning_rate"] == 0.1

    def test_custom_params(self):
        model = LightGBMModel(n_estimators=50, num_leaves=15, learning_rate=0.05)
        params = model.get_params()

        assert params["n_estimators"] == 50
        assert params["num_leaves"] == 15
        assert params["learning_rate"] == 0.05

    def test_fit_returns_dict(self, X_2d, y):
        model = LightGBMModel(n_estimators=10)
        result = model.fit(X_2d, y)

        assert isinstance(result, dict)
        assert "train_loss" in result

    def test_predict_returns_array(self, X_2d, y):
        model = LightGBMModel(n_estimators=10)
        model.fit(X_2d, y)
        preds = model.predict(X_2d)

        assert isinstance(preds, np.ndarray)
        assert preds.shape == (100,)

    def test_fit_predict_3d(self, X_3d, y):
        model = LightGBMModel(n_estimators=10)
        model.fit(X_3d, y)
        preds = model.predict(X_3d)

        assert preds.shape == (100,)

    def test_inherits_sklearn(self):
        model = LightGBMModel()
        assert isinstance(model, SKLearnModel)
