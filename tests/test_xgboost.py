"""Tests for XGBoostModel."""
import pytest
import numpy as np
from src.models.XGBoostModel import XGBoostModel
from src.models.sklearn_wrapper import SKLearnModel


@pytest.fixture
def X_2d():
    return np.random.randn(100, 5)


@pytest.fixture
def X_3d():
    return np.random.randn(100, 10, 5)


@pytest.fixture
def y():
    return np.random.randn(100)


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
        assert issubclass(XGBoostModel, SKLearnModel)
