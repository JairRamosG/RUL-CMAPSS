"""Tests for RFModel (Random Forest)."""
import pytest
import numpy as np
from src.models.RFModel import RFModel
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
        assert issubclass(RFModel, SKLearnModel)
