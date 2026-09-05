"""Tests for SKLearnModel adapter."""
import pytest
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from src.models.sklearn_wrapper import SKLearnModel


class TestSKLearnModel:
    """Tests for SKLearnModel adapter."""

    @pytest.fixture
    def X_2d(self):
        """2D feature array (N, F)."""
        return np.random.randn(100, 5)

    @pytest.fixture
    def X_3d(self):
        """3D feature array (N, W, F)."""
        return np.random.randn(100, 10, 5)

    @pytest.fixture
    def y(self):
        """Target RUL array."""
        return np.random.randn(100)

    def test_fit_returns_dict(self, X_2d, y):
        """fit() returns dict with required keys."""
        model = SKLearnModel(SVR())
        result = model.fit(X_2d, y)

        assert isinstance(result, dict)
        assert "train_loss" in result
        assert "val_loss" in result
        assert "epoch_trained" in result
        assert "early_stopped" in result

    def test_predict_returns_array(self, X_2d, y):
        """predict() returns numpy array."""
        model = SKLearnModel(SVR())
        model.fit(X_2d, y)
        preds = model.predict(X_2d)

        assert isinstance(preds, np.ndarray)
        assert preds.shape == (100,)

    def test_flatten_3d_to_2d(self, X_3d, y):
        """SKLearnModel flattens 3D input to 2D."""
        model = SKLearnModel(SVR())
        result = model.fit(X_3d, y)

        assert result["train_loss"] == 0.0
        preds = model.predict(X_3d)
        assert preds.shape == (100,)

    def test_flatten_preserves_2d(self, X_2d, y):
        """SKLearnModel preserves 2D input."""
        model = SKLearnModel(SVR())
        model.fit(X_2d, y)
        preds = model.predict(X_2d)

        assert preds.shape == (100,)

    def test_get_params_returns_dict(self):
        """get_params() returns model parameters."""
        model = SKLearnModel(SVR(C=2.0, epsilon=0.5))
        params = model.get_params()

        assert isinstance(params, dict)
        assert params["C"] == 2.0
        assert params["epsilon"] == 0.5

    def test_with_random_forest(self, X_2d, y):
        """SKLearnModel works with RandomForest."""
        model = SKLearnModel(RandomForestRegressor(n_estimators=10))
        model.fit(X_2d, y)
        preds = model.predict(X_2d)

        assert preds.shape == (100,)
