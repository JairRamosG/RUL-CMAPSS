"""Tests for SVRModel."""
import pytest
import numpy as np
from src.models.SVRModel import SVRModel


class TestSVRModel:
    """Tests for SVRModel."""

    @pytest.fixture
    def X_2d(self):
        return np.random.randn(100, 5)

    @pytest.fixture
    def X_3d(self):
        return np.random.randn(100, 10, 5)

    @pytest.fixture
    def y(self):
        return np.random.randn(100)

    def test_default_params(self):
        """SVRModel has correct default parameters."""
        model = SVRModel()
        params = model.get_params()

        assert params["C"] == 1.0
        assert params["epsilon"] == 0.1
        assert params["kernel"] == "rbf"

    def test_custom_params(self):
        """SVRModel accepts custom parameters."""
        model = SVRModel(C=2.0, epsilon=0.5, kernel="linear")
        params = model.get_params()

        assert params["C"] == 2.0
        assert params["epsilon"] == 0.5
        assert params["kernel"] == "linear"

    def test_fit_returns_dict(self, X_2d, y):
        """fit() returns dict with required keys."""
        model = SVRModel()
        result = model.fit(X_2d, y)

        assert isinstance(result, dict)
        assert "train_loss" in result
        assert "val_loss" in result
        assert "epoch_trained" in result
        assert "early_stopped" in result

    def test_predict_returns_array(self, X_2d, y):
        """predict() returns numpy array with correct shape."""
        model = SVRModel()
        model.fit(X_2d, y)
        preds = model.predict(X_2d)

        assert isinstance(preds, np.ndarray)
        assert preds.shape == (100,)

    def test_fit_predict_3d(self, X_3d, y):
        """SVRModel works with 3D input (flattens automatically)."""
        model = SVRModel()
        result = model.fit(X_3d, y)
        preds = model.predict(X_3d)

        assert result["train_loss"] == 0.0
        assert preds.shape == (100,)

    def test_inherits_from_sklearn(self):
        """SVRModel inherits from SKLearnModel."""
        from src.models.sklearn_wrapper import SKLearnModel

        model = SVRModel()
        assert isinstance(model, SKLearnModel)
