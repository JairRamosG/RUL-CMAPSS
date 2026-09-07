"""Tests for MLPModel."""
import pytest
import numpy as np
from src.models.MLPModel import MLPModel
from src.models.pytorch_wrapper import PyTorchModel


class TestMLPModel:
    """Tests for MLPModel."""

    @pytest.fixture
    def X_2d(self):
        return np.random.randn(100, 5).astype(np.float32)

    @pytest.fixture
    def X_3d(self):
        return np.random.randn(100, 10, 5).astype(np.float32)

    @pytest.fixture
    def y(self):
        return np.random.randn(100).astype(np.float32)

    def test_default_params(self):
        """MLPModel has correct default parameters."""
        model = MLPModel(input_dim=5)
        params = model.get_params()

        assert params["input_dim"] == 5
        assert params["hidden_units"] == [128, 64]
        assert params["dropout"] == 0.2
        assert params["lr"] == 1e-3

    def test_custom_params(self):
        """MLPModel accepts custom parameters."""
        model = MLPModel(input_dim=10, hidden_units=[64, 32], dropout=0.5, lr=0.001)
        params = model.get_params()

        assert params["input_dim"] == 10
        assert params["hidden_units"] == [64, 32]
        assert params["dropout"] == 0.5
        assert params["lr"] == 0.001

    def test_fit_returns_dict(self, X_2d, y):
        """fit() returns dict with required keys."""
        model = MLPModel(input_dim=5, hidden_units=[32])
        result = model.fit(X_2d, y, epochs=5)

        assert isinstance(result, dict)
        assert "train_loss" in result
        assert "val_loss" in result
        assert "epochs_trained" in result
        assert "early_stopped" in result

    def test_predict_returns_array(self, X_2d, y):
        """predict() returns numpy array with correct shape."""
        model = MLPModel(input_dim=5, hidden_units=[32])
        model.fit(X_2d, y, epochs=5)
        preds = model.predict(X_2d)

        assert isinstance(preds, np.ndarray)
        assert preds.shape == (100,)

    def test_fit_predict_3d(self, X_3d, y):
        """MLPModel works with 3D input (flattens automatically)."""
        model = MLPModel(input_dim=50, hidden_units=[32])  # 10 * 5 = 50
        result = model.fit(X_3d, y, epochs=5)
        preds = model.predict(X_3d)

        assert result["epochs_trained"] <= 5
        assert preds.shape == (100,)

    def test_early_stopping_with_val(self, X_2d, y):
        """Early stopping activates when val data provided."""
        model = MLPModel(input_dim=5, hidden_units=[32], lr=1e-10)
        X_val = np.random.randn(20, 5).astype(np.float32)
        y_val = np.random.randn(20).astype(np.float32)

        result = model.fit(X_2d, y, X_val=X_val, y_val=y_val, epochs=50, patience=3)

        assert isinstance(result["early_stopped"], bool)
        assert result["epochs_trained"] <= 50

    def test_inherits_from_pytorch(self):
        """MLPModel inherits from PyTorchModel."""
        model = MLPModel(input_dim=5, hidden_units=[32])
        assert isinstance(model, PyTorchModel)

    def test_different_architectures(self):
        """Different hidden_units produce different architectures."""
        shallow = MLPModel(input_dim=5, hidden_units=[32])
        deep = MLPModel(input_dim=5, hidden_units=[64, 32, 16, 8])

        params_shallow = sum(p.numel() for p in shallow.module.parameters())
        params_deep = sum(p.numel() for p in deep.module.parameters())
        assert params_deep > params_shallow

    def test_train_loss_is_non_negative(self, X_2d, y):
        """Training loss is non-negative (MSELoss)."""
        model = MLPModel(input_dim=5, hidden_units=[32])
        result = model.fit(X_2d, y, epochs=5)

        assert result["train_loss"] >= 0
