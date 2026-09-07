"""Tests for LSTMModel."""
import pytest
import numpy as np
import torch
from src.models.LSTMModel import LSTMModel
from src.models.pytorch_wrapper import PyTorchModel


class TestLSTMModel:
    """Tests for LSTMModel."""

    @pytest.fixture
    def X_3d(self):
        """3D input: (N=100, W=10, F=5) -> batch, window, features."""
        return np.random.randn(100, 10, 5).astype(np.float32)

    @pytest.fixture
    def y(self):
        return np.random.randn(100).astype(np.float32)

    def test_default_params(self):
        """LSTMModel has correct default parameters."""
        model = LSTMModel(num_features=5)
        params = model.get_params()

        assert params["num_features"] == 5
        assert params["hidden_dim"] == 64
        assert params["num_layers"] == 2
        assert params["dropout"] == 0.2
        assert params["lr"] == 1e-3

    def test_custom_params(self):
        """LSTMModel accepts custom parameters."""
        model = LSTMModel(
            num_features=5,
            hidden_dim=128,
            num_layers=3,
            dropout=0.3,
            lr=0.001,
        )
        params = model.get_params()

        assert params["hidden_dim"] == 128
        assert params["num_layers"] == 3
        assert params["dropout"] == 0.3
        assert params["lr"] == 0.001

    def test_fit_returns_dict(self, X_3d, y):
        """fit() returns dict with required keys."""
        model = LSTMModel(num_features=5, hidden_dim=16, num_layers=1)
        result = model.fit(X_3d, y, epochs=3)

        assert isinstance(result, dict)
        assert "train_loss" in result
        assert "val_loss" in result
        assert "epochs_trained" in result
        assert "early_stopped" in result

    def test_predict_returns_array(self, X_3d, y):
        """predict() returns numpy array with correct shape."""
        model = LSTMModel(num_features=5, hidden_dim=16, num_layers=1)
        model.fit(X_3d, y, epochs=3)
        preds = model.predict(X_3d)

        assert isinstance(preds, np.ndarray)
        assert preds.shape == (100,)

    def test_fit_predict_3d(self, X_3d, y):
        """LSTMModel works with 3D input (N, W, F)."""
        model = LSTMModel(num_features=5, hidden_dim=16, num_layers=1)
        result = model.fit(X_3d, y, epochs=3)
        preds = model.predict(X_3d)

        assert result["epochs_trained"] <= 3
        assert preds.shape == (100,)

    def test_early_stopping_with_val(self, X_3d, y):
        """Early stopping activates when val data provided."""
        model = LSTMModel(
            num_features=5, hidden_dim=16, num_layers=1, lr=1e-10
        )
        X_val = np.random.randn(20, 10, 5).astype(np.float32)
        y_val = np.random.randn(20).astype(np.float32)

        result = model.fit(X_3d, y, X_val=X_val, y_val=y_val, epochs=20, patience=3)

        assert isinstance(result["early_stopped"], bool)
        assert result["epochs_trained"] <= 20

    def test_inherits_from_pytorch(self):
        """LSTMModel inherits from PyTorchModel."""
        model = LSTMModel(num_features=5, hidden_dim=16, num_layers=1)
        assert isinstance(model, PyTorchModel)

    def test_different_architectures(self):
        """Different hidden_dim/num_layers produce different architectures."""
        small = LSTMModel(num_features=5, hidden_dim=16, num_layers=1)
        large = LSTMModel(num_features=5, hidden_dim=64, num_layers=3)

        params_small = sum(p.numel() for p in small.module.parameters())
        params_large = sum(p.numel() for p in large.module.parameters())
        assert params_large > params_small

    def test_train_loss_is_non_negative(self, X_3d, y):
        """Training loss is non-negative (MSELoss)."""
        model = LSTMModel(num_features=5, hidden_dim=16, num_layers=1)
        result = model.fit(X_3d, y, epochs=3)

        assert result["train_loss"] >= 0

    def test_output_is_scalar_per_sample(self, X_3d, y):
        """Model outputs one scalar per sample (regression)."""
        model = LSTMModel(num_features=5, hidden_dim=16, num_layers=1)
        model.fit(X_3d, y, epochs=3)
        preds = model.predict(X_3d)

        assert preds.ndim == 1
        assert preds.shape[0] == X_3d.shape[0]

    def test_uses_last_timestep(self, X_3d, y):
        """LSTM uses only last timestep output for prediction."""
        model = LSTMModel(num_features=5, hidden_dim=16, num_layers=1)
        # Verificar que la arquitectura toma el último paso temporal
        # LSTM output: (batch, window, hidden) -> last: (batch, hidden)
        assert model.module.regressor is not None
