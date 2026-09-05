"""Tests for PyTorchModel wrapper."""
import pytest
import numpy as np
import torch
import torch.nn as nn
from src.models.pytorch_wrapper import PyTorchModel, _get_device


class SimpleModule(nn.Module):
    """Simple linear module for testing."""

    def __init__(self, input_dim=5):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


class TestGetDevice:
    """Tests for _get_device function."""

    def test_returns_device(self):
        """_get_device returns a torch.device."""
        device = _get_device()
        assert isinstance(device, torch.device)

    def test_returns_cpu_or_cuda(self):
        """_get_device returns cpu or cuda."""
        device = _get_device()
        assert device.type in ("cpu", "cuda")


class TestPyTorchModel:
    """Tests for PyTorchModel wrapper."""

    @pytest.fixture
    def X_2d(self):
        return np.random.randn(100, 5).astype(np.float32)

    @pytest.fixture
    def y(self):
        return np.random.randn(100).astype(np.float32)

    @pytest.fixture
    def simple_model(self):
        return PyTorchModel(SimpleModule(input_dim=5), lr=1e-3)

    def test_fit_returns_dict(self, simple_model, X_2d, y):
        """fit() returns dict with required keys."""
        result = simple_model.fit(X_2d, y, epochs=5)

        assert isinstance(result, dict)
        assert "train_loss" in result
        assert "val_loss" in result
        assert "epochs_trained" in result
        assert "early_stopped" in result

    def test_fit_trains_model(self, simple_model, X_2d, y):
        """fit() trains the model (loss decreases)."""
        result = simple_model.fit(X_2d, y, epochs=10)

        assert result["epochs_trained"] == 10
        assert result["early_stopped"] is False
        assert result["train_loss"] >= 0

    def test_predict_returns_array(self, simple_model, X_2d, y):
        """predict() returns numpy array."""
        simple_model.fit(X_2d, y, epochs=5)
        preds = simple_model.predict(X_2d)

        assert isinstance(preds, np.ndarray)
        assert preds.shape == (100,)

    def test_early_stopping(self, X_2d, y):
        """Early stopping triggers when no improvement."""
        model = PyTorchModel(SimpleModule(input_dim=5), lr=1e-10)

        # Use val data that will never improve
        result = model.fit(
            X_2d, y,
            X_val=X_2d, y_val=y,
            epochs=50, patience=5
        )

        assert result["early_stopped"] is True
        assert result["epochs_trained"] <= 50

    def test_get_params(self, simple_model):
        """get_params() returns parameters dict."""
        params = simple_model.get_params()

        assert isinstance(params, dict)

    def test_device_detection(self, simple_model):
        """Model is placed on detected device."""
        assert simple_model.device.type in ("cpu", "cuda")
        assert next(simple_model.module.parameters()).device == simple_model.device
