"""Shared fixtures for the test suite (synthetic data samples only)."""

import numpy as np
import pytest


@pytest.fixture
def X_2d():
    """2D feature matrix of shape (100, 5)."""
    return np.random.randn(100, 5)


@pytest.fixture
def X_3d():
    """3D feature tensor of shape (100, 10, 5)."""
    return np.random.randn(100, 10, 5)


@pytest.fixture
def y():
    """Target RUL vector of shape (100,)."""
    return np.random.randn(100)
