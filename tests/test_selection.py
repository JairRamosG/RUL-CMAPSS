"""
Tests for feature selection module (Mutual Information + Random Forest).
TDD verification suite for Issue #5.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import re

from src.features.selection import (
    MutualInfoSelector,
    RFFeatureSelector,
    create_feature_selector,
)


@pytest.fixture
def synthetic_regression_data():
    """Generates synthetic dataset with informative and noise features.
    
    Returns:
        X: np.ndarray of shape (100, 10), where:
            - cols 0, 1, 2 are strongly informative (linear / quadratic with y)
            - cols 3, 4 are moderately informative
            - cols 5, 6, 7, 8, 9 are pure random Gaussian noise
        y: np.ndarray of shape (100,)
        feature_names: list of feature names
    """
    np.random.seed(42)
    n_samples = 150
    
    # Informative signals
    x0 = np.linspace(10, 100, n_samples)
    x1 = np.sin(x0 / 10.0) * 50.0
    x2 = x0 ** 1.5 / 10.0
    x3 = x0 + np.random.normal(0, 5, n_samples)
    x4 = -0.5 * x0 + np.random.normal(0, 5, n_samples)
    
    # Noise signals
    noise = np.random.normal(0, 10, size=(n_samples, 5))
    
    X = np.column_stack([x0, x1, x2, x3, x4, noise])
    y = 2.0 * x0 + 1.5 * x1 + 0.8 * x2 + np.random.normal(0, 2, n_samples)
    
    feature_names = [f"feat_{i}" for i in range(10)]
    return X, y, feature_names


class TestMutualInfoSelector:
    """Unit tests for MutualInfoSelector filter transformer."""

    def test_inherits_sklearn_base(self):
        """Must inherit from BaseEstimator and TransformerMixin."""
        selector = MutualInfoSelector()
        assert isinstance(selector, BaseEstimator)
        assert isinstance(selector, TransformerMixin)

    def test_requires_y(self, synthetic_regression_data):
        """Calling fit(X) without y must raise ValueError."""
        X, _, _ = synthetic_regression_data
        selector = MutualInfoSelector()
        with pytest.raises(ValueError, match=re.escape("Se requiere un target 'y' porque es selección supervisada (target de entrenamiento solamente)")):
            selector.fit(X, y=None)

    def test_percentile_selection_shape(self, synthetic_regression_data):
        """Selecting top 50th percentile of 10 features should yield 5 features."""
        X, y, _ = synthetic_regression_data
        selector = MutualInfoSelector(percentile=50, random_state=42)
        X_trans = selector.fit_transform(X, y)

        assert X_trans.shape == (150, 5)
        assert selector.n_features_in_ == 10
        assert len(selector.selected_indices_) == 5

    def test_scores_and_support_attributes(self, synthetic_regression_data):
        """After fit, scores_ and support_ attributes must be populated."""
        X, y, _ = synthetic_regression_data
        selector = MutualInfoSelector(percentile=60, random_state=42)
        selector.fit(X, y)

        assert hasattr(selector, "scores_")
        assert len(selector.scores_) == 10
        assert hasattr(selector, "support_")
        assert selector.support_.dtype == bool
        assert np.sum(selector.support_) == 6

    def test_get_feature_names_out(self, synthetic_regression_data):
        """Correctly projects input feature names based on support mask."""
        X, y, feature_names = synthetic_regression_data
        selector = MutualInfoSelector(percentile=40, random_state=42)
        selector.fit(X, y)

        names_out = selector.get_feature_names_out(feature_names)
        assert len(names_out) == 4
        for name in names_out:
            assert name in feature_names

    def test_transform_preserves_leakage_boundary(self, synthetic_regression_data):
        """Transforming validation set applies learned mask from train set."""
        X, y, _ = synthetic_regression_data
        X_train, y_train = X[:100], y[:100]
        X_val = X[100:]

        selector = MutualInfoSelector(percentile=50, random_state=42)
        X_train_trans = selector.fit_transform(X_train, y_train)
        X_val_trans = selector.transform(X_val)

        assert X_train_trans.shape == (100, 5)
        assert X_val_trans.shape == (50, 5)


class TestRFFeatureSelector:
    """Unit tests for RFFeatureSelector embedded transformer."""

    def test_inherits_sklearn_base(self):
        """Must inherit from BaseEstimator and TransformerMixin."""
        selector = RFFeatureSelector()
        assert isinstance(selector, BaseEstimator)
        assert isinstance(selector, TransformerMixin)

    def test_requires_y(self, synthetic_regression_data):
        """Calling fit(X) without y must raise ValueError."""
        X, _, _ = synthetic_regression_data
        selector = RFFeatureSelector()
        with pytest.raises(ValueError, match=re.escape("se requiere de un vector de 'y' para el RUL")):
            selector.fit(X, y=None)

    def test_exact_k_features_selection(self, synthetic_regression_data):
        """Selecting exact n_features_to_select=4 yields 4 output features."""
        X, y, _ = synthetic_regression_data
        selector = RFFeatureSelector(n_features_to_select=4, random_state=42)
        X_trans = selector.fit_transform(X, y)

        assert X_trans.shape == (150, 4)
        assert len(selector.selected_indices_) == 4

    def test_feature_importances_attribute(self, synthetic_regression_data):
        """MDI importances are computed and sum roughly to 1."""
        X, y, _ = synthetic_regression_data
        selector = RFFeatureSelector(n_features_to_select=3, random_state=42)
        selector.fit(X, y)

        assert hasattr(selector, "feature_importances_")
        assert len(selector.feature_importances_) == 10
        assert np.isclose(np.sum(selector.feature_importances_), 1.0, atol=1e-3)

    def test_get_feature_names_out(self, synthetic_regression_data):
        """Returns the names of the top features chosen by Random Forest."""
        X, y, feature_names = synthetic_regression_data
        selector = RFFeatureSelector(n_features_to_select=3, random_state=42)
        selector.fit(X, y)

        names_out = selector.get_feature_names_out(feature_names)
        assert len(names_out) == 3
        # The first 3 features are known to be the strongest signals
        assert "feat_0" in names_out or "feat_1" in names_out or "feat_2" in names_out


class TestPipelineCompositionAndFactory:
    """Unit tests for composition and YAML factory creation."""

    def test_hybrid_pipeline_composition(self, synthetic_regression_data):
        """Composing MutualInfo + RandomForest via sklearn Pipeline works seamlessly."""
        X, y, feature_names = synthetic_regression_data
        
        # 10 features -> filter top 6 (60%) -> embedded top 3
        pipeline = Pipeline([
            ("filter", MutualInfoSelector(percentile=60, random_state=42)),
            ("embedded", RFFeatureSelector(n_features_to_select=3, random_state=42)),
        ])

        X_trans = pipeline.fit_transform(X, y)
        assert X_trans.shape == (150, 3)

        # Names out through pipeline steps
        step1_names = pipeline.named_steps["filter"].get_feature_names_out(feature_names)
        assert len(step1_names) == 6
        final_names = pipeline.named_steps["embedded"].get_feature_names_out(step1_names)
        assert len(final_names) == 3

    def test_factory_disabled_returns_passthrough(self, synthetic_regression_data):
        """When enabled=False, factory returns transformer that leaves X unchanged."""
        X, y, _ = synthetic_regression_data
        cfg = {"enabled": False}
        selector = create_feature_selector(cfg)
        
        X_trans = selector.fit_transform(X, y)
        assert np.array_equal(X, X_trans)

    def test_factory_hybrid_creation(self, synthetic_regression_data):
        """Factory creates working hybrid Pipeline from dictionary configuration."""
        X, y, _ = synthetic_regression_data
        cfg = {
            "enabled": True,
            "method": "hybrid",
            "mutual_info": {"percentile": 70},
            "rf_importance": {"n_features_to_select": 4, "n_estimators": 50},
        }
        selector = create_feature_selector(cfg)
        assert isinstance(selector, Pipeline)
        
        X_trans = selector.fit_transform(X, y)
        assert X_trans.shape == (150, 4)

    def test_factory_individual_methods(self, synthetic_regression_data):
        """Factory creates single transformers when method is not hybrid."""
        X, y, _ = synthetic_regression_data
        
        cfg_mi = {"enabled": True, "method": "mutual_info", "mutual_info": {"percentile": 50}}
        sel_mi = create_feature_selector(cfg_mi)
        assert isinstance(sel_mi, MutualInfoSelector)
        assert sel_mi.fit_transform(X, y).shape == (150, 5)

        cfg_rf = {"enabled": True, "method": "rf_importance", "rf_importance": {"n_features_to_select": 2}}
        sel_rf = create_feature_selector(cfg_rf)
        assert isinstance(sel_rf, RFFeatureSelector)
        assert sel_rf.fit_transform(X, y).shape == (150, 2)
