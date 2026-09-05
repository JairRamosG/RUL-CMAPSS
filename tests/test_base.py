"""Tests for BaseModel ABC."""
import pytest
from src.models.base import BaseModel


class TestBaseModel:
    """Tests for BaseModel abstract class."""

    def test_cannot_instantiate_directly(self):
        """BaseModel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseModel()

    def test_subclass_must_implement_fit(self):
        """Subclass without fit raises TypeError."""

        class IncompleteModel(BaseModel):
            def predict(self, X):
                return X

            def get_params(self):
                return {}

        with pytest.raises(TypeError):
            IncompleteModel()

    def test_subclass_must_implement_predict(self):
        """Subclass without predict raises TypeError."""

        class IncompleteModel(BaseModel):
            def fit(self, X_train, y_train, X_val=None, y_val=None):
                return {}

            def get_params(self):
                return {}

        with pytest.raises(TypeError):
            IncompleteModel()

    def test_subclass_must_implement_get_params(self):
        """Subclass without get_params raises TypeError."""

        class IncompleteModel(BaseModel):
            def fit(self, X_train, y_train, X_val=None, y_val=None):
                return {}

            def predict(self, X):
                return X

        with pytest.raises(TypeError):
            IncompleteModel()

    def test_complete_subclass_can_instantiate(self):
        """Subclass with all abstract methods can instantiate."""

        class CompleteModel(BaseModel):
            def fit(self, X_train, y_train, X_val=None, y_val=None):
                return {"train_loss": 0.0}

            def predict(self, X):
                return X

            def get_params(self):
                return {}

        model = CompleteModel()
        assert model is not None
