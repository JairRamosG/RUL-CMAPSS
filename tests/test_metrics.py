"""
Tests for evaluation metrics and profiling module.
"""

import pytest
import numpy as np

from src.evaluation.metrics import (
    rmse, mae, nasa_score,
    profile_resource_usage, get_system_info, ResourceProfiler
)


class TestRMSE:
    """Tests for rmse function."""
    
    def test_perfect_predictions(self):
        """Test RMSE with perfect predictions."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        
        assert rmse(y_true, y_pred) == 0.0
    
    def test_known_values(self):
        """Test RMSE with known values."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 3, 5])
        
        # RMSE = sqrt(((0)^2 + (1)^2 + (2)^2) / 3) = sqrt(5/3)
        expected = np.sqrt(5/3)
        
        assert abs(rmse(y_true, y_pred) - expected) < 1e-10
    
    def test_returns_float(self):
        """Test that RMSE returns a float."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 3])
        
        assert isinstance(rmse(y_true, y_pred), float)


class TestMAE:
    """Tests for mae function."""
    
    def test_perfect_predictions(self):
        """Test MAE with perfect predictions."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        
        assert mae(y_true, y_pred) == 0.0
    
    def test_known_values(self):
        """Test MAE with known values."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 3, 5])
        
        # MAE = (0 + 1 + 2) / 3 = 1
        expected = 1.0
        
        assert abs(mae(y_true, y_pred) - expected) < 1e-10
    
    def test_returns_float(self):
        """Test that MAE returns a float."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 3])
        
        assert isinstance(mae(y_true, y_pred), float)


class TestNASAScore:
    """Tests for nasa_score function."""
    
    def test_perfect_predictions(self):
        """Test NASA score with perfect predictions."""
        y_true = np.array([100, 50, 10])
        y_pred = np.array([100, 50, 10])
        
        assert nasa_score(y_true, y_pred) == 0.0
    
    def test_overestimation_penalized_more(self):
        """Test that overestimation is penalized more than underestimation."""
        y_true = np.array([100])
        
        # Overestimation by 10
        y_pred_over = np.array([110])
        score_over = nasa_score(y_true, y_pred_over)
        
        # Underestimation by 10
        y_pred_under = np.array([90])
        score_under = nasa_score(y_true, y_pred_under)
        
        # Overestimation should have higher penalty
        assert score_over > score_under
    
    def test_returns_float(self):
        """Test that NASA score returns a float."""
        y_true = np.array([100])
        y_pred = np.array([100])
        
        assert isinstance(nasa_score(y_true, y_pred), float)


class TestProfileResourceUsage:
    """Tests for profile_resource_usage context manager."""
    
    def test_returns_profiler(self):
        """Test that context manager returns a profiler."""
        with profile_resource_usage() as profiler:
            pass
        
        assert isinstance(profiler, ResourceProfiler)
    
    def test_elapsed_time_positive(self):
        """Test that elapsed time is positive."""
        with profile_resource_usage() as profiler:
            pass
        
        assert profiler.elapsed_time >= 0
    
    def test_peak_memory_positive(self):
        """Test that peak memory is positive."""
        with profile_resource_usage() as profiler:
            pass
        
        assert profiler.peak_memory_mb > 0
    
    def test_captures_work(self):
        """Test that profiler captures work done."""
        with profile_resource_usage() as profiler:
            # Do some work
            _ = np.random.randn(1000, 1000)
        
        assert profiler.elapsed_time > 0
        assert profiler.peak_memory_mb > 0


class TestGetSystemInfo:
    """Tests for get_system_info function."""
    
    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        info = get_system_info()
        assert isinstance(info, dict)
    
    def test_contains_cpu_info(self):
        """Test that dictionary contains CPU info."""
        info = get_system_info()
        assert "cpu" in info
        assert "cpu_count" in info["cpu"]
    
    def test_contains_memory_info(self):
        """Test that dictionary contains memory info."""
        info = get_system_info()
        assert "memory" in info
        assert "total_gb" in info["memory"]
