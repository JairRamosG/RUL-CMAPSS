"""Tests for evaluation metrics."""
import time
import pytest
import numpy as np
from src.evaluation.metrics import (
    rmse,
    mae,
    nasa_score,
    ResourceProfiler,
    profile_resource_usage,
    get_system_info,
    profile_function,
)


class TestRMSE:
    """Tests for Root Mean Squared Error."""

    def test_perfect_prediction(self):
        """RMSE is 0 when predictions are exact."""
        y_true = np.array([100, 200, 300])
        y_pred = np.array([100, 200, 300])

        assert rmse(y_true, y_pred) == 0.0

    def test_known_values(self):
        """RMSE calculates correctly with known values."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 4.0])  # error: [0, 0, 1]

        # sqrt(mean([0, 0, 1])) = sqrt(1/3) ≈ 0.577
        result = rmse(y_true, y_pred)
        assert np.isclose(result, np.sqrt(1 / 3), atol=1e-6)

    def test_symmetric_errors(self):
        """RMSE penalizes equally positive and negative errors."""
        y_true = np.array([10.0, 10.0])
        y_pred_pos = np.array([12.0, 12.0])  # error +2
        y_pred_neg = np.array([8.0, 8.0])    # error -2

        assert np.isclose(rmse(y_true, y_pred_pos), rmse(y_true, y_pred_neg))

    def test_penalizes_large_errors(self):
        """RMSE penalizes large errors more than small errors."""
        y_true = np.array([100.0, 100.0, 100.0])
        y_pred_small = np.array([101.0, 101.0, 101.0])  # all errors +1
        y_pred_large = np.array([103.0, 100.0, 100.0])  # one error +3

        # Large error in one sample should produce higher RMSE
        assert rmse(y_true, y_pred_large) > rmse(y_true, y_pred_small)

    def test_returns_scalar(self):
        """RMSE returns a scalar value."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])

        result = rmse(y_true, y_pred)
        assert np.isscalar(result) or result.ndim == 0


class TestMAE:
    """Tests for Mean Absolute Error."""

    def test_perfect_prediction(self):
        """MAE is 0 when predictions are exact."""
        y_true = np.array([100, 200, 300])
        y_pred = np.array([100, 200, 300])

        assert mae(y_true, y_pred) == 0.0

    def test_known_values(self):
        """MAE calculates correctly with known values."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 4.0])  # error: [0, 0, 1]

        # mean([0, 0, 1]) = 1/3 ≈ 0.333
        result = mae(y_true, y_pred)
        assert np.isclose(result, 1 / 3, atol=1e-6)

    def test_symmetric_errors(self):
        """MAE treats positive and negative errors equally."""
        y_true = np.array([10.0, 10.0])
        y_pred_pos = np.array([12.0, 12.0])  # error +2
        y_pred_neg = np.array([8.0, 8.0])    # error -2

        assert np.isclose(mae(y_true, y_pred_pos), mae(y_true, y_pred_neg))

    def test_linear_penalty(self):
        """MAE penalizes errors linearly (unlike RMSE)."""
        y_true = np.array([100.0, 100.0, 100.0])
        y_pred_small = np.array([101.0, 101.0, 101.0])  # all errors +1
        y_pred_large = np.array([103.0, 100.0, 100.0])  # one error +3

        # MAE: 1.0 vs 1.0 (linear)
        assert np.isclose(mae(y_true, y_pred_small), 1.0)
        assert np.isclose(mae(y_true, y_pred_large), 1.0)

    def test_returns_scalar(self):
        """MAE returns a scalar value."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])

        result = mae(y_true, y_pred)
        assert np.isscalar(result) or result.ndim == 0


class TestNASAScore:
    """Tests for NASA Scoring Function (Saxena et al., 2008)."""

    def test_perfect_prediction(self):
        """NASA Score is 0 when predictions are exact."""
        y_true = np.array([100, 200, 300])
        y_pred = np.array([100, 200, 300])

        assert nasa_score(y_true, y_pred) == 0.0

    def test_overestimation_penalized_more(self):
        """Overestimation (y_pred > y_true) is penalized more than underestimation."""
        y_true = np.array([100.0])

        # Underestimation by 10: d = -10, s = exp(10/13) - 1 ≈ 1.15
        y_pred_under = np.array([90.0])
        score_under = nasa_score(y_true, y_pred_under)

        # Overestimation by 10: d = +10, s = exp(10/10) - 1 ≈ 1.72
        y_pred_over = np.array([110.0])
        score_over = nasa_score(y_true, y_pred_over)

        assert score_over > score_under

    def test_asymmetry_ratio(self):
        """The asymmetry follows the formula: exp(10/10)-1 > exp(10/13)-1."""
        y_true = np.array([100.0])

        # Underestimation by 10
        y_pred_under = np.array([90.0])
        score_under = nasa_score(y_true, y_pred_under)

        # Overestimation by 10
        y_pred_over = np.array([110.0])
        score_over = nasa_score(y_true, y_pred_over)

        # exp(10/10) - 1 ≈ 1.718
        # exp(10/13) - 1 ≈ 1.150
        expected_over = np.exp(10 / 10) - 1
        expected_under = np.exp(10 / 13) - 1

        assert np.isclose(score_over, expected_over, atol=1e-6)
        assert np.isclose(score_under, expected_under, atol=1e-6)

    def test_known_values(self):
        """NASA Score calculates correctly with known values."""
        y_true = np.array([100.0])
        y_pred = np.array([110.0])  # overestimation by 10

        # d = 10, s = exp(10/10) - 1 ≈ 1.718
        result = nasa_score(y_true, y_pred)
        expected = np.exp(10 / 10) - 1

        assert np.isclose(result, expected, atol=1e-6)

    def test_multiple_samples(self):
        """NASA Score sums across multiple samples."""
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 210.0])  # both overestimation by 10

        # Each sample contributes exp(10/10) - 1 ≈ 1.718
        result = nasa_score(y_true, y_pred)
        expected = 2 * (np.exp(10 / 10) - 1)

        assert np.isclose(result, expected, atol=1e-6)

    def test_returns_scalar(self):
        """NASA Score returns a scalar value."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])

        result = nasa_score(y_true, y_pred)
        assert np.isscalar(result) or result.ndim == 0


class TestResourceProfiler:
    """Tests for ResourceProfiler context manager."""

    def test_measures_elapsed_time(self):
        """Elapsed time covers the profiled block."""
        with ResourceProfiler() as profiler:
            time.sleep(0.01)

        assert profiler.elapsed_time >= 0.01

    def test_peak_memory_positive(self):
        """Peak memory usage is a positive value in MB."""
        with ResourceProfiler() as profiler:
            _ = [0.0] * 100_000

        assert profiler.peak_memory_mb > 0

    def test_initial_values(self):
        """Profiler starts with zeroed metrics before entering the context."""
        profiler = ResourceProfiler()

        assert profiler.elapsed_time == 0.0
        assert profiler.peak_memory_mb == 0.0

    def test_to_dict_returns_metrics(self):
        """to_dict exposes elapsed_time and peak_memory_mb keys."""
        with ResourceProfiler() as profiler:
            pass

        result = profiler.to_dict()

        assert set(result.keys()) == {"elapsed_time", "peak_memory_mb"}
        assert result["elapsed_time"] >= 0.0
        assert result["peak_memory_mb"] > 0

    def test_exception_does_not_swallow(self):
        """Profiler does not suppress exceptions raised in the block."""
        with pytest.raises(ZeroDivisionError):
            with ResourceProfiler():
                1 / 0


class TestProfileResourceUsage:
    """Tests for profile_resource_usage context manager."""

    def test_yields_profiler_with_results(self):
        """Yields a ResourceProfiler populated after the block exits."""
        with profile_resource_usage() as profiler:
            time.sleep(0.005)

        assert isinstance(profiler, ResourceProfiler)
        assert profiler.elapsed_time >= 0.005
        assert profiler.peak_memory_mb > 0


class TestGetSystemInfo:
    """Tests for get_system_info function."""

    def test_returns_expected_structure(self):
        """System info contains cpu, memory and platform sections."""
        info = get_system_info()

        assert "cpu" in info
        assert "memory" in info
        assert "platform" in info

    def test_cpu_and_memory_values(self):
        """CPU count and total memory are positive."""
        info = get_system_info()

        assert info["cpu"]["cpu_count"] >= 1
        assert info["cpu"]["cpu_count_logical"] >= 1
        assert info["memory"]["total_gb"] > 0
        assert 0 < info["memory"]["percent_used"] <= 100


class TestProfileFunction:
    """Tests for profile_function decorator."""

    def test_returns_result_and_profiler(self):
        """Decorated function returns (result, profiler) tuple."""

        @profile_function
        def add(a, b):
            return a + b

        result, profiler = add(2, 3)

        assert result == 5
        assert isinstance(profiler, ResourceProfiler)
        assert profiler.elapsed_time >= 0.0

    def test_preserves_function_metadata(self):
        """Decorator preserves the wrapped function name."""

        @profile_function
        def sample():
            """Docstring here."""

        assert sample.__name__ == "sample"
        assert sample.__doc__ == "Docstring here."
