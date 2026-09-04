"""
Tests for feature engineering module.
"""

import pytest
import pandas as pd
import numpy as np

from src.features.engineering import create_windows, compute_rolling_stats, compute_trends


def _make_df(units: dict[int, int], n_features: int = 3) -> pd.DataFrame:
    """Helper: build a minimal DataFrame for testing.

    Args:
        units: Mapping {unit_number: n_cycles}.
        n_features: Number of sensor columns to generate.

    Returns:
        DataFrame with unit_number, time, rul, and sensor columns.
    """
    rows = []
    for unit, cycles in units.items():
        for t in range(1, cycles + 1):
            row = {"unit_number": unit, "time": t, "rul": cycles - t}
            for s in range(1, n_features + 1):
                row[f"sensor_{s}"] = unit * 100 + t * s
            rows.append(row)
    return pd.DataFrame(rows)


class TestCreateWindows:
    """Tests for create_windows function."""

    # ── Output shapes ──────────────────────────────────────────────

    def test_output_shapes(self):
        """X has shape (n_windows, window_size, n_features)."""
        df = _make_df({1: 50, 2: 50}, n_features=2)
        X, y = create_windows(df, window_size=10)

        assert X.ndim == 3
        assert X.shape[1] == 10
        assert X.shape[2] == 2
        assert y.ndim == 1
        assert len(y) == X.shape[0]

    def test_single_motor_many_cycles(self):
        """One motor with more cycles than window_size produces multiple windows."""
        df = _make_df({1: 20}, n_features=1)
        X, y = create_windows(df, window_size=5)

        # 20 cycles → starts 0..15 → 16 windows
        assert X.shape[0] == 16
        assert X.shape[1] == 5
        assert len(y) == 16

    def test_multiple_motors(self):
        """Windows from different motors are concatenated."""
        df = _make_df({1: 15, 2: 15}, n_features=1)
        X, y = create_windows(df, window_size=5)

        # Each motor: 15 cycles → 11 windows → total 22
        assert X.shape[0] == 22

    # ── Edge padding ───────────────────────────────────────────────

    def test_padding_short_motor(self):
        """Motor with T < window_size produces exactly one padded window."""
        df = _make_df({1: 8}, n_features=2)
        X, y = create_windows(df, window_size=10)

        assert X.shape[0] == 1
        assert X.shape[1] == 10

    def test_padding_repeats_first_row(self):
        """Padded positions are copies of the first recorded row."""
        df = _make_df({1: 5}, n_features=2)
        X, y = create_windows(df, window_size=10)

        # First 5 rows should be the padding (repeated first row)
        first_row = X[0, 5, :]  # First real cycle is at index 5
        for i in range(5):
            np.testing.assert_array_equal(X[0, i, :], first_row)

    def test_no_padding_when_enough_cycles(self):
        """No padding when T >= window_size."""
        df = _make_df({1: 30}, n_features=2)
        X, y = create_windows(df, window_size=10)

        # Window 0: cycles 1..10, no padding
        # First row of window should NOT be repeated
        assert not np.array_equal(X[0, 0, :], X[0, 1, :])

    # ── Target values ──────────────────────────────────────────────

    def test_target_is_rul_at_last_real_cycle(self):
        """y[i] equals the RUL at the last real cycle of the window."""
        df = _make_df({1: 10}, n_features=1)
        X, y = create_windows(df, window_size=5)

        # Window starting at t=1: covers t=1..5, last real cycle=5, rul=5
        assert y[0] == pytest.approx(5.0)
        # Window starting at t=5: covers t=5..9, last real cycle=9, rul=1
        assert y[4] == pytest.approx(1.0)
        # Window starting at t=6: covers t=6..10, last real cycle=10, rul=0
        assert y[5] == pytest.approx(0.0)

    def test_target_padded_motor(self):
        """Padded motor target equals RUL at last actual cycle."""
        df = _make_df({1: 5}, n_features=1)
        X, y = create_windows(df, window_size=10)

        # Last cycle is t=5, rul = 5 - 5 = 0
        assert y[0] == pytest.approx(0.0)

    # ── Values correctness ─────────────────────────────────────────

    def test_window_contains_correct_cycles(self):
        """Each window slices the correct consecutive cycles."""
        df = _make_df({1: 10}, n_features=1)
        X, y = create_windows(df, window_size=3)

        # Window 0: t=1,2,3 → sensor_1 = 101, 102, 103
        np.testing.assert_array_equal(X[0, :, 0], [101, 102, 103])
        # Window 1: t=2,3,4 → sensor_1 = 102, 103, 104
        np.testing.assert_array_equal(X[1, :, 0], [102, 103, 104])

    def test_no_cross_unit_contamination(self):
        """Windows never mix data from different motors."""
        df = _make_df({1: 10, 2: 10}, n_features=1)
        X, y = create_windows(df, window_size=5)

        # Motor 1 values: sensor_1 = 101..110
        # Motor 2 values: sensor_1 = 201..210
        # No window should contain values from both ranges
        for i in range(X.shape[0]):
            vals = X[i, :, 0]
            assert np.all(vals < 200) or np.all(vals >= 200), (
                f"Window {i} mixes motors: {vals}"
            )

    # ── RUL auto-computation ───────────────────────────────────────

    def test_auto_rul_when_column_absent(self):
        """RUL is computed automatically when 'rul' column is missing."""
        df = _make_df({1: 10}, n_features=1)
        df = df.drop(columns=["rul"])
        X, y = create_windows(df, window_size=5)

        # Should still produce valid targets
        assert y.shape[0] == X.shape[0]
        assert np.all(y >= 0)

    # ── Input validation ───────────────────────────────────────────

    def test_missing_unit_number_raises(self):
        """ValueError when unit_number is missing."""
        df = pd.DataFrame({"time": [1, 2], "sensor_1": [10, 20]})
        with pytest.raises(ValueError, match="unit_number"):
            create_windows(df)

    def test_missing_time_raises(self):
        """ValueError when time is missing."""
        df = pd.DataFrame({"unit_number": [1, 1], "sensor_1": [10, 20]})
        with pytest.raises(ValueError, match="time"):
            create_windows(df)

    def test_window_size_zero_raises(self):
        """ValueError when window_size < 1."""
        df = _make_df({1: 10})
        with pytest.raises(ValueError, match="ventana"):
            create_windows(df, window_size=0)

    def test_invalid_pad_strategy_raises(self):
        """ValueError for unsupported pad_strategy."""
        df = _make_df({1: 10})
        with pytest.raises(ValueError, match="pad_strategy"):
            create_windows(df, pad_strategy="zero")

    def test_no_feature_columns_raises(self):
        """ValueError when only identifiers are present."""
        df = pd.DataFrame({"unit_number": [1, 1], "time": [1, 2], "rul": [1, 0]})
        with pytest.raises(ValueError, match="columnas para entrenar"):
            create_windows(df)

    # ── Does not mutate input ──────────────────────────────────────

    def test_input_not_mutated(self):
        """Original DataFrame is not modified."""
        df = _make_df({1: 10}, n_features=2)
        original_cols = list(df.columns)
        _ = create_windows(df, window_size=5)
        assert list(df.columns) == original_cols


class TestComputeRollingStats:
    """Tests for compute_rolling_stats function."""

    # ── Output structure ────────────────────────────────────────────

    def test_adds_stat_columns(self):
        """Each feature gets 4 new columns (mean, std, min, max)."""
        df = _make_df({1: 50}, n_features=2)
        result = compute_rolling_stats(df, window_size=10)

        # Original 5 columns + 2 sensors * 4 stats = 8 new
        assert result.shape[1] == df.shape[1] + 8

    def test_column_naming(self):
        """New columns follow the pattern <col>_<stat>."""
        df = _make_df({1: 50}, n_features=2)
        result = compute_rolling_stats(df, window_size=10)

        expected = [
            "sensor_1_mean", "sensor_1_std", "sensor_1_min", "sensor_1_max",
            "sensor_2_mean", "sensor_2_std", "sensor_2_min", "sensor_2_max",
        ]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_custom_stat_types(self):
        """Only requested stats are computed."""
        df = _make_df({1: 50}, n_features=1)
        result = compute_rolling_stats(df, window_size=10, stat_types=["mean", "std"])

        assert "sensor_1_mean" in result.columns
        assert "sensor_1_std" in result.columns
        assert "sensor_1_min" not in result.columns
        assert "sensor_1_max" not in result.columns

    # ── Values correctness ─────────────────────────────────────────

    def test_rolling_mean_single_motor(self):
        """Rolling mean matches manual calculation."""
        df = _make_df({1: 5}, n_features=1)
        result = compute_rolling_stats(df, window_size=3)

        # sensor_1 values: 101, 102, 103, 104, 105
        # t=1: mean(101) = 101
        # t=2: mean(101, 102) = 101.5
        # t=3: mean(101, 102, 103) = 102
        # t=4: mean(102, 103, 104) = 103
        # t=5: mean(103, 104, 105) = 104
        expected = [101.0, 101.5, 102.0, 103.0, 104.0]
        np.testing.assert_array_almost_equal(
            result["sensor_1_mean"].values, expected
        )

    def test_rolling_std_single_motor(self):
        """Rolling std matches manual calculation."""
        df = _make_df({1: 3}, n_features=1)
        result = compute_rolling_stats(df, window_size=3)

        # t=1: std(101) = NaN (single value, ddof=1)
        assert np.isnan(result["sensor_1_std"].iloc[0])
        # t=2: std(101, 102) = 0.7071...
        assert result["sensor_1_std"].iloc[1] == pytest.approx(0.7071, abs=0.01)
        # t=3: std(101, 102, 103) = 1.0
        assert result["sensor_1_std"].iloc[2] == pytest.approx(1.0)

    def test_rolling_min_max(self):
        """Rolling min and max are correct."""
        df = _make_df({1: 5}, n_features=1)
        result = compute_rolling_stats(df, window_size=3)

        # t=3: window = [101, 102, 103] → min=101, max=103
        assert result["sensor_1_min"].iloc[2] == pytest.approx(101.0)
        assert result["sensor_1_max"].iloc[2] == pytest.approx(103.0)
        # t=5: window = [103, 104, 105] → min=103, max=105
        assert result["sensor_1_min"].iloc[4] == pytest.approx(103.0)
        assert result["sensor_1_max"].iloc[4] == pytest.approx(105.0)

    # ── Per-unit isolation ──────────────────────────────────────────

    def test_no_cross_unit_contamination(self):
        """Stats from one motor do not leak into another."""
        df = _make_df({1: 5, 2: 5}, n_features=1)
        result = compute_rolling_stats(df, window_size=3)

        # Motor 1 sensor_1: 101..105, Motor 2: 201..205
        # Mean at t=3 for motor 1 should be ~102, for motor 2 ~202
        m1 = result[result["unit_number"] == 1]
        m2 = result[result["unit_number"] == 2]

        assert m1["sensor_1_mean"].iloc[2] == pytest.approx(102.0)
        assert m2["sensor_1_mean"].iloc[2] == pytest.approx(202.0)

    # ── NaN handling for early cycles ───────────────────────────────

    def test_nan_for_insufficient_history(self):
        """First window_size - 1 rows per unit have NaN stats."""
        df = _make_df({1: 20}, n_features=1)
        result = compute_rolling_stats(df, window_size=5)

        # With min_periods=1, actually no NaN — stats computed from available
        # But window_size=5 means full window only at t>=5
        # min_periods=1 means all rows get values, so check the full window
        assert result["sensor_1_mean"].iloc[4] == pytest.approx(
            np.mean([101, 102, 103, 104, 105])
        )

    # ── Does not mutate input ──────────────────────────────────────

    def test_input_not_mutated(self):
        """Original DataFrame is not modified."""
        df = _make_df({1: 10}, n_features=2)
        original_cols = list(df.columns)
        original_shape = df.shape
        _ = compute_rolling_stats(df, window_size=5)

        assert list(df.columns) == original_cols
        assert df.shape == original_shape

    # ── Input validation ───────────────────────────────────────────

    def test_missing_unit_number_raises(self):
        """ValueError when unit_number is missing."""
        df = pd.DataFrame({"time": [1, 2], "sensor_1": [10, 20]})
        with pytest.raises(ValueError, match="unit_number"):
            compute_rolling_stats(df)

    def test_empty_stat_types_raises(self):
        """ValueError when stat_types is empty."""
        df = _make_df({1: 10})
        with pytest.raises(ValueError, match="No existen etadísticos para las variables Time Delay Embedding"):
            compute_rolling_stats(df, stat_types=[])

    def test_no_feature_columns_raises(self):
        """ValueError when only identifiers are present."""
        df = pd.DataFrame({"unit_number": [1, 1], "time": [1, 2], "rul": [1, 0]})
        with pytest.raises(ValueError, match="columnas"):
            compute_rolling_stats(df)


class TestComputeTrends:
    """Tests for compute_trends function."""

    # ── Output structure ────────────────────────────────────────────

    def test_default_adds_one_delta_column_per_feature(self):
        """Default delta_steps=[1] adds one _delta1 column per feature."""
        df = _make_df({1: 10}, n_features=2)
        result = compute_trends(df)

        assert "sensor_1_delta1" in result.columns
        assert "sensor_2_delta1" in result.columns
        assert result.shape[1] == df.shape[1] + 2

    def test_multiple_delta_steps(self):
        """delta_steps=[1,2,3] adds 3 columns per feature."""
        df = _make_df({1: 10}, n_features=2)
        result = compute_trends(df, delta_steps=[1, 2, 3])

        for s in [1, 2, 3]:
            assert f"sensor_1_delta{s}" in result.columns
            assert f"sensor_2_delta{s}" in result.columns
        # 2 sensors * 3 deltas = 6 new columns
        assert result.shape[1] == df.shape[1] + 6

    # ── Values correctness ─────────────────────────────────────────

    def test_delta1_values(self):
        """delta1 = current - previous cycle; first cycle is 0."""
        df = _make_df({1: 5}, n_features=1)
        result = compute_trends(df, delta_steps=[1])

        # sensor_1: 101, 102, 103, 104, 105
        # delta1: 0, 1, 1, 1, 1  (fillna(0.0) for first cycle)
        assert result["sensor_1_delta1"].iloc[0] == pytest.approx(0.0)
        assert result["sensor_1_delta1"].iloc[1] == pytest.approx(1.0)
        assert result["sensor_1_delta1"].iloc[4] == pytest.approx(1.0)

    def test_delta3_values(self):
        """delta3 = current - cycle 3 steps back; first 3 cycles are 0."""
        df = _make_df({1: 6}, n_features=1)
        result = compute_trends(df, delta_steps=[3])

        # sensor_1: 101, 102, 103, 104, 105, 106
        # delta3: 0, 0, 0, 3, 3, 3  (fillna(0.0) for first 3 cycles)
        assert result["sensor_1_delta3"].iloc[0] == pytest.approx(0.0)
        assert result["sensor_1_delta3"].iloc[2] == pytest.approx(0.0)
        assert result["sensor_1_delta3"].iloc[3] == pytest.approx(3.0)
        assert result["sensor_1_delta3"].iloc[5] == pytest.approx(3.0)

    def test_decreasing_sensor_produces_negative_delta(self):
        """Negative delta when sensor value decreases."""
        df = pd.DataFrame({
            "unit_number": [1, 1, 1],
            "time": [1, 2, 3],
            "rul": [2, 1, 0],
            "sensor_1": [100, 90, 80],
        })
        result = compute_trends(df, delta_steps=[1])

        assert result["sensor_1_delta1"].iloc[1] == pytest.approx(-10.0)
        assert result["sensor_1_delta1"].iloc[2] == pytest.approx(-10.0)

    # ── base_features_only ─────────────────────────────────────────

    def test_base_features_only_skips_stat_columns(self):
        """With base_features_only=True, deltas ignore _mean, _std, etc."""
        df = _make_df({1: 10}, n_features=1)
        df = compute_rolling_stats(df, window_size=5)
        result = compute_trends(df, delta_steps=[1], base_features_only=True)

        # Should have delta for sensor_1 but NOT for sensor_1_mean etc.
        assert "sensor_1_delta1" in result.columns
        assert "sensor_1_mean_delta1" not in result.columns
        assert "sensor_1_std_delta1" not in result.columns

    def test_base_features_only_false_computes_all(self):
        """With base_features_only=False, deltas are computed on everything."""
        df = _make_df({1: 10}, n_features=1)
        df = compute_rolling_stats(df, window_size=5)
        result = compute_trends(df, delta_steps=[1], base_features_only=False)

        # Should have delta for sensor_1 AND sensor_1_mean etc.
        assert "sensor_1_delta1" in result.columns
        assert "sensor_1_mean_delta1" in result.columns

    # ── Per-unit isolation ──────────────────────────────────────────

    def test_no_cross_unit_contamination(self):
        """Deltas are computed independently per motor."""
        df = _make_df({1: 5, 2: 5}, n_features=1)
        result = compute_trends(df, delta_steps=[1])

        # Motor 1: deltas are all 1.0 (101→102→103→104→105)
        m1 = result[result["unit_number"] == 1]
        assert m1["sensor_1_delta1"].iloc[1] == pytest.approx(1.0)

        # Motor 2: deltas are also 1.0 (201→202→203→204→205)
        m2 = result[result["unit_number"] == 2]
        assert m2["sensor_1_delta1"].iloc[1] == pytest.approx(1.0)

    def test_different_motors_different_absolute_values(self):
        """Delta values are independent, not mixed across motors."""
        df = pd.DataFrame({
            "unit_number": [1, 1, 2, 2],
            "time": [1, 2, 1, 2],
            "rul": [1, 0, 1, 0],
            "sensor_1": [100, 120, 500, 550],
        })
        result = compute_trends(df, delta_steps=[1])

        m1 = result[result["unit_number"] == 1]
        m2 = result[result["unit_number"] == 2]

        assert m1["sensor_1_delta1"].iloc[1] == pytest.approx(20.0)
        assert m2["sensor_1_delta1"].iloc[1] == pytest.approx(50.0)

    # ── Does not mutate input ──────────────────────────────────────

    def test_input_not_mutated(self):
        """Original DataFrame is not modified."""
        df = _make_df({1: 10}, n_features=2)
        original_cols = list(df.columns)
        original_shape = df.shape
        _ = compute_trends(df)

        assert list(df.columns) == original_cols
        assert df.shape == original_shape

    # ── Input validation ───────────────────────────────────────────

    def test_missing_unit_number_raises(self):
        """ValueError when unit_number is missing."""
        df = pd.DataFrame({"time": [1, 2], "sensor_1": [10, 20]})
        with pytest.raises(ValueError, match="columnas"):
            compute_trends(df)

    def test_negative_delta_step_raises(self):
        """ValueError when delta_steps contains non-positive values."""
        df = _make_df({1: 10})
        with pytest.raises(ValueError, match="positivos"):
            compute_trends(df, delta_steps=[0, -1])

    def test_no_feature_columns_raises(self):
        """ValueError when only identifiers are present."""
        df = pd.DataFrame({"unit_number": [1, 1], "time": [1, 2], "rul": [1, 0]})
        with pytest.raises(ValueError, match="No hay valores de sensores en el DataFrame"):
            compute_trends(df)
