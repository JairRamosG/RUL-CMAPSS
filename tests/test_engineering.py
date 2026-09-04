"""
Tests for feature engineering module.
"""

import pytest
import pandas as pd
import numpy as np

from src.features.engineering import create_windows


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
        # Window starting at t=6: covers t=6..10, last real cycle=10, rul=0
        assert y[4] == pytest.approx(0.0)

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
        with pytest.raises(ValueError, match="window_size"):
            create_windows(df, window_size=0)

    def test_invalid_pad_strategy_raises(self):
        """ValueError for unsupported pad_strategy."""
        df = _make_df({1: 10})
        with pytest.raises(ValueError, match="pad_strategy"):
            create_windows(df, pad_strategy="zero")

    def test_no_feature_columns_raises(self):
        """ValueError when only identifiers are present."""
        df = pd.DataFrame({"unit_number": [1, 1], "time": [1, 2], "rul": [1, 0]})
        with pytest.raises(ValueError, match="No feature columns"):
            create_windows(df)

    # ── Does not mutate input ──────────────────────────────────────

    def test_input_not_mutated(self):
        """Original DataFrame is not modified."""
        df = _make_df({1: 10}, n_features=2)
        original_cols = list(df.columns)
        _ = create_windows(df, window_size=5)
        assert list(df.columns) == original_cols
