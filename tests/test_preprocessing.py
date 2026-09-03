"""
Tests for preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np

from src.data.preprocessing import (
    create_groups,
    remove_constant_sensors,
    compute_piecewise_rul,
)


class TestCreateGroups:
    """Tests for create_groups function."""

    def test_returns_numpy_array(self):
        """Test that create_groups returns a numpy array."""
        df = pd.DataFrame({
            "unit_number": [1, 1, 2, 2, 3, 3],
            "time": [1, 2, 1, 2, 1, 2],
        })
        result = create_groups(df)
        assert isinstance(result, np.ndarray)

    def test_returns_correct_length(self):
        """Test that output length matches input length."""
        df = pd.DataFrame({
            "unit_number": [1, 1, 2, 2, 3, 3],
            "time": [1, 2, 1, 2, 1, 2],
        })
        result = create_groups(df)
        assert len(result) == len(df)

    def test_unique_units(self):
        """Test that no unit appears in multiple groups."""
        df = pd.DataFrame({
            "unit_number": [1, 1, 2, 2, 3, 3, 4, 4],
            "time": [1, 2, 1, 2, 1, 2, 1, 2],
        })
        result = create_groups(df)

        for unit in df["unit_number"].unique():
            mask = df["unit_number"] == unit
            group_values = result[mask]
            assert len(np.unique(group_values)) == 1, (
                f"Unit {unit} appears in multiple groups: {np.unique(group_values)}"
            )

    def test_preserves_row_order(self):
        """Test that groups align with original row indices."""
        df = pd.DataFrame({
            "unit_number": [10, 10, 20, 20, 30, 30],
            "time": [1, 2, 1, 2, 1, 2],
        })
        result = create_groups(df)

        # Rows 0-1 belong to unit 10, rows 2-3 to unit 20, rows 4-5 to unit 30
        assert result[0] == result[1]
        assert result[2] == result[3]
        assert result[4] == result[5]
        assert result[0] != result[2] != result[4]

    def test_single_unit(self):
        """Test with a single unit."""
        df = pd.DataFrame({
            "unit_number": [1, 1, 1, 1],
            "time": [1, 2, 3, 4],
        })
        result = create_groups(df)
        assert len(result) == 4
        assert len(np.unique(result)) == 1

    def test_many_units(self):
        """Test with 100 units (FD001 scale)."""
        units = list(range(1, 101))
        df = pd.DataFrame({
            "unit_number": np.repeat(units, 50),
            "time": np.tile(range(1, 51), 100),
        })
        result = create_groups(df)
        assert len(result) == 5000
        # Each unit maps to exactly one group
        for unit in units:
            mask = df["unit_number"] == unit
            assert len(np.unique(result[mask])) == 1


class TestRemoveConstantSensors:
    """Tests for remove_constant_sensors function."""

    def test_removes_correct_columns(self):
        """Test that specified sensor columns are removed."""
        df = pd.DataFrame({
            "unit_number": [1, 1],
            "sensor_1": [0.0, 0.0],
            "sensor_2": [1.0, 2.0],
            "sensor_3": [3.0, 4.0],
        })
        result = remove_constant_sensors(df, [1])

        assert "sensor_1" not in result.columns
        assert "sensor_2" in result.columns
        assert "sensor_3" in result.columns

    def test_removes_multiple_columns(self):
        """Test removing several sensors at once."""
        df = pd.DataFrame({
            "unit_number": [1, 1],
            "sensor_1": [0.0, 0.0],
            "sensor_5": [0.0, 0.0],
            "sensor_10": [0.0, 0.0],
            "sensor_11": [1.0, 2.0],
        })
        result = remove_constant_sensors(df, [1, 5, 10])

        assert "sensor_1" not in result.columns
        assert "sensor_5" not in result.columns
        assert "sensor_10" not in result.columns
        assert "sensor_11" in result.columns

    def test_preserves_non_sensor_columns(self):
        """Test that unit_number and time columns are preserved."""
        df = pd.DataFrame({
            "unit_number": [1, 1],
            "time": [1, 2],
            "setting_1": [0.5, 0.5],
            "sensor_1": [0.0, 0.0],
            "sensor_2": [1.0, 2.0],
        })
        result = remove_constant_sensors(df, [1])

        assert "unit_number" in result.columns
        assert "time" in result.columns
        assert "setting_1" in result.columns

    def test_ignores_nonexistent_sensors(self):
        """Test that nonexistent sensor IDs are silently ignored."""
        df = pd.DataFrame({
            "unit_number": [1, 1],
            "sensor_2": [1.0, 2.0],
        })
        result = remove_constant_sensors(df, [1, 5, 99])

        assert "sensor_2" in result.columns
        assert result.shape == df.shape

    def test_preserves_row_count(self):
        """Test that row count is unchanged after removal."""
        df = pd.DataFrame({
            "unit_number": [1, 1, 1],
            "sensor_1": [0.0, 0.0, 0.0],
            "sensor_2": [1.0, 2.0, 3.0],
        })
        result = remove_constant_sensors(df, [1])
        assert len(result) == len(df)

    def test_empty_removal_list(self):
        """Test with empty removal list returns all columns."""
        df = pd.DataFrame({
            "unit_number": [1, 1],
            "sensor_1": [0.0, 0.0],
            "sensor_2": [1.0, 2.0],
        })
        result = remove_constant_sensors(df, [])
        assert list(result.columns) == list(df.columns)


class TestComputePiecewiseRul:
    """Tests for compute_piecewise_rul function."""

    def test_adds_rul_column(self):
        """Test that rul column is added to the DataFrame."""
        df = pd.DataFrame({
            "unit_number": [1, 1, 1],
            "time": [1, 2, 3],
        })
        result = compute_piecewise_rul(df)
        assert "rul" in result.columns

    def test_rul_does_not_exceed_rul_max(self):
        """Test that RUL values never exceed rul_max."""
        df = pd.DataFrame({
            "unit_number": [1, 1, 1, 1, 1],
            "time": [1, 2, 3, 4, 5],
        })
        result = compute_piecewise_rul(df, rul_max=125)
        assert result["rul"].max() <= 125

    def test_rul_ends_at_zero(self):
        """Test that RUL equals 0 at the last cycle (failure)."""
        df = pd.DataFrame({
            "unit_number": [1, 1, 1],
            "time": [10, 20, 30],
        })
        result = compute_piecewise_rul(df)
        assert result["rul"].iloc[-1] == 0

    def test_rul_decreases_monotonically(self):
        """Test that RUL is monotonically non-increasing."""
        df = pd.DataFrame({
            "unit_number": [1, 1, 1, 1, 1],
            "time": [1, 2, 3, 4, 5],
        })
        result = compute_piecewise_rul(df)
        assert (result["rul"].diff().dropna() <= 0).all()

    def test_rul_values_correct(self):
        """Test exact RUL values for known input."""
        df = pd.DataFrame({
            "unit_number": [1, 1, 1, 1, 1],
            "time": [1, 2, 3, 4, 5],
        })
        result = compute_piecewise_rul(df, rul_max=125)
        # failure at cycle 5: raw RUL = [4, 3, 2, 1, 0]
        expected = [4, 3, 2, 1, 0]
        assert list(result["rul"]) == expected

    def test_rul_truncation(self):
        """Test that RUL is truncated when exceeding rul_max."""
        df = pd.DataFrame({
            "unit_number": [1, 1, 1, 1],
            "time": [1, 2, 3, 4],
        })
        result = compute_piecewise_rul(df, rul_max=2)
        # failure at 4: raw RUL = [3, 2, 1, 0], truncated to [2, 2, 1, 0]
        assert list(result["rul"]) == [2, 2, 1, 0]

    def test_multiple_units(self):
        """Test with two units having different lifespans."""
        df = pd.DataFrame({
            "unit_number": [1, 1, 1, 2, 2],
            "time": [1, 2, 3, 1, 2],
        })
        result = compute_piecewise_rul(df)
        # Unit 1 fails at 3: RUL = [2, 1, 0]
        # Unit 2 fails at 2: RUL = [1, 0]
        assert list(result["rul"]) == [2, 1, 0, 1, 0]

    def test_preserves_original_columns(self):
        """Test that original columns are not modified."""
        df = pd.DataFrame({
            "unit_number": [1, 1],
            "time": [1, 2],
            "sensor_2": [10.0, 20.0],
        })
        result = compute_piecewise_rul(df)
        assert "unit_number" in result.columns
        assert "time" in result.columns
        assert "sensor_2" in result.columns

    def test_missing_column_raises(self):
        """Test that missing required columns raise ValueError."""
        df = pd.DataFrame({"sensor_1": [1, 2]})
        with pytest.raises(ValueError):
            compute_piecewise_rul(df)

    def test_custom_rul_max(self):
        """Test with different rul_max values."""
        df = pd.DataFrame({
            "unit_number": [1, 1, 1],
            "time": [1, 2, 3],
        })
        result = compute_piecewise_rul(df, rul_max=100)
        assert result["rul"].max() <= 100
        assert result["rul"].iloc[-1] == 0
