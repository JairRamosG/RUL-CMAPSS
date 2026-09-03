"""
Tests for preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np

from src.data.preprocessing import create_groups, remove_constant_sensors


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
