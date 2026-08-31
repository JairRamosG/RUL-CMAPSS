"""
Tests for data loader module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.data.loader import load_cmapss, get_sensor_names, get_setting_names, get_data_info


class TestLoadCmapss:
    """Tests for load_cmapss function."""
    
    def test_load_fd001(self):
        """Test loading FD001 subset."""
        train, test, rul = load_cmapss("FD001")
        
        assert isinstance(train, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)
        assert isinstance(rul, pd.DataFrame)
        
        # Check shapes
        assert train.shape[1] == 26
        assert test.shape[1] == 26
        assert rul.shape[1] == 1
    
    def test_load_fd002(self):
        """Test loading FD002 subset."""
        train, test, rul = load_cmapss("FD002")
        
        assert train.shape[1] == 26
        assert test.shape[1] == 26
        assert rul.shape[1] == 1
    
    def test_load_fd003(self):
        """Test loading FD003 subset."""
        train, test, rul = load_cmapss("FD003")
        
        assert train.shape[1] == 26
        assert test.shape[1] == 26
        assert rul.shape[1] == 1
    
    def test_load_fd004(self):
        """Test loading FD004 subset."""
        train, test, rul = load_cmapss("FD004")
        
        assert train.shape[1] == 26
        assert test.shape[1] == 26
        assert rul.shape[1] == 1
    
    def test_column_names(self):
        """Test that column names are correct."""
        train, test, rul = load_cmapss("FD001")
        
        expected_columns = [
            "unit_number", "time",
            "setting_1", "setting_2", "setting_3",
        ] + [f"sensor_{i}" for i in range(1, 22)]
        
        assert list(train.columns) == expected_columns
        assert list(test.columns) == expected_columns
    
    def test_no_null_values(self):
        """Test that there are no null values in data."""
        train, test, rul = load_cmapss("FD001")
        
        assert train.isnull().sum().sum() == 0
        assert test.isnull().sum().sum() == 0
        assert rul.isnull().sum().sum() == 0
    
    def test_unit_number_first_column(self):
        """Test that unit_number is the first column."""
        train, test, rul = load_cmapss("FD001")
        
        assert train.columns[0] == "unit_number"
        assert test.columns[0] == "unit_number"
    
    def test_time_second_column(self):
        """Test that time is the second column."""
        train, test, rul = load_cmapss("FD001")
        
        assert train.columns[1] == "time"
        assert test.columns[1] == "time"
    
    def test_invalid_subset(self):
        """Test that invalid subset raises ValueError."""
        with pytest.raises(ValueError, match="Invalid subset"):
            load_cmapss("FD005")
    
    def test_case_insensitive_subset(self):
        """Test that subset input is case-insensitive."""
        train1, _, _ = load_cmapss("fd001")
        train2, _, _ = load_cmapss("FD001")
        train3, _, _ = load_cmapss("Fd001")
        
        assert train1.shape == train2.shape == train3.shape
    
    def test_whitespace_in_subset(self):
        """Test that whitespace in subset is trimmed."""
        train1, _, _ = load_cmapss("FD001")
        train2, _, _ = load_cmapss("  FD001  ")
        
        assert train1.shape == train2.shape
    
    def test_unit_numbers_are_integers(self):
        """Test that unit_number values are integers."""
        train, test, rul = load_cmapss("FD001")
        
        assert train["unit_number"].dtype in [np.int64, np.int32, int]
        assert test["unit_number"].dtype in [np.int64, np.int32, int]


class TestGetSensorNames:
    """Tests for get_sensor_names function."""
    
    def test_returns_list(self):
        """Test that function returns a list."""
        sensors = get_sensor_names()
        assert isinstance(sensors, list)
    
    def test_returns_21_sensors(self):
        """Test that function returns 21 sensor names."""
        sensors = get_sensor_names()
        assert len(sensors) == 21
    
    def test_sensor_names_format(self):
        """Test that sensor names are in correct format."""
        sensors = get_sensor_names()
        for i, sensor in enumerate(sensors, 1):
            assert sensor == f"sensor_{i}"


class TestGetSettingNames:
    """Tests for get_setting_names function."""
    
    def test_returns_list(self):
        """Test that function returns a list."""
        settings = get_setting_names()
        assert isinstance(settings, list)
    
    def test_returns_3_settings(self):
        """Test that function returns 3 setting names."""
        settings = get_setting_names()
        assert len(settings) == 3
    
    def test_setting_names_format(self):
        """Test that setting names are in correct format."""
        settings = get_setting_names()
        assert settings == ["setting_1", "setting_2", "setting_3"]


class TestGetDataInfo:
    """Tests for get_data_info function."""
    
    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        info = get_data_info("FD001")
        assert isinstance(info, dict)
    
    def test_contains_required_keys(self):
        """Test that dictionary contains required keys."""
        info = get_data_info("FD001")
        
        required_keys = [
            "subset", "train_shape", "test_shape", "rul_shape",
            "num_machines_train", "num_machines_test", "columns",
            "sensors", "settings"
        ]
        
        for key in required_keys:
            assert key in info
