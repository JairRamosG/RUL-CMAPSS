"""
Tests for YAML configuration files.

This module validates that all configuration files for C-MAPSS subsets
are correctly structured and contain valid parameters.
"""

import pytest
import yaml
from pathlib import Path
from typing import Dict, Any


# Configuration files to test
CONFIG_FILES = [
    "configs/config_FD001.yaml",
    "configs/config_FD002.yaml",
    "configs/config_FD003.yaml",
    "configs/config_FD004.yaml",
]

# Required fields structure
REQUIRED_FIELDS = ["subset", "description", "data", "sensors", "models", "evaluation", "experiment"]
REQUIRED_DATA = ["data_dir", "rul_max", "window_size"]
REQUIRED_SENSORS = ["remove"]
REQUIRED_EVALUATION = ["cv_folds"]
REQUIRED_EXPERIMENT = ["random_seed", "mlflow_tracking_uri", "mlflow_experiment_name"]
REQUIRED_STATISTICAL_TEST = ["alpha", "higher_is_better", "force_test", "metrics_to_compare"]

# Valid model names
VALID_MODELS = ["associative_memory", "random_forest", "xgboost", "lightgbm", "svr", "mlp", "cnn1d", "lstm"]


class TestYAMLStructure:
    """Tests for YAML file structure and loading."""
    
    @pytest.mark.parametrize("config_file", CONFIG_FILES)
    def test_yaml_loads_without_errors(self, config_file: str):
        """Test that each YAML file loads without errors."""
        path = Path(config_file)
        assert path.exists(), f"Config file not found: {config_file}"
        
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        
        assert config is not None, f"Config file is empty: {config_file}"
    
    @pytest.mark.parametrize("config_file", CONFIG_FILES)
    def test_required_fields_present(self, config_file: str):
        """Test that all required fields are present."""
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        
        missing = []
        for field in REQUIRED_FIELDS:
            if field not in config:
                missing.append(field)
        
        assert not missing, f"Missing fields in {config_file}: {missing}"
    
    @pytest.mark.parametrize("config_file", CONFIG_FILES)
    def test_data_fields_present(self, config_file: str):
        """Test that all data fields are present."""
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        
        missing = []
        for field in REQUIRED_DATA:
            if field not in config.get("data", {}):
                missing.append(f"data.{field}")
        
        assert not missing, f"Missing data fields in {config_file}: {missing}"
    
    @pytest.mark.parametrize("config_file", CONFIG_FILES)
    def test_sensors_fields_present(self, config_file: str):
        """Test that all sensor fields are present."""
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        
        missing = []
        for field in REQUIRED_SENSORS:
            if field not in config.get("sensors", {}):
                missing.append(f"sensors.{field}")
        
        assert not missing, f"Missing sensor fields in {config_file}: {missing}"
    
    @pytest.mark.parametrize("config_file", CONFIG_FILES)
    def test_evaluation_fields_present(self, config_file: str):
        """Test that all evaluation fields are present."""
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        
        missing = []
        for field in REQUIRED_EVALUATION:
            if field not in config.get("evaluation", {}):
                missing.append(f"evaluation.{field}")
        
        assert not missing, f"Missing evaluation fields in {config_file}: {missing}"
    
    @pytest.mark.parametrize("config_file", CONFIG_FILES)
    def test_experiment_fields_present(self, config_file: str):
        """Test that all experiment fields are present."""
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        
        missing = []
        for field in REQUIRED_EXPERIMENT:
            if field not in config.get("experiment", {}):
                missing.append(f"experiment.{field}")
        
        assert not missing, f"Missing experiment fields in {config_file}: {missing}"
    
    @pytest.mark.parametrize("config_file", CONFIG_FILES)
    def test_statistical_test_fields_present(self, config_file: str):
        """Test that all statistical_test fields are present."""
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        
        stat_cfg = config.get("evaluation", {}).get("statistical_test", {})
        missing = []
        for field in REQUIRED_STATISTICAL_TEST:
            if field not in stat_cfg:
                missing.append(f"evaluation.statistical_test.{field}")
        
        assert not missing, f"Missing statistical_test fields in {config_file}: {missing}"


class TestDataTypes:
    """Tests for correct data types in configuration."""
    
    @pytest.mark.parametrize("config_file", CONFIG_FILES)
    def test_subset_is_string(self, config_file: str):
        """Test that subset is a string."""
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        
        assert isinstance(config.get("subset"), str), "subset must be a string"
    
    @pytest.mark.parametrize("config_file", CONFIG_FILES)
    def test_description_is_string(self, config_file: str):
        """Test that description is a string."""
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        
        assert isinstance(config.get("description"), str), "description must be a string"
    
    @pytest.mark.parametrize("config_file", CONFIG_FILES)
    def test_rul_max_is_int(self, config_file: str):
        """Test that rul_max is an integer."""
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        
        assert isinstance(config.get("data", {}).get("rul_max"), int), "data.rul_max must be an integer"
    
    @pytest.mark.parametrize("config_file", CONFIG_FILES)
    def test_window_size_is_int(self, config_file: str):
        """Test that window_size is an integer."""
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        
        assert isinstance(config.get("data", {}).get("window_size"), int), "data.window_size must be an integer"
    
    @pytest.mark.parametrize("config_file", CONFIG_FILES)
    def test_sensors_remove_is_list(self, config_file: str):
        """Test that sensors.remove is a list."""
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        
        assert isinstance(config.get("sensors", {}).get("remove"), list), "sensors.remove must be a list"
    
    @pytest.mark.parametrize("config_file", CONFIG_FILES)
    def test_cv_folds_is_int(self, config_file: str):
        """Test that evaluation.cv_folds is an integer."""
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        
        assert isinstance(config.get("evaluation", {}).get("cv_folds"), int), "evaluation.cv_folds must be an integer"
    
    @pytest.mark.parametrize("config_file", CONFIG_FILES)
    def test_metrics_to_compare_is_list(self, config_file: str):
        """Test that evaluation.statistical_test.metrics_to_compare is a list."""
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        
        stat_cfg = config.get("evaluation", {}).get("statistical_test", {})
        assert isinstance(stat_cfg.get("metrics_to_compare"), list), "evaluation.statistical_test.metrics_to_compare must be a list"


class TestValueRanges:
    """Tests for valid value ranges."""
    
    @pytest.mark.parametrize("config_file", CONFIG_FILES)
    def test_rul_max_range(self, config_file: str):
        """Test that rul_max is within valid range."""
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        
        rul_max = config.get("data", {}).get("rul_max", 0)
        assert 100 <= rul_max <= 200, f"data.rul_max must be between 100 and 200, got {rul_max}"
    
    @pytest.mark.parametrize("config_file", CONFIG_FILES)
    def test_window_size_range(self, config_file: str):
        """Test that window_size is within valid range."""
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        
        window_size = config.get("data", {}).get("window_size", 0)
        assert 10 <= window_size <= 100, f"data.window_size must be between 10 and 100, got {window_size}"
    
    @pytest.mark.parametrize("config_file", CONFIG_FILES)
    def test_sensors_remove_range(self, config_file: str):
        """Test that sensors to remove are within valid range."""
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        
        for sensor in config.get("sensors", {}).get("remove", []):
            assert isinstance(sensor, int), f"Sensor must be an integer, got {type(sensor)}"
            assert 1 <= sensor <= 21, f"Sensor must be between 1 and 21, got {sensor}"
    
    @pytest.mark.parametrize("config_file", CONFIG_FILES)
    def test_cv_folds_range(self, config_file: str):
        """Test that cv_folds is within valid range."""
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        
        cv_folds = config.get("evaluation", {}).get("cv_folds", 0)
        assert 2 <= cv_folds <= 10, f"evaluation.cv_folds must be between 2 and 10, got {cv_folds}"


class TestModels:
    """Tests for model configuration."""
    
    @pytest.mark.parametrize("config_file", CONFIG_FILES)
    def test_all_models_present(self, config_file: str):
        """Test that all 8 models are present."""
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        
        models = [m.get("name") for m in config.get("models", [])]
        
        for model in VALID_MODELS:
            assert model in models, f"Missing model: {model}"
    
    @pytest.mark.parametrize("config_file", CONFIG_FILES)
    def test_models_count(self, config_file: str):
        """Test that exactly 8 models are present."""
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        
        models = config.get("models", [])
        assert len(models) == 8, f"Expected 8 models, got {len(models)}"
    

class TestDescriptions:
    """Tests for description quality."""
    
    @pytest.mark.parametrize("config_file", CONFIG_FILES)
    def test_description_minimum_length(self, config_file: str):
        """Test that description is at least 50 characters."""
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        
        description = config.get("description", "")
        assert len(description) >= 50, f"Description too short ({len(description)} chars), minimum is 50"
    
    @pytest.mark.parametrize("config_file", CONFIG_FILES)
    def test_subset_matches_filename(self, config_file: str):
        """Test that subset matches filename."""
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        
        expected_subset = Path(config_file).stem.replace("config_", "")
        actual_subset = config.get("subset", "")
        
        assert actual_subset == expected_subset, f"Subset mismatch: expected {expected_subset}, got {actual_subset}"
