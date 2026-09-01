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
    "configs/config_FD001.yml",
    "configs/config_FD002.yml",
    "configs/config_FD003.yml",
    "configs/config_FD004.yml",
]

# Required fields structure
REQUIRED_FIELDS = ["subset", "description", "data", "sensors", "models", "evaluation", "experiment", "profiling"]
REQUIRED_DATA = ["data_dir", "rul_max", "window_size", "overlap"]
REQUIRED_SENSORS = ["remove", "normalize"]
REQUIRED_EVALUATION = ["prediction_metrics", "cv_folds", "cv_group_by"]
REQUIRED_EXPERIMENT = ["random_seed", "n_jobs", "verbose", "mlflow_tracking_uri", "mlflow_experiment_name"]
REQUIRED_PROFILING = ["enabled", "metrics"]

# Valid model names
VALID_MODELS = ["associative_memory", "random_forest", "xgboost", "svr", "mlp", "cnn1d", "lstm"]

# Required profiling metrics
REQUIRED_PROFILING_METRICS = ["train_time", "inference_latency", "ram_train", "ram_inference"]


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
    def test_profiling_fields_present(self, config_file: str):
        """Test that all profiling fields are present."""
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        
        missing = []
        for field in REQUIRED_PROFILING:
            if field not in config.get("profiling", {}):
                missing.append(f"profiling.{field}")
        
        assert not missing, f"Missing profiling fields in {config_file}: {missing}"


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
    def test_overlap_is_int(self, config_file: str):
        """Test that overlap is an integer."""
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        
        assert isinstance(config.get("data", {}).get("overlap"), int), "data.overlap must be an integer"
    
    @pytest.mark.parametrize("config_file", CONFIG_FILES)
    def test_sensors_remove_is_list(self, config_file: str):
        """Test that sensors.remove is a list."""
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        
        assert isinstance(config.get("sensors", {}).get("remove"), list), "sensors.remove must be a list"
    
    @pytest.mark.parametrize("config_file", CONFIG_FILES)
    def test_normalize_is_bool(self, config_file: str):
        """Test that sensors.normalize is a boolean."""
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        
        assert isinstance(config.get("sensors", {}).get("normalize"), bool), "sensors.normalize must be a boolean"
    
    @pytest.mark.parametrize("config_file", CONFIG_FILES)
    def test_prediction_metrics_is_list(self, config_file: str):
        """Test that evaluation.prediction_metrics is a list."""
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        
        assert isinstance(config.get("evaluation", {}).get("prediction_metrics"), list), "evaluation.prediction_metrics must be a list"
    
    @pytest.mark.parametrize("config_file", CONFIG_FILES)
    def test_cv_folds_is_int(self, config_file: str):
        """Test that evaluation.cv_folds is an integer."""
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        
        assert isinstance(config.get("evaluation", {}).get("cv_folds"), int), "evaluation.cv_folds must be an integer"
    
    @pytest.mark.parametrize("config_file", CONFIG_FILES)
    def test_profiling_enabled_is_bool(self, config_file: str):
        """Test that profiling.enabled is a boolean."""
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        
        assert isinstance(config.get("profiling", {}).get("enabled"), bool), "profiling.enabled must be a boolean"
    
    @pytest.mark.parametrize("config_file", CONFIG_FILES)
    def test_profiling_metrics_is_list(self, config_file: str):
        """Test that profiling.metrics is a list."""
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        
        assert isinstance(config.get("profiling", {}).get("metrics"), list), "profiling.metrics must be a list"


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
    def test_all_seven_models_present(self, config_file: str):
        """Test that all 7 models are present."""
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        
        models = [m.get("name") for m in config.get("models", [])]
        
        for model in VALID_MODELS:
            assert model in models, f"Missing model: {model}"
    
    @pytest.mark.parametrize("config_file", CONFIG_FILES)
    def test_models_count(self, config_file: str):
        """Test that exactly 7 models are present."""
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        
        models = config.get("models", [])
        assert len(models) == 7, f"Expected 7 models, got {len(models)}"
    
    @pytest.mark.parametrize("config_file", CONFIG_FILES)
    def test_search_space_defined(self, config_file: str):
        """Test that search_space is defined for each model."""
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        
        for model in config.get("models", []):
            assert "search_space" in model, f"Model {model.get('name')} missing search_space"
            assert model["search_space"], f"Model {model.get('name')} has empty search_space"


class TestProfiling:
    """Tests for profiling configuration."""
    
    @pytest.mark.parametrize("config_file", CONFIG_FILES)
    def test_profiling_enabled(self, config_file: str):
        """Test that profiling is enabled."""
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        
        assert config.get("profiling", {}).get("enabled") is True, "profiling must be enabled"
    
    @pytest.mark.parametrize("config_file", CONFIG_FILES)
    def test_required_profiling_metrics(self, config_file: str):
        """Test that all required profiling metrics are present."""
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        
        metrics = config.get("profiling", {}).get("metrics", [])
        
        for metric in REQUIRED_PROFILING_METRICS:
            assert metric in metrics, f"Missing profiling metric: {metric}"


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
