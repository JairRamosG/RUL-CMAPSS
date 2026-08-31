"""
Data loading module for C-MAPSS dataset.

This module provides functions to load and validate the C-MAPSS dataset
from the NASA prognostics data repository.
"""

import os
from pathlib import Path
from typing import Tuple, List

import pandas as pd
import numpy as np


# Column names for C-MAPSS dataset
COLUMN_NAMES = [
    "unit_number",
    "time",
    "setting_1",
    "setting_2",
    "setting_3",
] + [f"sensor_{i}" for i in range(1, 22)]

# Valid subsets
VALID_SUBSETS = ["FD001", "FD002", "FD003", "FD004"]


def get_sensor_names() -> List[str]:
    """Return list of sensor column names.
    
    Returns:
        List of sensor column names (sensor_1 to sensor_21).
    """
    return [f"sensor_{i}" for i in range(1, 22)]


def get_setting_names() -> List[str]:
    """Return list of setting column names.
    
    Returns:
        List of setting column names (setting_1 to setting_3).
    """
    return ["setting_1", "setting_2", "setting_3"]


def load_cmapss(subset: str, data_dir: str = "datos") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load C-MAPSS dataset for a given subset.
    
    Args:
        subset: Subset identifier (FD001, FD002, FD003, or FD004).
        data_dir: Path to data directory relative to project root.
        
    Returns:
        Tuple of (train_df, test_df, rul_df) DataFrames.
        
    Raises:
        ValueError: If subset is not one of FD001, FD002, FD003, FD004.
        FileNotFoundError: If data files are not found.
    """
    # Validate subset
    if subset not in VALID_SUBSETS:
        raise ValueError(
            f"Invalid subset '{subset}'. Must be one of: {VALID_SUBSETS}"
        )
    
    # Get project root (assuming this file is in src/data/)
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / data_dir
    
    # File paths
    train_file = data_path / f"train_{subset}.txt"
    test_file = data_path / f"test_{subset}.txt"
    rul_file = data_path / f"RUL_{subset}.txt"
    
    # Check if files exist
    for file_path in [train_file, test_file, rul_file]:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {file_path}"
            )
    
    # Load data
    train_df = pd.read_csv(
        train_file,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES
    )
    
    test_df = pd.read_csv(
        test_file,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES
    )
    
    rul_df = pd.read_csv(
        rul_file,
        sep=r"\s+",
        header=None,
        names=["rul"]
    )
    
    # Check for null values
    null_counts = train_df.isnull().sum().sum()
    if null_counts > 0:
        print(f"Warning: {null_counts} null values found in training data")
    
    null_counts = test_df.isnull().sum().sum()
    if null_counts > 0:
        print(f"Warning: {null_counts} null values found in test data")
    
    if rul_df.empty:
        raise ValueError(f"RUL file is empty: {rul_file}")
    
    return train_df, test_df, rul_df


def get_data_info(subset: str) -> dict:
    """Get basic information about a C-MAPSS subset.
    
    Args:
        subset: Subset identifier (FD001, FD002, FD003, or FD004).
        
    Returns:
        Dictionary with dataset information.
    """
    train_df, test_df, rul_df = load_cmapss(subset)
    
    return {
        "subset": subset,
        "train_shape": train_df.shape,
        "test_shape": test_df.shape,
        "rul_shape": rul_df.shape,
        "num_machines_train": train_df["unit_number"].nunique(),
        "num_machines_test": test_df["unit_number"].nunique(),
        "columns": list(train_df.columns),
        "sensors": get_sensor_names(),
        "settings": get_setting_names(),
    }
