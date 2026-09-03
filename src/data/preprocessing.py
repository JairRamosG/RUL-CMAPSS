"""
Preprocessing modules for RUL-CMAPSS pipeline.

Provides data preparation functions with strict data-leakage prevention:
GroupKFold splitting, sensor filtering, Piecewise Linear RUL, and
per-fold scaling.
"""

import numpy as np
import pandas as pd


def create_groups(df: pd.DataFrame) -> np.ndarray:
    """Extract unit IDs for GroupKFold splitting.

    Returns an array of shape (n_samples,) where each element is the
    unit_number of the corresponding row. This array is suitable for
    passing to sklearn.model_selection.GroupKFold.split().

    Args:
        df: DataFrame with a ``unit_number`` column.

    Returns:
        numpy array of unit IDs aligned with df rows.

    Raises:
        ValueError: If ``unit_number`` column is missing from df.
    """
    if "unit_number" not in df.columns:
        raise ValueError("DataFrame must contain a 'unit_number' column")

    return df["unit_number"].values
