"""
Evaluation metrics and resource profiling module.

This module provides functions for evaluating model performance
and profiling computational resource usage.
"""

import time
import functools
from typing import Dict, Any
from contextlib import contextmanager

import psutil
import numpy as np


class ResourceProfiler:
    """Context manager for profiling resource usage.
    
    Captures elapsed time and peak memory usage during execution.
    
    Attributes:
        elapsed_time: Time elapsed in seconds.
        peak_memory_mb: Peak memory usage in megabytes.
    """
    
    def __init__(self):
        """Initialize the profiler."""
        self.elapsed_time: float = 0.0
        self.peak_memory_mb: float = 0.0
        self._start_time: float = 0.0
        self._start_memory: float = 0.0
    
    def __enter__(self):
        """Start profiling."""
        self._start_time = time.time()
        self._start_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        self.peak_memory_mb = self._start_memory
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop profiling and calculate final metrics."""
        self.elapsed_time = time.time() - self._start_time
        current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        self.peak_memory_mb = max(self.peak_memory_mb, current_memory)
        return False


@contextmanager
def profile_resource_usage():
    """Context manager for profiling resource usage.
    
    Usage:
        with profile_resource_usage() as profiler:
            # Your code here
            pass
        print(f"Time: {profiler.elapsed_time:.2f}s")
        print(f"Peak memory: {profiler.peak_memory_mb:.2f}MB")
    
    Yields:
        ResourceProfiler instance with elapsed_time and peak_memory_mb.
    """
    profiler = ResourceProfiler()
    with profiler:
        yield profiler


def get_system_info() -> Dict[str, Any]:
    """Get system information for profiling context.
    
    Returns:
        Dictionary with system information.
    """
    cpu_info = {
        "cpu_count": psutil.cpu_count(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
    }
    
    memory = psutil.virtual_memory()
    memory_info = {
        "total_gb": memory.total / (1024 ** 3),
        "available_gb": memory.available / (1024 ** 3),
        "percent_used": memory.percent,
    }
    
    return {
        "cpu": cpu_info,
        "memory": memory_info,
        "platform": psutil.os.name,
    }


# Profiling decorator
def profile_function(func):
    """Decorator to profile function execution.
    
    Usage:
        @profile_function
        def my_function():
            pass
        
        result, profiler = my_function()
        print(f"Time: {profiler.elapsed_time:.2f}s")
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with profile_resource_usage() as profiler:
            result = func(*args, **kwargs)
        return result, profiler
    return wrapper


# Evaluation metrics
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error.
    
    Args:
        y_true: True values.
        y_pred: Predicted values.
        
    Returns:
        RMSE value.
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error.
    
    Args:
        y_true: True values.
        y_pred: Predicted values.
        
    Returns:
        MAE value.
    """
    return np.mean(np.abs(y_true - y_pred))


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate NASA Scoring Function.
    
    The NASA scoring function penalizes overestimation more severely
    than underestimation, as overestimation can lead to catastrophic failure.
    
    Args:
        y_true: True RUL values.
        y_pred: Predicted RUL values.
        
    Returns:
        NASA score (lower is better).
    """
    d = y_pred - y_true
    scores = np.where(
        d < 0,
        np.exp(-d / 13) - 1,  # Overestimation
        np.exp(d / 10) - 1    # Underestimation
    )
    return np.sum(scores)
