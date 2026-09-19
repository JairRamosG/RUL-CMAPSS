"""
Test estadísticos sobre grupos pareados.

This module provides functions for comparing model performance using
appropriate statistical tests under a block design (same folds/iterations).

The module is agnostic to the variable being tested - it works with
RMSE, inference time, training time, memory usage, or any numeric array.

Test selection logic:
    1. Check normality of two-way ANOVA residuals (Shapiro-Wilk)
        e_ij = y_ij - mean(block_i) - mean(model_j) + grand_mean
    2. Select parametric or non-parametric test:
        - Normal → Repeated Measures ANOVA + Bonferroni post-hoc
        - Non-normal → Friedman + Nemenyi post-hoc

Reference:
    Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets.
    Journal of Machine Learning Research, 7, 1-30.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy import stats


@dataclass
class ComparisonResult:
    """Result of a statistical comparison between models.

    Attributes:
        test_used: Name of the statistical test executed.
        omnibus_p_value: P-value from the omnibus test.
        significant: Whether the result is significant (p < alpha).
        reason: Justification for why this test was chosen.
        post_hoc_matrix: Dictionary with pairwise p-values (if applicable).
        rankings: Dictionary with average ranks per model (rank 1.0 is best).
    """
    test_used: str
    omnibus_p_value: float
    significant: bool
    reason: str
    post_hoc_matrix: Optional[dict] = None
    rankings: Optional[dict] = None

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary for MLflow export."""
        return {
            "test_used": self.test_used,
            "omnibus_p_value": self.omnibus_p_value,
            "significant": self.significant,
            "reason": self.reason,
            "post_hoc_matrix": self.post_hoc_matrix,
            "rankings": self.rankings,
        }


def check_normality(
    data: np.ndarray, alpha: float = 0.05, subsample_threshold: int = 5000
) -> Tuple[bool, float, str]:
    """Check if data follows a normal distribution using Shapiro-Wilk test.

    If N > subsample_threshold, applies deterministic subsampling (seed=42)
    to prevent over-sensitivity to trivial deviations from normality.

    Args:
        data: Array of numeric values to test.
        alpha: Significance level for the test.
        subsample_threshold: Maximum sample size before subsampling.

    Returns:
        Tuple of (is_normal, p_value, reason).
    """
    data = np.asarray(data).flatten()

    if len(data) < 3:
        return False, 0.0, "Insufficient data (N < 3)"

    # Subsampling for large N to avoid over-sensitivity
    if len(data) > subsample_threshold:
        rng = np.random.default_rng(seed=42)
        indices = rng.choice(len(data), size=subsample_threshold, replace=False)
        data_sub = data[indices]
        stat, p_value = stats.shapiro(data_sub)
        reason = (
            f"Shapiro-Wilk on subsample N={subsample_threshold} "
            f"(original N={len(data)}): p={p_value:.4f}"
        )
    else:
        stat, p_value = stats.shapiro(data)
        reason = f"Shapiro-Wilk (N={len(data)}): p={p_value:.4f}"

    is_normal = p_value > alpha
    return is_normal, float(p_value), reason


def friedman_test(
    *groups: np.ndarray,
    alpha: float = 0.05,
    model_names: Optional[List[str]] = None,
    higher_is_better: bool = False,
) -> ComparisonResult:
    """Perform Friedman test with Nemenyi post-hoc analysis.

    Non-parametric test for comparing k >= 3 paired groups.
    If significant, performs Nemenyi post-hoc test to identify
    which pairs of models differ significantly.

    Args:
        *groups: Variable number of arrays, one per model (same length).
        alpha: Significance level.
        model_names: Optional list of model names.
        higher_is_better: Whether higher values are better (default False: lower is better).

    Returns:
        ComparisonResult with test details, post-hoc matrix, and rankings.
    """
    if len(groups) < 2:
        raise ValueError("Need at least 2 groups for comparison")

    groups_arr = [np.asarray(g) for g in groups]
    n_models = len(groups_arr)
    n_obs = len(groups_arr[0])

    # Validate all groups have same length
    for i, g in enumerate(groups_arr):
        if len(g) != n_obs:
            raise ValueError(
                f"All groups must have same length. Group 0 has {n_obs}, "
                f"group {i} has {len(g)}"
            )

    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(n_models)]

    # Friedman test (omnibus)
    stat, p_value = stats.friedmanchisquare(*groups_arr)
    significant = bool(p_value < alpha)

    # Calculate average ranks (rank 1.0 is best performing)
    stacked = np.column_stack(groups_arr)
    rank_input = -stacked if higher_is_better else stacked
    ranks = np.apply_along_axis(stats.rankdata, 1, rank_input)
    avg_ranks = np.mean(ranks, axis=0)
    rankings = {name: float(rank) for name, rank in zip(model_names, avg_ranks)}

    # Post-hoc Nemenyi test if significant
    post_hoc_matrix = None
    if significant and n_models >= 3:
        try:
            import scikit_posthocs as sp

            # scikit-posthocs expects matrix [n_obs x n_models]
            nemenyi_result = sp.posthoc_nemenyi_friedman(stacked)
            post_hoc_matrix = {}
            for i, name_i in enumerate(model_names):
                post_hoc_matrix[name_i] = {}
                for j, name_j in enumerate(model_names):
                    post_hoc_matrix[name_i][name_j] = float(nemenyi_result.iloc[i, j])
        except ImportError:
            post_hoc_matrix = {
                "error": "scikit-posthocs not installed. Install with: uv add scikit-posthocs"
            }

    reason = f"Friedman test (non-parametric): Q={stat:.4f}, p={p_value:.4f}."

    return ComparisonResult(
        test_used="Friedman + Nemenyi",
        omnibus_p_value=float(p_value),
        significant=significant,
        reason=reason,
        post_hoc_matrix=post_hoc_matrix,
        rankings=rankings,
    )


def repeated_measures_anova(
    *groups: np.ndarray,
    alpha: float = 0.05,
    model_names: Optional[List[str]] = None,
    higher_is_better: bool = False,
) -> ComparisonResult:
    """Perform Repeated Measures ANOVA with Bonferroni post-hoc analysis.

    Parametric test for comparing k >= 3 paired groups.
    Includes verification of sphericity (Greenhouse-Geisser epsilon).
    If significant, performs paired t-tests with Bonferroni correction.

    Args:
        *groups: Variable number of arrays, one per model (same length).
        alpha: Significance level.
        model_names: Optional list of model names.
        higher_is_better: Whether higher values are better (default False: lower is better).

    Returns:
        ComparisonResult with test details, post-hoc matrix, and rankings.
    """
    if len(groups) < 2:
        raise ValueError("Need at least 2 groups for comparison")

    groups_arr = [np.asarray(g) for g in groups]
    n_models = len(groups_arr)
    n_obs = len(groups_arr[0])

    # Validate all groups have same length
    for i, g in enumerate(groups_arr):
        if len(g) != n_obs:
            raise ValueError(
                f"All groups must have same length. Group 0 has {n_obs}, "
                f"group {i} has {len(g)}"
            )

    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(n_models)]

    # Calculate average ranks (rank 1.0 is best performing)
    stacked = np.column_stack(groups_arr)
    rank_input = -stacked if higher_is_better else stacked
    ranks = np.apply_along_axis(stats.rankdata, 1, rank_input)
    avg_ranks = np.mean(ranks, axis=0)
    rankings = {name: float(rank) for name, rank in zip(model_names, avg_ranks)}

    # Repeated Measures ANOVA using pingouin
    try:
        import pandas as pd
        import pingouin as pg

        # Create DataFrame in long format: [subject, model, value]
        data_list = []
        for fold_idx in range(n_obs):
            for group, name in zip(groups_arr, model_names):
                data_list.append(
                    {"subject": fold_idx, "model": name, "value": float(group[fold_idx])}
                )

        df = pd.DataFrame(data_list)

        # Perform RM-ANOVA with sphericity evaluation
        aov = pg.rm_anova(dv="value", within="model", subject="subject", data=df, correction=True)
        f_stat = float(aov["F"].values[0])
        df_between = float(aov["ddof1"].values[0])
        df_error = float(aov["ddof2"].values[0])
        eps = float(aov["eps"].values[0]) if "eps" in aov.columns else 1.0

        # Use Greenhouse-Geisser corrected p-value if sphericity is violated (eps < 0.75)
        p_corr_col = None
        for col in ["p_GG_corr", "p-GG-corr", "p_corr", "p-corr"]:
            if col in aov.columns:
                p_corr_col = col
                break

        if p_corr_col and eps < 0.75:
            p_value = float(aov[p_corr_col].values[0])
            sphericity_note = f"Greenhouse-Geisser correction applied (eps={eps:.4f})"
        else:
            p_value = float(aov["p_unc"].values[0])
            sphericity_note = f"Sphericity assumed (eps={eps:.4f})"

        significant = bool(p_value < alpha)

        # Post-hoc Bonferroni-corrected paired t-tests if significant
        post_hoc_matrix = None
        if significant:
            # Pre-initialize symmetric matrix to avoid overwriting entries
            post_hoc_matrix = {
                name_i: {name_j: 1.0 for name_j in model_names}
                for name_i in model_names
            }
            n_comparisons = n_models * (n_models - 1) // 2

            for i in range(n_models):
                for j in range(i + 1, n_models):
                    name_i = model_names[i]
                    name_j = model_names[j]
                    _, p_val = stats.ttest_rel(groups_arr[i], groups_arr[j])
                    p_corrected = float(min(p_val * n_comparisons, 1.0))
                    post_hoc_matrix[name_i][name_j] = p_corrected
                    post_hoc_matrix[name_j][name_i] = p_corrected

        reason = (
            f"Repeated Measures ANOVA: F={f_stat:.4f}, df=({df_between:.1f}, {df_error:.1f}), "
            f"p={p_value:.4f}. {sphericity_note}."
        )

    except (ImportError, ZeroDivisionError, ValueError, Exception) as e:
        # Fallback: one-way ANOVA if pingouin is not available or degrees of freedom too small (e.g. N=2 in dry-run)
        stat, p_value = stats.f_oneway(*groups_arr)
        significant = bool(p_value < alpha)
        post_hoc_matrix = {
            "note": f"Pingouin RM-ANOVA fallback ({type(e).__name__}): using one-way ANOVA"
        }
        reason = (
            f"ANOVA fallback ({type(e).__name__}): F={stat:.4f}, "
            f"p={p_value:.4f}."
        )

    return ComparisonResult(
        test_used="Repeated Measures ANOVA",
        omnibus_p_value=float(p_value),
        significant=significant,
        reason=reason,
        post_hoc_matrix=post_hoc_matrix,
        rankings=rankings,
    )


def compare_multiple_models(
    *groups: np.ndarray,
    alpha: float = 0.05,
    model_names: Optional[List[str]] = None,
    higher_is_better: bool = False,
    force_test: Optional[str] = None,
) -> ComparisonResult:
    """Compare multiple models using the appropriate statistical test.

    Automatically selects between Repeated Measures ANOVA (parametric)
    and Friedman + Nemenyi (non-parametric) based on normality of residuals.

    The module is agnostic to the variable being tested - works with
    RMSE, inference time, training time, memory usage, or any numeric array.

    Args:
        *groups: Variable number of arrays with results per fold for each model.
        alpha: Significance level.
        model_names: Optional list of model names.
        higher_is_better: Whether higher values are better (default False: lower is better).
        force_test: Force specific test ("friedman" or "rm_anova").

    Returns:
        ComparisonResult with test details, post-hoc matrix, and rankings.
    """
    if len(groups) < 2:
        raise ValueError("Need at least 2 models for comparison")

    groups_arr = [np.asarray(g) for g in groups]
    n_models = len(groups_arr)

    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(n_models)]

    if len(model_names) != n_models:
        raise ValueError(
            f"Number of model names ({len(model_names)}) must match "
            f"number of groups ({n_models})"
        )

    # Check normality of residuals in Randomized Complete Block Design (RCBD)
    # Formula: e_ij = y_ij - mean(block_i) - mean(model_j) + grand_mean
    stacked = np.column_stack(groups_arr)
    block_means = np.mean(stacked, axis=1, keepdims=True)
    model_means = np.mean(stacked, axis=0, keepdims=True)
    grand_mean = np.mean(stacked)

    residuals = stacked - block_means - model_means + grand_mean
    residuals_flat = residuals.flatten()
    is_normal, normal_p, normal_reason = check_normality(residuals_flat, alpha)

    # Force test if specified
    if force_test == "friedman":
        return friedman_test(
            *groups_arr, alpha=alpha, model_names=model_names, higher_is_better=higher_is_better
        )
    elif force_test == "rm_anova":
        return repeated_measures_anova(
            *groups_arr, alpha=alpha, model_names=model_names, higher_is_better=higher_is_better
        )
    elif force_test is not None:
        raise ValueError(
            f"Invalid force_test value: '{force_test}'. "
            f"Must be 'friedman', 'rm_anova', or None."
        )

    # Auto-select based on normality of two-way residuals
    if is_normal:
        result = repeated_measures_anova(
            *groups_arr, alpha=alpha, model_names=model_names, higher_is_better=higher_is_better
        )
        result.reason = (
            f"Normal residuals detected (Shapiro p={normal_p:.4f}). "
            f"Using parametric test: {result.test_used}. {result.reason}"
        )
    else:
        result = friedman_test(
            *groups_arr, alpha=alpha, model_names=model_names, higher_is_better=higher_is_better
        )
        result.reason = (
            f"Non-normal residuals detected (Shapiro p={normal_p:.4f}). "
            f"Using non-parametric test: {result.test_used}. {result.reason}"
        )

    return result
