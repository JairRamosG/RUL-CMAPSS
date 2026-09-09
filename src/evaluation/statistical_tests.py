"""
Test estadísticos para la comparación de los modelos

Este módulo tiene funciónes para comparar desempeño de los modelos usando
los test estadísticos apropiados, tiene una selección automática basada en 
las características de los datos.

Lógica de selección para los test:
    1. Primero checar la normalidad como primer supuesto (Shapiro-Wilk)
    2. Checar Homosedasticidad como segundo supuesto (Levene)
    3. Seleccionar test paramétrico o no paramétrico dependiendo de los supuestos:
        - Normal + Homosedasticidad = paired t-test (two models) o ANOVA (multiple)
        - No se cumplen = Wilcoxon signed-rank (two) o Friedman + Nemenyi (multiple)

Referencia:
    Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets.
    Journal of Machine Learning Research, 7, 1-30.
"""

"""
Statistical tests for model comparison.

This module provides functions for comparing model performance using
appropriate statistical tests, with automatic test selection based on
data characteristics.

Reference:
    Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets.
    Journal of Machine Learning Research, 7, 1-30.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy import stats


@dataclass
class ComparisonResult:
    """Result of a statistical comparison between models.

    Attributes:
        test_name: Name of the statistical test used.
        statistic: Test statistic value.
        p_value: P-value from the test.
        significant: Whether the result is significant at alpha level.
        alpha: Significance level used.
        normality_p_values: P-values from normality tests.
        equal_variance: Whether variances are equal (if applicable).
        effect_size: Effect size measure value.
        effect_size_name: Name of the effect size metric.
        interpretation: Human-readable interpretation of the result.
    """
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    alpha: float
    normality_p_values: List[float] = field(default_factory=list)
    equal_variance: Optional[bool] = None
    effect_size: Optional[float] = None
    effect_size_name: Optional[str] = None
    interpretation: str = ""


def check_normality(scores: np.ndarray, alpha: float = 0.05) -> Tuple[bool, float]:
    """Check if scores follow a normal distribution using Shapiro-Wilk test."""
    if len(scores) < 3:
        return False, 0.0
    
    stat, p_value = stats.shapiro(scores)
    return p_value > alpha, float(p_value)


def check_variance_homogeneity(
    *groups: np.ndarray, 
    alpha: float = 0.05
) -> Tuple[bool, float]:
    """Check if score groups have equal variances using Levene's test."""
    if len(groups) < 2:
        return True, 1.0
        
    stat, p_value = stats.levene(*groups)
    return p_value > alpha, float(p_value)


def _cohens_d(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """Calculate Cohen's d effect size for two paired samples."""
    diff = scores_a - scores_b
    std_diff = np.std(diff, ddof=1)
    return float(np.mean(diff) / std_diff) if std_diff > 0 else 0.0


def _eta_squared_kruskal(groups: List[np.ndarray]) -> float:
    """Calculate eta-squared effect size for Kruskal-Wallis test."""
    h_stat, _ = stats.kruskal(*groups)
    n = sum(len(g) for g in groups)
    k = len(groups)
    return float((h_stat - k + 1) / (n - k)) if n > k else 0.0


def _interpret_effect_size(effect_size: float, name: str) -> str:
    """Interpret effect size magnitude."""
    magnitude = abs(effect_size)
    
    if name in ["cohens_d", "rank_biserial_r"]:
        if magnitude < 0.2:
            qualifier = "Negligible"
        elif magnitude < 0.5:
            qualifier = "Small"
        elif magnitude < 0.8:
            qualifier = "Medium"
        else:
            qualifier = "Large"
            
    elif name == "eta_squared":
        if magnitude < 0.01:
            qualifier = "Negligible"
        elif magnitude < 0.06:
            qualifier = "Small"
        elif magnitude < 0.14:
            qualifier = "Medium"
        else:
            qualifier = "Large"
    else:
        return f"Effect size ({name}): {effect_size:.3f}"

    return f"{qualifier} effect ({name}={effect_size:.3f})"


def compare_two_models(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alpha: float = 0.05,
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
    higher_is_better: bool = True,
) -> ComparisonResult:
    """Compare two paired models using the appropriate statistical test."""
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)
    
    if len(scores_a) != len(scores_b):
        raise ValueError("Score arrays must have the same length for paired comparison")
    if len(scores_a) < 2:
        raise ValueError("Need at least 2 paired observations for comparison")
    
    # Check normality of paired differences
    diff = scores_a - scores_b
    normal, normal_p = check_normality(diff, alpha)
    
    # Select parametric or non-parametric paired test
    if normal:
        stat, p_value = stats.ttest_rel(scores_a, scores_b)
        test_name = "Paired t-test"
        effect = _cohens_d(scores_a, scores_b)
        effect_name = "cohens_d"
    else:
        stat, p_value = stats.wilcoxon(scores_a, scores_b)
        test_name = "Wilcoxon signed-rank test"
        n = len(diff)
        effect = 1 - (2 * stat) / (n * (n + 1)) if n > 0 else 0.0
        effect_name = "rank_biserial_r"
    
    significant = bool(p_value < alpha)
    
    # Build human-readable interpretation
    if significant:
        mean_a, mean_b = np.mean(scores_a), np.mean(scores_b)
        a_is_better = mean_a > mean_b if higher_is_better else mean_a < mean_b
        
        winner = model_a_name if a_is_better else model_b_name
        loser = model_b_name if a_is_better else model_a_name
        interp = f"{winner} significantly outperforms {loser} (p={p_value:.4f} < {alpha})"
    else:
        interp = f"No significant difference between {model_a_name} and {model_b_name} (p={p_value:.4f} >= {alpha})"
    
    effect_interp = _interpret_effect_size(effect, effect_name)
    interp += f". {effect_interp}"
    
    return ComparisonResult(
        test_name=test_name,
        statistic=float(stat),
        p_value=float(p_value),
        significant=significant,
        alpha=alpha,
        normality_p_values=[normal_p],
        equal_variance=None,
        effect_size=float(effect),
        effect_size_name=effect_name,
        interpretation=interp,
    )


def compare_multiple_models(
    *model_scores: np.ndarray,
    alpha: float = 0.05,
    model_names: Optional[List[str]] = None,
    higher_is_better: bool = True,
) -> ComparisonResult:
    """Compare multiple models using One-way ANOVA or Kruskal-Wallis test."""
    if len(model_scores) < 2:
        raise ValueError("Need at least 2 models for comparison")
    
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(model_scores))]
    
    if len(model_names) != len(model_scores):
        raise ValueError("Number of model names must match number of score arrays")
    
    groups = [np.asarray(g) for g in model_scores]
    
    # Check assumptions across groups
    normality_results = [check_normality(g, alpha) for g in groups]
    all_normal = all(is_norm for is_norm, _ in normality_results)
    normality_p_values = [p for _, p in normality_results]
    
    equal_var, levene_p = check_variance_homogeneity(*groups, alpha=alpha)
    
    # Select test based on parametric assumptions
    if all_normal and equal_var:
        stat, p_value = stats.f_oneway(*groups)
        test_name = "One-way ANOVA"
        
        k = len(groups)
        n = sum(len(g) for g in groups)
        denom = (stat * (k - 1) + (n - k))
        effect = (stat * (k - 1)) / denom if denom > 0 else 0.0
        effect_name = "eta_squared"
    else:
        stat, p_value = stats.kruskal(*groups)
        test_name = "Kruskal-Wallis H test"
        effect = _eta_squared_kruskal(groups)
        effect_name = "eta_squared"
    
    significant = bool(p_value < alpha)
    
    # Build interpretation
    if significant:
        means = [np.mean(g) for g in groups]
        best_idx = int(np.argmax(means)) if higher_is_better else int(np.argmin(means))
        best_model = model_names[best_idx]
        
        interp = f"There is a significant difference between models (p={p_value:.4f} < {alpha}). "
        interp += f"{best_model} has the best mean performance ({means[best_idx]:.4f})."
    else:
        interp = f"No significant difference between models (p={p_value:.4f} >= {alpha})."
    
    effect_interp = _interpret_effect_size(effect, effect_name)
    interp += f" {effect_interp}"
    
    return ComparisonResult(
        test_name=test_name,
        statistic=float(stat),
        p_value=float(p_value),
        significant=significant,
        alpha=alpha,
        normality_p_values=normality_p_values,
        equal_variance=equal_var,
        effect_size=float(effect),
        effect_size_name=effect_name,
        interpretation=interp,
    )