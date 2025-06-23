"""
Statistical Analysis Module for Bird Classification Benchmark

This module provides statistical methods for evaluating benchmark results
and calculating statistical requirements for valid comparisons.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import logging

# Fallback implementation for statistical functions
def norm_ppf(p):
    """Inverse normal CDF approximation using Beasley-Springer-Moro algorithm"""
    if p <= 0 or p >= 1:
        raise ValueError("p must be between 0 and 1")
    
    # Constants for approximation
    a = [0, -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [0, -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01]
    
    if p < 0.5:
        q = np.sqrt(-2.0 * np.log(p))
    else:
        q = np.sqrt(-2.0 * np.log(1.0 - p))
    
    t = q - (a[1] + a[2]*q + a[3]*q**2 + a[4]*q**3 + a[5]*q**4 + a[6]*q**5) / (1 + b[1]*q + b[2]*q**2 + b[3]*q**3 + b[4]*q**4 + b[5]*q**5)
    
    return -t if p < 0.5 else t

def norm_cdf(x):
    """Normal CDF approximation"""
    return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi)))

def t_ppf(p, df):
    """t-distribution PPF approximation"""
    return norm_ppf(p) * np.sqrt(df / (df - 2)) if df > 2 else norm_ppf(p)

def ttest_rel_simple(a, b):
    """Simple paired t-test"""
    diff = np.array(a) - np.array(b)
    n = len(diff)
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    t_stat = mean_diff / (std_diff / np.sqrt(n))
    # Approximate p-value using normal distribution for simplicity
    p_val = 2 * (1 - norm_cdf(abs(t_stat)))
    return t_stat, p_val

logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """Statistical analyzer for benchmark results."""
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize the statistical analyzer.
        
        Args:
            confidence_level: Confidence level for statistical tests (default 0.95)
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def calculate_confidence_interval(self, data: List[float]) -> Tuple[float, float]:
        """
        Calculate confidence interval for a list of values.
        
        Args:
            data: List of numerical values
            
        Returns:
            Tuple containing (lower_bound, upper_bound) of confidence interval
        """
        data = np.array(data)
        n = len(data)
        mean = np.mean(data)
        std_err = np.std(data, ddof=1) / np.sqrt(n)  # Standard error
        
        # Use t-distribution for small samples
        if n < 30:
            t_val = t_ppf(1 - self.alpha/2, n-1)
            margin_error = t_val * std_err
        else:
            z_val = norm_ppf(1 - self.alpha/2)
            margin_error = z_val * std_err
        
        return (mean - margin_error, mean + margin_error)
    
    def paired_ttest(self, values1: List[float], values2: List[float]) -> Dict[str, Any]:
        """
        Perform paired t-test between two sets of values.
        
        Args:
            values1: First set of values
            values2: Second set of values
            
        Returns:
            Dictionary with test results
        """
        if len(values1) != len(values2):
            raise ValueError("Both value lists must have the same length")
        
        # Perform paired t-test
        statistic, p_value = ttest_rel_simple(values1, values2)
        
        # Calculate effect size (Cohen's d for paired samples)
        differences = np.array(values1) - np.array(values2)
        effect_size = np.mean(differences) / np.std(differences, ddof=1)
        
        # Determine practical significance
        if abs(effect_size) < 0.2:
            practical_significance = "negligible"
        elif abs(effect_size) < 0.5:
            practical_significance = "small"
        elif abs(effect_size) < 0.8:
            practical_significance = "medium"
        else:
            practical_significance = "large"
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'effect_size': effect_size,
            'practical_significance': practical_significance,
            'mean_difference': np.mean(differences),
            'confidence_interval': self.calculate_confidence_interval(differences.tolist())
        }
    
    def calculate_sample_size(self, effect_size: float, power: float = 0.8) -> int:
        """
        Calculate required sample size for detecting a given effect size.
        
        Args:
            effect_size: Minimum effect size to detect
            power: Statistical power (default 0.8)
            
        Returns:
            Required sample size
        """
        # Effect size for two-proportion test
        z_alpha_2 = norm_ppf(1 - self.alpha/2)
        z_beta = norm_ppf(power)
        
        # Approximate sample size calculation
        n = ((z_alpha_2 + z_beta) / effect_size) ** 2
        
        return int(np.ceil(n))
    
    def analyze_multiple_runs(self, results: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Analyze results from multiple benchmark runs.
        
        Args:
            results: List of dictionaries containing metrics for each run
            
        Returns:
            Statistical analysis of multiple runs
        """
        if not results:
            raise ValueError("No results provided for analysis")
        
        # Extract metrics
        metrics = {}
        for key in results[0].keys():
            metrics[key] = [result[key] for result in results]
        
        # Calculate statistics for each metric
        analysis = {}
        for metric_name, values in metrics.items():
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1)
            ci_lower, ci_upper = self.calculate_confidence_interval(values)
            
            analysis[metric_name] = {
                'mean': mean_val,
                'std': std_val,
                'min': np.min(values),
                'max': np.max(values),
                'confidence_interval': (ci_lower, ci_upper),
                'coefficient_of_variation': std_val / mean_val if mean_val != 0 else 0,
                'values': values
            }
        
        return analysis
    
    def generate_statistical_report(self, analysis: Dict[str, Any], 
                                    comparison_results: Dict[str, Any] = None) -> str:
        """
        Generate a comprehensive statistical report.
        
        Args:
            analysis: Results from analyze_multiple_runs
            comparison_results: Optional results from paired t-test
            
        Returns:
            Formatted statistical report
        """
        report = []
        report.append("=" * 60)
        report.append("STATISTICAL ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Confidence Level: {self.confidence_level:.1%}")
        report.append(f"Significance Level (α): {self.alpha:.3f}")
        report.append("")
        
        # Metrics analysis
        report.append("METRICS ANALYSIS")
        report.append("-" * 40)
        for metric_name, stats_dict in analysis.items():
            report.append(f"\n{metric_name.upper()}:")
            report.append(f"  Mean: {stats_dict['mean']:.4f}")
            report.append(f"  Std Dev: {stats_dict['std']:.4f}")
            report.append(f"  {self.confidence_level:.1%} CI: ({stats_dict['confidence_interval'][0]:.4f}, {stats_dict['confidence_interval'][1]:.4f})")
            report.append(f"  Range: [{stats_dict['min']:.4f}, {stats_dict['max']:.4f}]")
            report.append(f"  CV: {stats_dict['coefficient_of_variation']:.3f}")
        
        # Comparison analysis
        if comparison_results:
            report.append("\n" + "="*40)
            report.append("MODEL COMPARISON")
            report.append("-" * 40)
            report.append(f"Mean Difference: {comparison_results['mean_difference']:.4f}")
            report.append(f"Effect Size (Cohen's d): {comparison_results['effect_size']:.3f}")
            report.append(f"Practical Significance: {comparison_results['practical_significance']}")
            report.append(f"Statistical Significance: {'Yes' if comparison_results['significant'] else 'No'} (p={comparison_results['p_value']:.4f})")
            report.append(f"Difference CI: ({comparison_results['confidence_interval'][0]:.4f}, {comparison_results['confidence_interval'][1]:.4f})")
        
        return "\n".join(report)


def calculate_benchmark_statistical_requirements(
    confidence_level: float = 0.95,
    power: float = 0.8,
    effect_size: float = 0.03,
    baseline_accuracy: float = 0.85
) -> Dict[str, Any]:
    """
    Calculate statistical requirements for benchmark evaluation.
    
    Args:
        confidence_level: Statistical confidence level (default 0.95)
        power: Statistical power (default 0.8)
        effect_size: Minimum detectable effect size (default 0.03 = 3%)
        baseline_accuracy: Expected baseline accuracy (default 0.85)
    
    Returns:
        Dictionary with statistical requirements and recommendations
    """
    alpha = 1 - confidence_level
    z_alpha_2 = norm_ppf(1 - alpha/2)
    z_beta = norm_ppf(power)
    
    # Sample size calculation for accuracy comparison
    p1 = baseline_accuracy
    p2 = baseline_accuracy + effect_size
    p_pooled = (p1 + p2) / 2
    
    # Two-proportion z-test sample size
    n_required = (2 * p_pooled * (1 - p_pooled) * (z_alpha_2 + z_beta)**2) / (p1 - p2)**2
    
    # Conservative estimate
    n_conservative = int(np.ceil(n_required * 1.2))  # 20% buffer
    
    # Multiple runs calculation
    runs_needed = max(3, int(np.ceil(10 / (effect_size * 100))))  # Heuristic
    
    return {
        'statistical_parameters': {
            'confidence_level': confidence_level,
            'power': power,
            'effect_size': effect_size,
            'significance_level': alpha
        },
        'sample_size_requirements': {
            'minimum_required': int(np.ceil(n_required)),
            'recommended': n_conservative,
            'per_class_minimum': int(np.ceil(n_conservative / 9)),  # 9 classes
        },
        'multiple_runs': {
            'recommended_runs': runs_needed,
            'minimum_runs': 3
        },
        'recommendations': {
            'total_samples': f"At least {n_conservative} samples for {effect_size:.1%} effect detection",
            'per_class': f"At least {int(np.ceil(n_conservative / 9))} samples per class",
            'runs': f"Run benchmark {runs_needed} times for robust statistics",
            'significance': f"Use α={alpha:.3f} for statistical significance testing"
        }
    }


def power_analysis(sample_size: int, effect_size: float, alpha: float = 0.05) -> float:
    """
    Calculate statistical power given sample size and effect size.
    
    Args:
        sample_size: Sample size for the test
        effect_size: Effect size to detect
        alpha: Significance level (default 0.05)
        
    Returns:
        Statistical power
    """
    z_alpha_2 = norm_ppf(1 - alpha/2)
    z_score = effect_size * np.sqrt(sample_size) - z_alpha_2
    power = norm_cdf(z_score)
    
    return power 