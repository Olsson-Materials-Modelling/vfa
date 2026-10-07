"""Contrasts between two populations of cells: location, scale and a distribution-free effect size, each with a whole-cell bootstrap interval."""
import numpy as np
from scipy import stats
from .intervals import quantile


def cliff_delta(a, b):
    """Cliff's delta P(a > b) - P(a < b) in [-1, 1], from the Mann-Whitney statistic."""
    return float(2 * stats.mannwhitneyu(a, b, alternative='two-sided').statistic / (len(a) * len(b)) - 1)


def cliff_delta_interval(a, b, seed=0, n_replicates=1000, alpha=0.05):
    """Cliff's delta with a percentile interval under resampling of both populations."""
    a, b = np.asarray(a, float), np.asarray(b, float); rng = np.random.default_rng(seed)
    reps = np.array([cliff_delta(a[rng.integers(0, a.size, a.size)], b[rng.integers(0, b.size, b.size)]) for _ in range(n_replicates)])
    return dict(delta=cliff_delta(a, b), lower=quantile(reps, alpha / 2), upper=quantile(reps, 1 - alpha / 2), p=float(stats.mannwhitneyu(a, b, alternative='two-sided').pvalue), n_a=int(a.size), n_b=int(b.size))


def location_shift(a, reference):
    """Difference of means a - reference in units of the reference population's cell-to-cell standard deviation."""
    a, r = np.asarray(a, float), np.asarray(reference, float); return float((a.mean() - r.mean()) / r.std(ddof=1))


def scale_ratio(a, reference, seed=0, n_replicates=4000, alpha=0.05):
    """Ratio of standard deviations with a percentile interval and the Brown-Forsythe p-value."""
    a, r = np.asarray(a, float), np.asarray(reference, float); rng = np.random.default_rng(seed)
    reps = np.array([a[rng.integers(0, a.size, a.size)].std(ddof=1) / r[rng.integers(0, r.size, r.size)].std(ddof=1) for _ in range(n_replicates)])
    return dict(ratio=float(a.std(ddof=1) / r.std(ddof=1)), lower=quantile(reps, alpha / 2), upper=quantile(reps, 1 - alpha / 2), brown_forsythe_p=float(stats.levene(a, r, center='median').pvalue))
