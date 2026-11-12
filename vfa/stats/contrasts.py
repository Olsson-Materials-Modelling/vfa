"""Contrasts between two populations of cells: location, scale and a distribution-free effect size, each with a whole-cell bootstrap interval."""
import numpy as np
from scipy import stats
from .intervals import quantile


def _cells(x, name, least=1):
    v = np.asarray(x, float)
    if v.ndim != 1 or v.size < least or not np.all(np.isfinite(v)): raise ValueError(f'{name}: needs one finite value per cell, at least {least} cells, not shape {v.shape}')
    return v


def _spread(x):
    return 64 * np.finfo(float).eps * max(float(np.abs(x).max()), np.finfo(float).tiny)


def cliff_delta(a, b):
    """Cliff's delta P(a > b) - P(a < b) in [-1, 1], from the Mann-Whitney statistic."""
    a, b = _cells(a, 'cliff_delta'), _cells(b, 'cliff_delta')
    return float(2 * stats.mannwhitneyu(a, b, alternative='two-sided').statistic / (len(a) * len(b)) - 1)


def cliff_delta_interval(a, b, seed=0, n_replicates=1000, alpha=0.05):
    """Cliff's delta with a percentile interval under resampling of both populations."""
    a, b = _cells(a, 'cliff_delta_interval'), _cells(b, 'cliff_delta_interval'); rng = np.random.default_rng(seed)
    reps = np.array([cliff_delta(a[rng.integers(0, a.size, a.size)], b[rng.integers(0, b.size, b.size)]) for _ in range(n_replicates)])
    return dict(delta=cliff_delta(a, b), lower=quantile(reps, alpha / 2), upper=quantile(reps, 1 - alpha / 2), p=float(stats.mannwhitneyu(a, b, alternative='two-sided').pvalue), n_a=int(a.size), n_b=int(b.size))


def location_shift(a, reference):
    """Difference of means a - reference in units of the reference population's cell-to-cell standard deviation."""
    a, r = _cells(a, 'location_shift'), _cells(reference, 'location_shift', 2)
    if r.std() <= _spread(r): raise ValueError('location_shift: the reference has no spread')
    return float((a.mean() - r.mean()) / r.std(ddof=1))


def scale_ratio(a, reference, seed=0, n_replicates=4000, alpha=0.05):
    """Ratio of standard deviations with a percentile interval over the resamples in which both populations have a spread, and the Brown-Forsythe p-value."""
    a, r = _cells(a, 'scale_ratio', 3), _cells(reference, 'scale_ratio', 3); rng = np.random.default_rng(seed)
    ta, tr = _spread(a), _spread(r)
    if a.std() <= ta or r.std() <= tr: raise ValueError(f"scale_ratio: the {'reference' if r.std() <= tr else 'population'} has no spread")
    sd = np.array([(a[rng.integers(0, a.size, a.size)].std(ddof=1), r[rng.integers(0, r.size, r.size)].std(ddof=1)) for _ in range(n_replicates)]); ok = (sd[:, 0] > ta) & (sd[:, 1] > tr); reps = sd[ok, 0] / sd[ok, 1]
    if not ok.any(): raise ValueError('scale_ratio: no resample has a spread in both populations')
    return dict(ratio=float(a.std(ddof=1) / r.std(ddof=1)), lower=quantile(reps, alpha / 2), upper=quantile(reps, 1 - alpha / 2), brown_forsythe_p=float(stats.levene(a, r, center='median').pvalue), n_replicates=int(ok.sum()))
