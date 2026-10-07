"""Intervals for cell-level statistics under whole-cell resampling: left-inverse quantiles, PCG64 with the given seed."""
import numpy as np


def quantile(samples, p):
    """Left-inverse empirical quantile."""
    x = np.sort(np.asarray(samples, float)); return float(x[min(max(int(np.ceil(p * x.size)) - 1, 0), x.size - 1)])


def percentile_bootstrap(values, statistic=np.mean, n_replicates=10000, seed=0, alpha=0.05):
    """Estimate and percentile interval of a statistic of cell values under whole-cell resampling."""
    v = np.asarray(values, float); rng = np.random.default_rng(seed); reps = np.array([statistic(v[rng.integers(0, v.size, v.size)]) for _ in range(n_replicates)])
    return dict(estimate=float(statistic(v)), lower=quantile(reps, alpha / 2), upper=quantile(reps, 1 - alpha / 2), kind='percentile', n_cells=int(v.size), n_replicates=n_replicates, seed=seed)


def exact_lattice_bootstrap(values, alpha=0.05, max_grid=1 << 24):
    """Exact bootstrap interval of a mean for a lattice-valued cell statistic (counts, indicators), by n-fold convolution; None otherwise."""
    v = np.asarray(values, float); u = np.unique(v); n = v.size
    if u.size == 1: return dict(estimate=float(u[0]), lower=float(u[0]), upper=float(u[0]), kind='exact', n_cells=n)
    d = np.diff(u); h = d.min(); k = (u - u[0]) / h
    if not np.allclose(k, np.round(k), atol=1e-9): return None
    npts = int(round(k[-1])) + 1; grid = n * (npts - 1) + 1
    if grid > max_grid: return None
    pmf = np.bincount(np.round((v - u[0]) / h).astype(int), minlength=npts) / n; size = 1 << int(np.ceil(np.log2(grid)))
    conv = np.clip(np.fft.irfft(np.fft.rfft(pmf, size) ** n, size)[:grid], 0, None); cdf = np.cumsum(conv / conv.sum())
    lo, hi = (min(int(np.searchsorted(cdf, q, side='left')), grid - 1) for q in (alpha / 2, 1 - alpha / 2))
    return dict(estimate=float(v.mean()), lower=float(u[0] + lo / n * h), upper=float(u[0] + hi / n * h), kind='exact', n_cells=n)


def bounded_prevalence(values, seed=0, alpha=0.05, n_replicates=10000):
    """Interval of a prevalence in [0, 1]: exact when the cell statistic is lattice valued, percentile otherwise; limits clipped to [0, 1]."""
    r = exact_lattice_bootstrap(values, alpha) or percentile_bootstrap(values, np.mean, n_replicates, seed, alpha)
    r['lower'], r['upper'] = float(np.clip(r['lower'], 0, 1)), float(np.clip(r['upper'], 0, 1)); return r


def wilson(k, n, z=1.959963984540054):
    """Wilson interval of a proportion k/n; the normal approximation fails at small counts."""
    if n == 0: return (np.nan, np.nan)
    p = k / n; c = p + z * z / (2 * n); w = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); d = 1 + z * z / n; return float((c - w) / d), float((c + w) / d)


def influence_mean(x):
    """Estimate and per-cell influence contributions of a mean; the squared contributions sum to the squared standard error."""
    x = np.asarray(x, float); est = float(x.mean()); return est, (x - est) / x.size


def influence_ratio(numerator, denominator):
    """Estimate and per-cell influence contributions of a pooled ratio sum n / sum d, the site-prevalence estimator with the cell as the unit."""
    a, b = np.asarray(numerator, float), np.asarray(denominator, float); est = float(a.sum() / b.sum()); return est, (a - est * b) / b.sum()


def maxt_simultaneous(influence, estimates, seed=0, n_replicates=10000, alpha=0.05):
    """Simultaneous band over a declared family: one Gaussian multiplier per cell shared across the family, studentised by the original standard errors; zero-variance members excluded and listed."""
    inf = np.asarray(influence, float); est = np.asarray(estimates, float); ses = np.sqrt((inf ** 2).sum(0)); inc = ses > 0; rng = np.random.default_rng(seed)
    if not inc.any(): raise ValueError('maxt_simultaneous: every family member has zero cell-level variance')
    sup = np.empty(n_replicates); chunk = max(1, min(n_replicates, int(4e7 // inf.shape[0])))
    for s in range(0, n_replicates, chunk):
        g = rng.standard_normal((min(chunk, n_replicates - s), inf.shape[0])); sup[s:s + len(g)] = np.abs((g @ inf[:, inc]) / ses[inc]).max(1)
    crit = quantile(sup, 1 - alpha); lo = np.where(inc, est - crit * ses, np.nan); hi = np.where(inc, est + crit * ses, np.nan)
    return dict(critical_value=crit, lower=lo, upper=hi, standard_errors=ses, excluded=np.flatnonzero(~inc), n_cells=int(inf.shape[0]), n_replicates=n_replicates, seed=seed)


def planning_n(cell_variance, estimate, target_absolute=0.002, target_relative=0.2, z=1.959963984540054):
    """Cells for a two-sided 95 % half-width at the observed cell variance, absolute and relative."""
    n_abs = z * z * cell_variance / target_absolute ** 2; n_rel = z * z * cell_variance / (target_relative * abs(estimate)) ** 2 if estimate else np.nan
    return dict(n_absolute=float(np.ceil(n_abs)), n_relative=float(np.ceil(n_rel)) if np.isfinite(n_rel) else np.nan)
