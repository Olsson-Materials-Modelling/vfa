"""Intervals for cell-level statistics: whole-cell resampling, exact lattice convolution, Clopper-Pearson and Wilson intervals."""
import numpy as np
from scipy import stats

Z_975 = float(stats.norm.ppf(0.975))


def quantile(samples, p):
    """Left-inverse empirical quantile."""
    x = np.sort(np.asarray(samples, float).ravel())
    if x.size == 0: raise ValueError('quantile: no samples')
    return float(x[min(max(int(np.ceil(round(p * x.size, 9))) - 1, 0), x.size - 1)])


def percentile_bootstrap(values, statistic=np.mean, n_replicates=10000, seed=0, alpha=0.05):
    """Estimate and percentile interval of a statistic of cell values under whole-cell resampling."""
    v = np.asarray(values, float)
    if v.ndim != 1 or v.size == 0: raise ValueError(f'percentile_bootstrap: one value per cell is needed, not shape {v.shape}')
    if not np.all(np.isfinite(v)): raise ValueError('percentile_bootstrap: the cell values must be finite')
    if not 0 < alpha < 1: raise ValueError(f'percentile_bootstrap: alpha must lie in (0, 1), not {alpha}')
    rng = np.random.default_rng(seed); reps = np.array([statistic(v[rng.integers(0, v.size, v.size)]) for _ in range(n_replicates)])
    return dict(estimate=float(statistic(v)), lower=quantile(reps, alpha / 2), upper=quantile(reps, 1 - alpha / 2), kind='percentile', n_cells=int(v.size), n_replicates=n_replicates, seed=seed)


def exact_lattice_bootstrap(values, alpha=0.05, max_grid=1 << 24):
    """Exact bootstrap interval of a mean for a lattice-valued cell statistic (counts, indicators), by n-fold convolution; None off a lattice or above max_grid."""
    v = np.asarray(values, float)
    if v.ndim != 1 or v.size == 0: raise ValueError(f'exact_lattice_bootstrap: one value per cell is needed, not shape {v.shape}')
    if not np.all(np.isfinite(v)): raise ValueError('exact_lattice_bootstrap: the cell values must be finite')
    u = np.unique(v); n = v.size
    if u.size == 1: return dict(estimate=float(u[0]), lower=float(u[0]), upper=float(u[0]), kind='exact', n_cells=n)
    for m in range(1, 65):
        h = np.diff(u).min() / m; k = (u - u[0]) / h; npts = int(round(k[-1])) + 1; grid = n * (npts - 1) + 1
        if grid > max_grid: return None
        if np.abs(k - np.round(k)).max() <= 1e-6 + 16 * np.finfo(float).eps * np.abs(u).max() / h: break
    else: return None
    pmf = np.bincount(np.round((v - u[0]) / h).astype(int), minlength=npts) / n; size = 1 << int(np.ceil(np.log2(grid)))
    conv = np.clip(np.fft.irfft(np.fft.rfft(pmf, size) ** n, size)[:grid], 0, None); cdf = np.cumsum(conv / conv.sum())
    lo, hi = (min(int(np.searchsorted(cdf, q, side='left')), grid - 1) for q in (alpha / 2, 1 - alpha / 2))
    return dict(estimate=float(v.mean()), lower=float(u[0] + lo / n * h), upper=float(u[0] + hi / n * h), kind='exact', n_cells=n)


def clopper_pearson(k, n, alpha=0.05):
    """Clopper-Pearson interval of a proportion k/n from beta quantiles; k and n may be effective, non-integer counts."""
    if not 0 <= k <= n or n <= 0: raise ValueError(f'clopper_pearson: need 0 <= k <= n and n > 0, not k={k}, n={n}')
    return (float(stats.beta.ppf(alpha / 2, k, n - k + 1)) if k > 0 else 0.0), (float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k)) if k < n else 1.0)


def bounded_prevalence(values, alpha=0.05, sites=None):
    """Interval of the mean of cell fractions in [0, 1]: Clopper-Pearson over the cells, or at the effective number of the given sites."""
    v = np.asarray(values, float); n = v.size; N = None
    if v.ndim != 1 or n == 0 or not np.isfinite(v).all() or v.min() < 0 or v.max() > 1: raise ValueError(f'bounded_prevalence: one finite fraction in [0, 1] per cell is needed, not shape {v.shape}')
    if sites is not None:
        m = np.asarray(sites, float)
        if m.shape != v.shape or not np.isfinite(m).all() or (m < 1).any() or (m != np.round(m)).any(): raise ValueError(f'bounded_prevalence: sites must be one whole count of at least 1 per cell, not shape {m.shape} for values of shape {v.shape}')
        k = v * m
        if (np.abs(k - np.round(k)) > 1e-6 + 1e-12 * m).any(): raise ValueError('bounded_prevalence: every value must be a whole number of its cell\'s sites')
        v = np.round(k) / m; N = float(m.sum()) if (m == m[0]).all() else float(n * n / np.sum(1 / m))
    x = float(v.mean())
    if N is None or n == 1 or np.unique(v).size == 1: ne, kind = float(n), 'clopper_pearson'
    else: ne, kind = min(n * x * (1 - x) / v.var(ddof=1) * (stats.t.ppf(1 - alpha / 2, N - 1) / stats.t.ppf(1 - alpha / 2, n - 1)) ** 2, N), 'effective_clopper_pearson'
    lo, hi = clopper_pearson(min(x * ne, ne), ne, alpha)
    return dict(estimate=x, lower=lo, upper=hi, kind=kind, n_cells=n, n_sites=None if sites is None else int(np.sum(sites)), n_effective=float(ne))


def wilson(k, n, z=Z_975):
    """Wilson interval of a proportion k/n."""
    if n < 1 or not 0 <= k <= n: raise ValueError(f'wilson: need 0 <= k <= n and n > 0, not k={k}, n={n}')
    p = k / n; c = p + z * z / (2 * n); w = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); d = 1 + z * z / n; return float((c - w) / d), float((c + w) / d)


def influence_mean(x):
    """Estimate and per-cell influence contributions (x - mean)/n of a mean; their squares sum to (n - 1)/n of the squared standard error."""
    x = np.asarray(x, float); est = float(x.mean()); return est, (x - est) / x.size


def maxt_simultaneous(influence, estimates, seed=0, n_replicates=10000, alpha=0.05, groups=None):
    """Simultaneous band over a declared family of estimates from a whole-cell studentized bootstrap of the largest |t|, resampled within groups when given."""
    inf = np.asarray(influence, float); est = np.asarray(estimates, float)
    if inf.ndim != 2 or inf.shape[0] < 2 or est.shape != inf.shape[1:]: raise ValueError(f'maxt_simultaneous: influence must be cells x members with at least two cells and one estimate per member, not {inf.shape} and {est.shape}')
    n = inf.shape[0]; lab = np.zeros(n, int) if groups is None else np.unique(np.asarray(groups), return_inverse=True)[1].ravel()
    if lab.shape != (n,): raise ValueError(f'maxt_simultaneous: groups must give one label per cell, not shape {np.shape(groups)}')
    idx = [np.flatnonzero(lab == g) for g in range(lab.max() + 1)]
    if min(len(k) for k in idx) < 4: raise ValueError('maxt_simultaneous: every group needs at least four cells')
    f = np.array([len(k) / (len(k) - 1) for k in idx])[lab][:, None]; ses = np.sqrt(((inf - np.array([inf[k].mean(0) for k in idx])[lab]) ** 2 * f).sum(0)); inc = ses > 0; rng = np.random.default_rng(seed)
    if not inc.any(): raise ValueError('maxt_simultaneous: every family member has zero cell-level variance')
    I = inf[:, inc]; floor = 1e-8 * ses[inc]; sup = np.empty(n_replicates); skipped = 0; chunk = max(1, min(n_replicates, int(4e7 // I.size)))
    for s in range(0, n_replicates, chunk):
        m = min(chunk, n_replicates - s); rows = np.concatenate([k[rng.integers(0, len(k), (m, len(k)))] for k in idx], axis=1); R = I[rows]; d = R.sum(1)
        se = np.sqrt(sum(((R[:, c] - R[:, c].mean(1, keepdims=True)) ** 2).sum(1) * len(k) / (len(k) - 1) for c, k in zip(np.split(np.arange(n), np.cumsum([len(k) for k in idx])[:-1]), idx)))
        ok = se > floor; skipped += int((~ok).sum())
        with np.errstate(divide='ignore', invalid='ignore'): t = np.where(ok, np.abs(d) / se, 0.0)
        sup[s:s + m] = t.max(1)
    crit = quantile(sup, 1 - alpha); lo = np.where(inc, est - crit * ses, np.nan); hi = np.where(inc, est + crit * ses, np.nan)
    return dict(critical_value=crit, lower=lo, upper=hi, standard_errors=ses, excluded=np.flatnonzero(~inc), n_cells=int(n), n_replicates=n_replicates, n_without_spread=skipped, seed=seed)
