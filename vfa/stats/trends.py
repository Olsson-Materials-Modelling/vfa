"""Trends along an ordering variable: smoothing in order, held-out window, prefix convergence, straight-line change."""
import numpy as np
from scipy import stats
from .intervals import quantile
from .models import log_loss


def smooth_in_order(Y, at, width, leave_out=False):
    """Local-linear smoothing of the columns of Y (rows in order) with Gaussian weights of width cells, at positions at."""
    Y = np.asarray(Y, float).reshape(len(Y), -1); r = np.arange(len(Y), dtype=float); out = np.empty((len(at), Y.shape[1]))
    for i, g in enumerate(at):
        w = np.exp(-0.5 * ((r - g) / width) ** 2)
        if leave_out: w[int(round(g))] = 0.0
        X = np.c_[np.ones(len(r)), r - g]; out[i] = np.linalg.lstsq(X.T @ (w[:, None] * X), X.T @ (w[:, None] * Y), rcond=None)[0][0]
    return out


def held_out_window(Y, widths):
    """The width (cells) whose leave-one-out smoothing best predicts the cells, with the loss per width and per column choices."""
    Y = np.asarray(Y, float).reshape(len(Y), -1); ws = sorted(set(widths))
    if len(ws) < 2 or ws[0] <= 0: raise ValueError(f'held_out_window: need at least two distinct positive widths, not {list(widths)}')
    onehot = np.allclose(Y.sum(1), 1) and np.isin(Y, (0, 1)).all(); var = Y.var(0, ddof=1); scale = np.where(var > 0, var, 1.0); col = []
    for w in ws:
        P = smooth_in_order(Y, np.arange(len(Y)), w, leave_out=True); col.append(np.full(Y.shape[1], log_loss(P, Y.argmax(1), range(Y.shape[1]))) if onehot else ((P - Y) ** 2).mean(0) / scale)
    col = np.array(col); losses = col.mean(1); i = int(np.argmin(losses))
    return dict(widths=ws, losses=[float(x) for x in losses], chosen=ws[i], edge='lower' if i == 0 else 'upper' if i == len(ws) - 1 else None, column_chosen=None if onehot else [ws[j] for j in np.argmin(col, 0)])


def positions(x, quantiles=(0.025, 0.975)):
    """Quantiles of the ordering variable and their fractional positions in the sorted cells."""
    x = np.sort(np.asarray(x, float))
    if x.size == 0 or not np.all(np.isfinite(x)): raise ValueError('positions: one finite ordering value per cell is needed')
    u, first, count = np.unique(x, return_index=True, return_counts=True); q = np.quantile(x, quantiles); return q, np.interp(q, u, first + (count - 1) / 2)


def prefix_halfwidth(x):
    """Running mean and uncorrected 95 % t half-width over every prefix of cells, by Welford's recurrence."""
    x = np.asarray(x, float); n = np.arange(1, len(x) + 1); mean = np.empty(len(x)); m2 = np.empty(len(x)); m = s = 0.0
    for i, v in enumerate(x): d = v - m; m += d / (i + 1); s += d * (v - m); mean[i], m2[i] = m, s
    with np.errstate(invalid='ignore', divide='ignore'): h = stats.t.ppf(0.975, n - 1) * np.sqrt(m2 / (n - 1)) / np.sqrt(n)
    return mean, h


def linear_change(x, y, lo, hi, n_replicates=2000, seed=0, alpha=0.05):
    """Change of y between x = lo and x = hi by Theil-Sen and least squares, with whole-cell percentile intervals."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 1 or x.shape != y.shape or x.size < 3 or not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))): raise ValueError(f'linear_change: x and y must be finite, of one shape and hold at least 3 cells, not {x.shape} and {y.shape}')
    rng = np.random.default_rng(seed); J = [rng.integers(0, x.size, x.size) for _ in range(n_replicates)]
    ts = stats.theilslopes(y, x); b, b0 = np.polyfit(x, y, 1); d = hi - lo
    J = [j for j in J if np.unique(x[j]).size > 1]
    if not J: raise ValueError('linear_change: no resample holds two distinct x values')
    bts = np.array([stats.theilslopes(y[j], x[j])[0] for j in J]) * d; bls = np.array([np.polyfit(x[j], y[j], 1)[0] for j in J]) * d; rs = stats.spearmanr(x, y)
    return dict(theil_sen=dict(low=ts[1] + ts[0] * lo, high=ts[1] + ts[0] * hi, change=ts[0] * d, lower=quantile(bts, alpha / 2), upper=quantile(bts, 1 - alpha / 2)),
                least_squares=dict(low=b0 + b * lo, high=b0 + b * hi, change=b * d, lower=quantile(bls, alpha / 2), upper=quantile(bls, 1 - alpha / 2)),
                spearman=float(rs.statistic), p=float(rs.pvalue), n=int(x.size), n_replicates=n_replicates, seed=seed)
