"""Trends along an ordering variable: smoothing in order with a held-out window, prefix convergence, and the change between two positions by a straight line, Theil-Sen and least squares side by side."""
import numpy as np
from scipy import stats
from .intervals import quantile


def smooth_in_order(Y, at, width, leave_out=False):
    """Local-linear smoothing of the columns of Y (rows in order) with Gaussian weights of width cells, at positions at; exact for a linear trend."""
    Y = np.asarray(Y, float).reshape(len(Y), -1); r = np.arange(len(Y), dtype=float); out = np.empty((len(at), Y.shape[1]))
    for i, g in enumerate(at):
        w = np.exp(-0.5 * ((r - g) / width) ** 2)
        if leave_out: w[int(round(g))] = 0.0
        X = np.c_[np.ones(len(r)), r - g]; out[i] = np.linalg.lstsq(X.T @ (w[:, None] * X), X.T @ (w[:, None] * Y), rcond=None)[0][0]
    return out


def held_out_window(Y, widths):
    """The width whose leave-one-out smoothing best predicts each cell: log loss for class indicators, squared error otherwise."""
    Y = np.asarray(Y, float).reshape(len(Y), -1); onehot = np.allclose(Y.sum(1), 1) and np.isin(Y, (0, 1)).all(); losses = []
    for w in widths:
        P = smooth_in_order(Y, np.arange(len(Y)), w, leave_out=True)
        losses.append(float(-np.mean(np.log(np.clip((P * Y).sum(1), 1e-12, None)))) if onehot else float(np.mean((P - Y) ** 2)))
    return dict(widths=list(widths), losses=losses, chosen=widths[int(np.argmin(losses))])


def positions(x, quantiles=(0.025, 0.975)):
    """Quantiles of the ordering variable and their fractional positions in the sorted cells."""
    x = np.sort(np.asarray(x, float)); q = np.quantile(x, quantiles); return q, np.interp(q, x, np.arange(len(x)))


def prefix_halfwidth(x):
    """Running mean and uncorrected 95 % t half-width over every prefix of cells."""
    x = np.asarray(x, float); n = np.arange(1, len(x) + 1); mean = np.cumsum(x) / n
    with np.errstate(invalid='ignore', divide='ignore'): sd = np.sqrt(np.clip((np.cumsum(x ** 2) - n * mean ** 2) / (n - 1), 0, None)); h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return mean, h


def linear_change(x, y, lo, hi, n_replicates=2000, seed=0, alpha=0.05):
    """Change of y between x = lo and x = hi by a straight line, Theil-Sen and least squares side by side, with percentile intervals from the same whole-cell resamples."""
    x, y = np.asarray(x, float), np.asarray(y, float); rng = np.random.default_rng(seed); J = [rng.integers(0, x.size, x.size) for _ in range(n_replicates)]
    ts = stats.theilslopes(y, x); b, b0 = np.polyfit(x, y, 1); d = hi - lo
    bts = np.array([stats.theilslopes(y[j], x[j])[0] for j in J]) * d; bls = np.array([np.polyfit(x[j], y[j], 1)[0] for j in J]) * d; rs = stats.spearmanr(x, y)
    return dict(theil_sen=dict(low=ts[1] + ts[0] * lo, high=ts[1] + ts[0] * hi, change=ts[0] * d, lower=quantile(bts, alpha / 2), upper=quantile(bts, 1 - alpha / 2)),
                least_squares=dict(low=b0 + b * lo, high=b0 + b * hi, change=b * d, lower=quantile(bls, alpha / 2), upper=quantile(bls, 1 - alpha / 2)),
                spearman=float(rs.statistic), p=float(rs.pvalue), n=int(x.size), n_replicates=n_replicates, seed=seed)
