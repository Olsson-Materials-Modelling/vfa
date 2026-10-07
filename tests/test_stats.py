"""Closed forms and invariances of the statistics."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vfa.stats.intervals import wilson, percentile_bootstrap, exact_lattice_bootstrap, bounded_prevalence, maxt_simultaneous, influence_mean, quantile
from vfa.stats.contrasts import cliff_delta, cliff_delta_interval, scale_ratio, location_shift
from vfa.stats.trends import smooth_in_order, held_out_window, linear_change, prefix_halfwidth, positions
from vfa.stats.models import softmax_fit, softmax_predict, held_out_shares, standardise


def test_wilson_and_quantile():
    lo, hi = wilson(0, 10); z = 1.959963984540054; assert lo == 0 and abs(hi - z * z / (10 + z * z)) < 1e-12
    assert quantile([1, 2, 3, 4], 0.5) == 2 and quantile([1, 2, 3, 4], 0.51) == 3


def test_bootstrap_of_a_constant_and_exact_lattice():
    r = percentile_bootstrap(np.full(20, 0.3), n_replicates=200); assert np.isclose(r['lower'], 0.3) and np.isclose(r['upper'], 0.3)
    v = np.array([0] * 7 + [1] * 3); e = exact_lattice_bootstrap(v); assert e['kind'] == 'exact' and e['lower'] <= 0.3 <= e['upper'] and abs(e['lower'] - 0.0) < 1e-9 and abs(e['upper'] - 0.6) < 1e-9   # binomial(10, 0.3) 2.5 % and 97.5 % points
    assert exact_lattice_bootstrap(np.array([0.1, 0.25, 0.9])) is None and bounded_prevalence(v)['kind'] == 'exact'


def test_maxt_covers_the_zero_variance_case():
    rng = np.random.default_rng(0); x = rng.normal(size=(40, 3)); x[:, 2] = 1.0; est = x.mean(0); inf = np.stack([influence_mean(x[:, k])[1] for k in range(3)], 1)
    r = maxt_simultaneous(inf, est, n_replicates=2000); assert list(r['excluded']) == [2] and r['critical_value'] > 1.96 and r['lower'][0] < est[0] < r['upper'][0]


def test_cliff_and_scale():
    a, b = np.arange(10.0), np.arange(10.0) + 20; assert cliff_delta(a, b) == -1 and cliff_delta(a, a) == 0 and cliff_delta_interval(a, b, n_replicates=50)['upper'] == -1
    rng = np.random.default_rng(1); r = scale_ratio(rng.normal(0, 2, 400), rng.normal(0, 1, 400), n_replicates=300); assert 1.7 < r['ratio'] < 2.3 and r['brown_forsythe_p'] < 1e-6 and abs(location_shift([1, 2, 3], [0, 1, 2])) == 1


def test_smoothing_is_exact_on_a_line_and_the_window_is_chosen():
    y = 2.0 * np.arange(50) + 1; s = smooth_in_order(y, [0, 10.5, 49], 5.0); assert abs(s[:, 0] - (2 * np.array([0, 10.5, 49]) + 1)).max() < 1e-9
    rng = np.random.default_rng(2); Y = np.stack([np.where(rng.random(200) < np.linspace(0.1, 0.9, 200), 1.0, 0.0)], 1); Y = np.hstack([Y, 1 - Y]); w = held_out_window(Y, (2, 5, 20, 60)); assert w['chosen'] != 2 and w['losses'][0] > min(w['losses'])      # a linear trend: every width beyond the narrowest fits
    q, at = positions(np.linspace(0, 1, 101)); assert abs(q[0] - 0.025) < 1e-9 and abs(at[0] - 2.5) < 1e-9
    m, h = prefix_halfwidth(np.ones(5)); assert abs(m - 1).max() < 1e-12 and np.isnan(h[0]) and h[-1] == 0


def test_linear_change_on_an_exact_line():
    x = np.linspace(2.9, 3.2, 40); r = linear_change(x, 3 * x - 1, 2.92, 3.15, n_replicates=50)
    for k in ('theil_sen', 'least_squares'): assert abs(r[k]['change'] - 3 * 0.23) < 1e-9 and abs(r[k]['lower'] - r[k]['change']) < 1e-9 and abs(r[k]['upper'] - r[k]['change']) < 1e-9


def test_softmax_separates_classes():
    rng = np.random.default_rng(3); X = np.vstack([rng.normal(-2, 0.5, (40, 2)), rng.normal(2, 0.5, (40, 2))]); y = np.array(['a'] * 40 + ['b'] * 40)
    W = softmax_fit(standardise(X, X), y, ['a', 'b']); P = softmax_predict(W, standardise(X, X)); assert (np.array(['a', 'b'])[P.argmax(1)] == y).all()
    r = held_out_shares(X, y, X[:40], ['a', 'b'], n_replicates=20); assert r['predicted'][0] > 0.95
