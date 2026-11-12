"""Closed forms, invariances and exact coverage of the statistics."""
import itertools, sys
from pathlib import Path
import numpy as np
import pytest
from scipy import stats
from scipy.special import gammaln
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vfa.stats import intervals
from vfa.stats.intervals import wilson, percentile_bootstrap, exact_lattice_bootstrap, bounded_prevalence, maxt_simultaneous, influence_mean, quantile, clopper_pearson
from vfa.stats.contrasts import cliff_delta, cliff_delta_interval, scale_ratio, location_shift
from vfa.stats.trends import smooth_in_order, held_out_window, linear_change, prefix_halfwidth, positions
from vfa.stats.models import softmax_fit, softmax_predict, held_out_shares, learning_curve, standardise, log_loss


def _infimum_coverage(n, interval, lo=0.01, hi=0.5):
    L, U = np.array([interval(k, n) for k in range(n + 1)]).T; k = np.arange(n + 1); b = np.unique(np.r_[L, U, lo, hi]); b = b[(b >= lo) & (b <= hi)]
    return min(min(pm[(L < x) & (U >= x)].sum() if x > lo else 1.0, pm[(L <= x) & (U > x)].sum() if x < hi else 1.0, pm[(L <= x) & (U >= x)].sum()) for x in b for pm in [stats.binom.pmf(k, n, x)])


def _prevalence(k, n):
    r = bounded_prevalence(np.r_[np.ones(k), np.zeros(n - k)]); return r['lower'], r['upper']


def test_wilson_and_quantile():
    z = intervals.Z_975; lo, hi = wilson(0, 10); assert z == stats.norm.ppf(0.975) and lo == 0 and abs(hi - z * z / (10 + z * z)) < 1e-12
    assert quantile([1, 2, 3, 4], 0.5) == 2 and quantile([1, 2, 3, 4], 0.51) == 3


def test_bootstrap_of_a_constant_and_exact_lattice():
    r = percentile_bootstrap(np.full(20, 0.3), n_replicates=200); assert np.isclose(r['lower'], 0.3) and np.isclose(r['upper'], 0.3)
    v = np.array([0] * 7 + [1] * 3); e = exact_lattice_bootstrap(v); assert e['kind'] == 'exact' and e['lower'] <= 0.3 <= e['upper'] and abs(e['lower'] - 0.0) < 1e-9 and abs(e['upper'] - 0.6) < 1e-9
    assert exact_lattice_bootstrap(np.array([0.0, 1.0, np.sqrt(2)])) is None
    for v in ([0, 2, 5, 5, 2, 0, 2], [0, 3, 5], [0.1, 0.25, 0.9]):
        w = np.asarray(v, float); e = exact_lattice_bootstrap(w); m = np.array(np.meshgrid(*[w] * len(w))).reshape(len(w), -1).mean(0)
        assert e['kind'] == 'exact' and abs(e['lower'] - quantile(m, 0.025)) < 1e-12 and abs(e['upper'] - quantile(m, 0.975)) < 1e-12


def test_lattice_tolerance_scales_with_the_step():
    assert exact_lattice_bootstrap(np.array([0.0] * 10 + [1.0] * 10 + [1000.004] * 2)) is None
    k = np.repeat(np.arange(11), 3); clean = exact_lattice_bootstrap(k / 10); near = exact_lattice_bootstrap(k * 0.1 + 1e-10 * np.sin(k)); big = exact_lattice_bootstrap(1e8 + k * 1e-3)
    assert near['kind'] == 'exact' and abs(near['lower'] - clean['lower']) < 1e-8 and abs(near['upper'] - clean['upper']) < 1e-8 and big is not None and abs(big['upper'] - big['lower'] - (clean['upper'] - clean['lower']) / 100) < 1e-6
    assert exact_lattice_bootstrap(k / 10 + 1e-4 * np.sin(k + 1)) is None


def test_prevalence_of_indicators_is_clopper_pearson_with_exact_coverage():
    for n in (10, 30, 100):
        assert _infimum_coverage(n, _prevalence) >= 0.95 - 1e-9 and _infimum_coverage(n, wilson) < 0.92
        r0, r1 = bounded_prevalence(np.zeros(n)), bounded_prevalence(np.ones(n)); assert r0['kind'] == r1['kind'] == 'clopper_pearson' and r0['lower'] == 0 and abs(r0['upper'] - (1 - 0.025 ** (1 / n))) < 1e-12 and abs(r1['lower'] - 0.025 ** (1 / n)) < 1e-12 and r1['upper'] == 1
    for k, n in ((0, 10), (3, 30), (29, 30), (17, 40)):
        ci = stats.binomtest(k, n).proportion_ci(0.95, 'exact'); lo, hi = _prevalence(k, n); assert abs(lo - ci.low) < 1e-12 and abs(hi - ci.high) < 1e-12


def test_prevalence_of_site_fractions_has_exact_coverage():
    n, m = 10, 4; C = np.array([c for c in itertools.product(range(n + 1), repeat=m + 1) if sum(c) == n]); R = [bounded_prevalence(np.repeat(np.arange(m + 1) / m, c), sites=np.full(n, m)) for c in C]
    L, U = np.array([r['lower'] for r in R]), np.array([r['upper'] for r in R]); lmc = gammaln(n + 1) - gammaln(C + 1).sum(1)
    assert len(C) == 1001 and (U - L > 0.15).all() and {r['kind'] for r in R} == {'clopper_pearson', 'effective_clopper_pearson'} and all(0 < r['n_effective'] <= n * m and r['n_sites'] == n * m for r in R)
    assert bounded_prevalence(np.r_[[0.25] * 5, [0.5] * 5], sites=np.full(n, m))['n_effective'] == n * m and {r['kind'] for r in (bounded_prevalence(np.repeat(np.arange(m + 1) / m, c)) for c in C)} == {'clopper_pearson'}
    for rho in (0.0, 0.3):
        for p in (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5):
            q = stats.binom.pmf(np.arange(m + 1), m, p) if rho == 0 else stats.betabinom.pmf(np.arange(m + 1), m, p * (1 - rho) / rho, (1 - p) * (1 - rho) / rho); pr = np.exp(lmc + C @ np.log(q))
            assert abs(pr.sum() - 1) < 1e-9 and pr[(L <= p) & (p <= U)].sum() >= 0.95


def test_prevalence_takes_the_sites_as_given_and_falls_back_to_the_cells():
    k = np.array([5, 5, 6, 5, 5, 6, 5, 5, 5, 6]); v = k / 61; w = v.copy(); w[1] = 5 * (1 / 61); s = np.r_[[68] * 5, [70] * 5]; assert w[1] != v[1]
    r, q = bounded_prevalence(v, sites=np.full(10, 61)), bounded_prevalence(w, sites=np.full(10, 61)); assert r == q and r['n_effective'] == r['n_sites'] == 610
    e = bounded_prevalence(np.array([3, 4, 5, 3, 4, 6, 3, 4, 5, 2]) / 49, sites=np.full(10, 49)); assert e['n_effective'] == e['n_sites'] == 490
    u = bounded_prevalence(k / s, sites=s); assert u['kind'] == 'effective_clopper_pearson' and u['n_sites'] == 690 and abs(u['n_effective'] - 100 / np.sum(1 / s)) < 1e-9
    m = np.r_[[5, 500] * 5]; c = bounded_prevalence(np.array([0, 10, 0, 9, 1, 11, 0, 10, 0, 12]) / m, sites=m); assert c['n_effective'] <= 100 / np.sum(1 / m) + 1e-9 < 100
    z = np.zeros(10); z[3] = 0.1 + 0.2 - 0.3; assert bounded_prevalence(z, sites=np.full(10, 68)) == bounded_prevalence(np.zeros(10), sites=np.full(10, 68))
    with pytest.raises(ValueError, match="whole number of its cell's sites"): bounded_prevalence([0.5 / 68, 1.3 / 68, 0], sites=[68] * 3)
    for x, kw in ((k / s, {}), (np.full(10, 3 / 68), dict(sites=np.full(10, 68))), (np.zeros(10), dict(sites=np.full(10, 68))), (v[:1], dict(sites=[61]))):
        r = bounded_prevalence(x, **kw); assert r['kind'] == 'clopper_pearson' and r['n_effective'] == x.size and r['n_sites'] == (None if not kw else 68 * x.size if x.size == 10 else 61)
    for bad in (np.full(9, 68), np.full(10, 68.5), np.r_[[68] * 9, 0], np.r_[[68] * 9, np.nan]):
        with pytest.raises(ValueError, match='bounded_prevalence: sites'): bounded_prevalence(k / 68, sites=bad)


def test_prevalence_rejects_values_outside_the_unit_interval():
    for v in ([0.2, 1.3], [-0.1, 0.5], [], [0.5, np.nan]):
        with pytest.raises(ValueError, match='bounded_prevalence'): bounded_prevalence(v)


def test_maxt_covers_the_zero_variance_case():
    rng = np.random.default_rng(0); x = rng.normal(size=(40, 3)); x[:, 2] = 1.0; est = x.mean(0); inf = np.stack([influence_mean(x[:, k])[1] for k in range(3)], 1)
    r = maxt_simultaneous(inf, est, n_replicates=2000); assert list(r['excluded']) == [2] and r['critical_value'] > intervals.Z_975 and r['lower'][0] < est[0] < r['upper'][0]


def test_cliff_and_scale():
    a, b = np.arange(10.0), np.arange(10.0) + 20; assert cliff_delta(a, b) == -1 and cliff_delta(a, a) == 0 and cliff_delta_interval(a, b, n_replicates=50)['upper'] == -1
    rng = np.random.default_rng(1); r = scale_ratio(rng.normal(0, 2, 400), rng.normal(0, 1, 400), n_replicates=300); assert 1.7 < r['ratio'] < 2.3 and r['brown_forsythe_p'] < 1e-6 and abs(location_shift([1, 2, 3], [0, 1, 2])) == 1


def test_smoothing_is_exact_on_a_line_and_the_window_is_chosen():
    y = 2.0 * np.arange(50) + 1; s = smooth_in_order(y, [0, 10.5, 49], 5.0); assert abs(s[:, 0] - (2 * np.array([0, 10.5, 49]) + 1)).max() < 1e-9
    rng = np.random.default_rng(2); Y = np.stack([np.where(rng.random(200) < np.linspace(0.1, 0.9, 200), 1.0, 0.0)], 1); Y = np.hstack([Y, 1 - Y]); w = held_out_window(Y, (2, 5, 20, 60)); assert w['chosen'] != 2 and w['losses'][0] > min(w['losses'])
    q, at = positions(np.linspace(0, 1, 101)); assert abs(q[0] - 0.025) < 1e-9 and abs(at[0] - 2.5) < 1e-9
    m, h = prefix_halfwidth(np.ones(5)); assert abs(m - 1).max() < 1e-12 and np.isnan(h[0]) and h[-1] == 0


def test_held_out_window_is_unit_free_and_reports_the_edge():
    rng = np.random.default_rng(4); t = np.arange(60.0); a = 0.05 * t + rng.normal(0, 1, 60); b = np.sin(t / 3) + rng.normal(0, 0.2, 60); W = (1, 2, 4, 8, 16, 32, 120)
    w = held_out_window(np.c_[a, b], W); assert w['column_chosen'][0] > w['column_chosen'][1] and w['edge'] is None
    for s in (1e-3, 1e3): assert held_out_window(np.c_[a * s, b], W)['chosen'] == w['chosen'] == held_out_window(np.c_[a, b * s], W)['chosen']
    assert held_out_window(a, W)['edge'] == 'upper' and held_out_window(b, (8, 16, 32))['edge'] == 'lower' and held_out_window(a, (120, 1, 8, 4))['widths'] == [1, 4, 8, 120]
    for bad in ((5,), (0, 2), (5, 5)):
        with pytest.raises(ValueError, match='held_out_window'): held_out_window(a, bad)


def test_held_out_window_scores_class_indicators_by_log_loss():
    rng = np.random.default_rng(5); y = (rng.random(40) < 0.4).astype(int); Y = np.c_[1 - y, y]; w = held_out_window(Y, (3, 10))
    assert abs(w['losses'][0] - log_loss(smooth_in_order(Y, np.arange(40), 3, leave_out=True), y, (0, 1))) < 1e-12 and w['column_chosen'] is None


def test_positions_put_a_tie_at_the_centre_of_its_cells():
    rho = np.repeat([0.052, 0.056, 0.060, 0.064, 0.068, 0.072], 5); q, at = positions(rho); assert list(q) == [0.052, 0.072] and list(at) == [2.0, 27.0]
    q, at = positions(-rho); assert list(at) == [2.0, 27.0]


def test_prefix_halfwidth_is_stable_at_a_large_offset():
    x = np.random.default_rng(0).normal(0, 1e-3, 50); m0, h0 = prefix_halfwidth(x); m, h = prefix_halfwidth(x - 56300.0)
    assert np.abs(h[1:] / h0[1:] - 1).max() < 1e-6 and np.abs(m + 56300.0 - m0).max() < 1e-9 and abs(h0[-1] - stats.t.ppf(0.975, 49) * x.std(ddof=1) / np.sqrt(50)) < 1e-15


def test_linear_change_on_an_exact_line():
    x = np.linspace(2.9, 3.2, 40); r = linear_change(x, 3 * x - 1, 2.92, 3.15, n_replicates=50)
    for k in ('theil_sen', 'least_squares'): assert abs(r[k]['change'] - 3 * 0.23) < 1e-9 and abs(r[k]['lower'] - r[k]['change']) < 1e-9 and abs(r[k]['upper'] - r[k]['change']) < 1e-9


def test_softmax_separates_classes():
    rng = np.random.default_rng(3); X = np.vstack([rng.normal(-2, 0.5, (40, 2)), rng.normal(2, 0.5, (40, 2))]); y = np.array(['a'] * 40 + ['b'] * 40)
    W = softmax_fit(standardise(X, X), y, ['a', 'b']); P = softmax_predict(W, standardise(X, X)); assert (np.array(['a', 'b'])[P.argmax(1)] == y).all()
    r = held_out_shares(X, y, X[:40], ['a', 'b'], n_replicates=20); assert r['predicted'][0] > 0.95


def test_quantile_of_nothing_raises():
    with pytest.raises(ValueError, match='quantile'): quantile([], 0.5)


def test_learning_curve_shares_sum_to_one_and_sharpen_with_cells():
    rng = np.random.default_rng(7); X = np.vstack([rng.normal(-2, 0.5, (30, 2)), rng.normal(2, 0.5, (30, 2))]); y = np.array(['a'] * 30 + ['b'] * 30); o = rng.permutation(60)
    rows = learning_curve(X[o[:40]], y[o[:40]], X[o[40:]], ['a', 'b'], (10, 40), n_replicates=30); by = {(r['n_train'], r['kind']): r for r in rows}
    assert len(rows) == 4 and all(abs(by[(n, 'a')]['predicted'] + by[(n, 'b')]['predicted'] - 1) < 1e-9 for n in (10, 40)) and all(r['lower'] <= r['predicted'] <= r['upper'] for r in rows)
    assert by[(40, 'a')]['upper'] - by[(40, 'a')]['lower'] <= by[(10, 'a')]['upper'] - by[(10, 'a')]['lower'] + 1e-12


def test_empty_inputs_raise():
    for call in (lambda: percentile_bootstrap([]), lambda: exact_lattice_bootstrap([]), lambda: wilson(0, 0), lambda: wilson(5, 3), lambda: cliff_delta([], [1.0]), lambda: location_shift([1.0], [1.0])):
        with pytest.raises(ValueError): call()


def test_too_few_or_non_finite_cells_raise():
    for call in (lambda: scale_ratio([1.0], [1.0, 2.0]), lambda: scale_ratio([1.0, 2.0], [1.0, np.nan]), lambda: percentile_bootstrap([1.0, np.nan])):
        with pytest.raises(ValueError): call()


def test_intervals_take_one_value_per_cell():
    sites = np.random.default_rng(0).uniform(size=(10, 50))
    for call in (lambda: percentile_bootstrap(sites), lambda: exact_lattice_bootstrap(np.round(sites)), lambda: bounded_prevalence(sites)):
        with pytest.raises(ValueError): call()


def test_contrasts_take_one_value_per_cell_and_report_the_replicates_used():
    two = np.ones((3, 4))
    for call in (lambda: cliff_delta(two, [1.0]), lambda: scale_ratio(two, [1.0, 2.0, 3.0]), lambda: location_shift(two, [1.0, 2.0]), lambda: scale_ratio([1.0, 2.0, 3.0], [2.0, 2.0, 2.0])):
        with pytest.raises(ValueError): call()
    r = scale_ratio([1.0, 2.0, 4.0], [1.0, 3.0, 4.0], n_replicates=500); assert 0 < r['n_replicates'] < 500 and 0 < r['lower'] and np.isfinite(r['upper'])
    for flat in ([2.0, 2.0, 2.0], [0.1 + 0.2, 0.3, 0.3]):
        with pytest.raises(ValueError, match='the population has no spread'): scale_ratio(flat, [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match='no resample has a spread in both populations'): scale_ratio([1.0, 1.0, 2.0], [1.0, 2.0, 3.0], seed=0, n_replicates=1)
    with pytest.raises(ValueError, match='the reference has no spread'): location_shift([1.0, 2.0], [0.1 + 0.2, 0.3, 0.3])
    far = 1e6 + np.array([0.0, 1.0, 2.0, 3.0]) * 1e-7; assert abs(scale_ratio(far, 2 * far - 1e6, n_replicates=200)['ratio'] - 0.5) < 1e-6 and location_shift(far, far) == 0
    x, y = [1.0, 2.0, 4.0, 7.0], [1.0, 3.0, 4.0, 9.0]; f, b = scale_ratio(x, y, n_replicates=4000), scale_ratio(y, x, n_replicates=4000); assert abs(np.log(f['lower'] * b['upper'])) < 0.15 and abs(np.log(f['upper'] * b['lower'])) < 0.15


def test_maxt_band_uses_sample_standard_errors_and_covers():
    rng = np.random.default_rng(4); x = rng.normal(size=8); e, inf = influence_mean(x); r = maxt_simultaneous(inf[:, None], np.array([e]), n_replicates=500)
    assert np.isclose(r['standard_errors'][0], x.std(ddof=1) / np.sqrt(8))
    hit = 0
    for k in range(150):
        y = rng.normal(size=(6, 3)); inf = np.column_stack([influence_mean(y[:, j])[1] for j in range(3)]); b = maxt_simultaneous(inf, y.mean(0), n_replicates=400, seed=k); hit += np.all((b['lower'] <= 0) & (0 <= b['upper']))
    assert hit / 150 >= 0.9
    with pytest.raises(ValueError, match='maxt_simultaneous: influence'): maxt_simultaneous(np.ones(5), np.ones(1))


def test_maxt_band_resamples_within_groups_of_cells():
    rng = np.random.default_rng(5); lab = np.repeat(np.arange(6), 5); hit = 0
    for k in range(120):
        x = rng.normal(size=30); inf = np.zeros((30, 6)); est = np.zeros(6)
        for g in range(6): est[g], inf[lab == g, g] = influence_mean(x[lab == g])
        b = maxt_simultaneous(inf, est, n_replicates=400, seed=k, groups=lab); hit += np.all((b['lower'] <= 0) & (0 <= b['upper']))
    assert hit / 120 >= 0.9
    e, inf = influence_mean(x[:10]); one = maxt_simultaneous(inf[:, None], np.array([e]), n_replicates=300, seed=2); assert one['critical_value'] == maxt_simultaneous(inf[:, None], np.array([e]), n_replicates=300, seed=2, groups=np.zeros(10))['critical_value']
    y = rng.normal(size=15) + np.repeat([0.0, 2.0, 4.0], 5); g = np.repeat(np.arange(3), 5); fam = np.zeros((15, 4)); val = np.zeros(4)
    for k in range(3): val[k], fam[g == k, k] = influence_mean(y[g == k])
    val[3], fam[:, 3] = influence_mean(y); sd = np.array([y[g == k].std(ddof=1) for k in range(3)]); b = maxt_simultaneous(fam, val, n_replicates=200, groups=g)
    assert np.allclose(b['standard_errors'], np.r_[sd / np.sqrt(5), np.sqrt((5 * sd ** 2).sum()) / 15])
    assert np.isfinite(maxt_simultaneous(fam[:12, :3], val[:3], n_replicates=50, groups=np.repeat(np.arange(3), 4))['critical_value'])
    for bad, msg in ((np.r_[np.zeros(7), np.ones(3)], 'at least four cells'), (np.zeros(9), 'one label per cell')):
        with pytest.raises(ValueError, match=msg): maxt_simultaneous(inf[:, None], np.array([e]), groups=bad)


def test_quantile_is_exact_at_float_products_and_inputs_are_checked():
    assert quantile(np.arange(100.0), 0.07) == 6.0 and quantile(np.arange(10000.0), 0.07) == 699.0
    for call, msg in ((lambda: percentile_bootstrap([1.0, 2.0], alpha=1.5), 'alpha'), (lambda: positions([]), 'positions'), (lambda: linear_change([0, 1, 2, 3], [0, 1, np.nan, 3], 0, 3), 'linear_change'),
                      (lambda: scale_ratio([1.0, 2.0, 3.0], [1.0, 2.0, np.nan]), 'finite')):
        with pytest.raises(ValueError, match=msg): call()


def _left_inverse(sorted_reps, p):
    return sorted_reps[int(np.ceil(p * len(sorted_reps))) - 1]


def test_prevalence_with_sites_takes_the_korn_graubard_effective_count():
    m = np.array([10, 20, 40, 20, 10, 40, 20, 20]); k = np.array([0, 6, 2, 1, 4, 13, 0, 3]); v = k / m; n = v.size; N = n * n / np.sum(1 / m); x = v.mean()
    ne = x * (1 - x) / (v.var(ddof=1) / n) * (stats.t.ppf(0.975, N - 1) / stats.t.ppf(0.975, n - 1)) ** 2; r = bounded_prevalence(v, sites=m)
    assert ne < N and r['kind'] == 'effective_clopper_pearson' and abs(r['n_effective'] - ne) < 1e-12 * ne and abs(r['estimate'] - x) < 1e-15
    assert abs(r['lower'] - stats.beta.ppf(0.025, x * ne, ne - x * ne + 1)) < 1e-12 and abs(r['upper'] - stats.beta.ppf(0.975, x * ne + 1, ne - x * ne)) < 1e-12
    assert bounded_prevalence([0.0, 1.0, 1.0], sites=[1, 1, 1])['n_sites'] == 3
    with pytest.raises(ValueError, match='bounded_prevalence: sites'): bounded_prevalence(v, sites=np.r_[m[:-1], 20.5])


def test_wilson_bounds_are_the_roots_of_the_score_equation():
    z = intervals.Z_975
    for k, n in ((3, 10), (1, 1), (0, 1), (7, 7), (12, 40)):
        p = k / n; assert abs(np.array(wilson(k, n)) - np.sort(np.roots([1 + z * z / n, -(2 * p + z * z / n), p * p]).real)).max() < 1e-12, (k, n)


def test_percentile_bootstrap_takes_left_inverse_quantiles_of_seeded_whole_cell_resamples():
    v = np.random.default_rng(9).gamma(2.0, 1.0, 13)
    for kw, (B, seed, alpha) in ((dict(), (10000, 0, 0.05)), (dict(n_replicates=999, seed=3), (999, 3, 0.05)), (dict(n_replicates=401, seed=5, alpha=0.2), (401, 5, 0.2))):
        rng = np.random.default_rng(seed); reps = np.sort([v[rng.integers(0, v.size, v.size)].mean() for _ in range(B)]); r = percentile_bootstrap(v, **kw)
        assert r['lower'] == _left_inverse(reps, alpha / 2) and r['upper'] == _left_inverse(reps, 1 - alpha / 2) and (r['n_replicates'], r['seed'], r['n_cells']) == (B, seed, 13)
    assert quantile([4.0, 1.0, 3.0, 2.0], 0.25) == quantile([4.0, 1.0, 3.0, 2.0], 0.0) == 1.0 and quantile([4.0, 1.0, 3.0, 2.0], 1.0) == 4.0


def test_clopper_pearson_lower_bound_of_one_success_is_closed_form():
    for n in (1, 7, 40):
        assert abs(clopper_pearson(1, n)[0] - (1 - 0.975 ** (1 / n))) < 1e-12 and abs(clopper_pearson(1, n, alpha=0.1)[0] - (1 - 0.95 ** (1 / n))) < 1e-12
    for k, n in ((0, 0), (5, 3), (-1, 3)):
        with pytest.raises(ValueError, match='clopper_pearson'): clopper_pearson(k, n)


def test_contrast_intervals_are_left_inverse_quantiles_of_seeded_resamples_of_both_populations():
    a, b = np.array([-0.65, -0.17, 1.66, 0.66, -1.64, -0.01, -0.62, 0.15, -1.61, 0.24, 0.24, 1.58]), np.array([0.82, 1.01, -0.99, 2.75, -1.42, 1.6, 0.17, -0.38, -0.16, -0.17])
    for kw, (B, seed, alpha) in ((dict(), (1000, 0, 0.05)), (dict(seed=4, n_replicates=301, alpha=0.2), (301, 4, 0.2))):
        rng = np.random.default_rng(seed); reps = np.sort([np.sign(a[rng.integers(0, a.size, a.size)][:, None] - b[rng.integers(0, b.size, b.size)][None]).mean() for _ in range(B)]); r = cliff_delta_interval(a, b, **kw)
        assert abs(r['delta'] - np.sign(a[:, None] - b[None]).mean()) < 1e-12 and abs(r['lower'] - _left_inverse(reps, alpha / 2)) < 1e-12 and abs(r['upper'] - _left_inverse(reps, 1 - alpha / 2)) < 1e-12
    a, b = np.array([1.0, 2.0, 4.0, 7.0]), np.array([1.0, 3.0, 4.0, 9.0, 4.0])
    for kw, (B, seed, alpha) in ((dict(), (4000, 0, 0.05)), (dict(seed=2, n_replicates=777, alpha=0.1), (777, 2, 0.1))):
        rng = np.random.default_rng(seed); sd = np.array([(a[rng.integers(0, a.size, a.size)].std(ddof=1), b[rng.integers(0, b.size, b.size)].std(ddof=1)) for _ in range(B)]); ok = (sd > 0).all(1); reps = np.sort(sd[ok, 0] / sd[ok, 1]); r = scale_ratio(a, b, **kw)
        assert 0 < ok.sum() < B and r['n_replicates'] == ok.sum() and r['lower'] == _left_inverse(reps, alpha / 2) and r['upper'] == _left_inverse(reps, 1 - alpha / 2) and abs(r['ratio'] - a.std(ddof=1) / b.std(ddof=1)) < 1e-15
    assert location_shift([3.0, 4.0, 5.0], [0.0, 2.0, 4.0]) == 1.0 and abs(location_shift([3.0], [0.0, 2.0]) - 2 / np.sqrt(2)) < 1e-15
    with pytest.raises(ValueError, match='cliff_delta_interval: needs one finite value per cell'): cliff_delta_interval([1.0, np.nan], [1.0, 2.0])


def test_maxt_critical_value_is_the_studentized_bootstrap_quantile_and_stays_finite():
    rng = np.random.default_rng(9); x = rng.normal(size=8); e, inf = influence_mean(x); r = maxt_simultaneous(inf[:, None], np.array([e]), n_replicates=500, seed=3)
    R = inf[np.random.default_rng(3).integers(0, 8, (500, 8))]; t = np.abs(R.sum(1)) / np.sqrt(((R - R.mean(1, keepdims=True)) ** 2).sum(1) * 8 / 7)
    assert r['critical_value'] == quantile(t, 0.95) and r['n_without_spread'] == 0
    for n, k in ((30, 1), (30, 2), (60, 2)):
        y = np.c_[rng.normal(size=n), np.r_[np.ones(k), np.zeros(n - k)]]; inf = np.column_stack([influence_mean(y[:, j])[1] for j in range(2)])
        b = maxt_simultaneous(inf, y.mean(0), n_replicates=2000); assert np.isfinite(b['critical_value']) and b['critical_value'] < 4 and b['n_without_spread'] > 0
    with pytest.raises(ValueError, match='at least four cells'): maxt_simultaneous(inf[:3], np.ones(2))


def test_linear_change_skips_resamples_with_one_x_value():
    r = linear_change([0.0, 0.0, 1.0], [1.0, 1.2, 2.0], 0.0, 1.0, n_replicates=200); assert np.isfinite(r['least_squares']['lower']) and np.isfinite(r['theil_sen']['upper'])
