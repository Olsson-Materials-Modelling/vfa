"""Total scattering against known answers: explicit reciprocal sums, the ideal gas, lattices, the two routes, the real-space round trip, the FSDP."""
import itertools, os, subprocess, sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lattices import cubic
from vfa.structure.cell import minimum_image
from vfa.structure.scattering import (concentrations, sine_transform, pair_correlations, transform_partials, direct_partials, ensemble_direct, ensemble_transform,
                                      total_sq, total_gr, real_space, fsdp)


def rsa(n, L, dmin, seed):
    """Random sequential addition of n points at least dmin apart in a cubic cell of side L."""
    rng = np.random.default_rng(seed); pts = np.empty((0, 3)); cell = np.eye(3) * L
    while len(pts) < n:
        x = rng.random(3) * L
        if not len(pts) or np.linalg.norm(minimum_image(pts - x, cell), axis=1).min() >= dmin: pts = np.vstack([pts, x])
    return pts, cell


def brute(sym, pos, cell, q_max, dq, w=None):
    """Shell means of S_ab(G) and of |sum_j w_j exp(i G.r_j)|^2 / N over 0 < |G| <= q_max, by explicit exponentials."""
    B = 2 * np.pi * np.linalg.inv(cell).T; H = np.ceil(q_max * np.linalg.norm(cell, axis=1) / (2 * np.pi)).astype(int) + 1
    h = np.array(np.meshgrid(*[np.arange(-k, k + 1) for k in H], indexing='ij')).reshape(3, -1).T; G = h @ B; Q = np.linalg.norm(G, axis=1); keep = (Q > 0) & (Q <= q_max); G, Q = G[keep], Q[keep]
    E = np.exp(1j * pos @ G.T); c = concentrations(sym); sp = list(c); A = {s: E[sym == s].sum(0) for s in sp}; N = len(sym); k = (Q / dq).astype(int); ks = np.unique(k)
    mean = lambda v: np.array([v[k == i].mean() for i in ks])
    S = {(a, b): 1 + (mean(abs(A[a]) ** 2) / N - c[a]) / c[a] ** 2 if a == b else 1 + mean((A[a] * A[b].conj()).real) / (N * c[a] * c[b]) for i, a in enumerate(sp) for b in sp[i:]}
    I = None if w is None else mean(abs(sum(np.broadcast_to(np.asarray(w[s](Q) if callable(w[s]) else w[s], float), Q.shape) * A[s] for s in sp)) ** 2) / N
    return mean(Q), S, np.array([(k == i).sum() for i in ks]), I


def two_species(n_a, n_b):
    """Symbols of n_a atoms A followed by n_b atoms B."""
    return np.array(['A'] * n_a + ['B'] * n_b)


def test_direct_matches_explicit_sum_in_a_triclinic_cell():
    rng = np.random.default_rng(3); cell = np.array([[7.1, 0, 0], [1.9, 6.4, 0], [-1.3, 2.2, 8.0]]); pos = rng.random((30, 3)) @ cell + 20.0; sym = two_species(11, 19)
    q, S, m = direct_partials(sym, pos, cell, 6.0, dq=0.3, chunk=30 * 7); qb, Sb, mb, _ = brute(sym, pos, cell, 6.0, 0.3)
    assert np.array_equal(m, mb) and abs(q - qb).max() < 1e-12 and all(abs(S[p] - Sb[p]).max() < 1e-10 for p in Sb)


def test_weights_recombine_to_the_total_intensity():
    pos, cell = rsa(40, 8.0, 1.6, 5); sym = two_species(15, 25); c = concentrations(sym); q, S, _ = direct_partials(sym, pos, cell, 7.0, dq=0.002)
    neutron = {'A': 1.7, 'B': -0.6}; xray = {'A': lambda Q: 14 * np.exp(-0.03 * Q ** 2) + 2, 'B': lambda Q: 8 * np.exp(-0.05 * Q ** 2)}
    for w in (neutron, xray, {s: xray[s](q) for s in xray}):
        F, St = total_sq(q, S, c, w); qb, _, _, I = brute(sym, pos, cell, 7.0, 0.002, {s: (w[s] if not isinstance(w[s], np.ndarray) else xray[s]) for s in w})
        wq = {s: np.broadcast_to(np.asarray(w[s](q) if callable(w[s]) else w[s], float), q.shape) for s in w}; b2 = sum(c[s] * wq[s] ** 2 for s in c); b1 = sum(c[s] * wq[s] for s in c)
        assert abs(qb - q).max() < 1e-12 and abs(F + b2 - I).max() < 1e-9 * abs(I).max() and abs(St - 1 - F / b1 ** 2).max() < 1e-12
    for X in (0.3, 2.5):
        F, St = total_sq(q, {p: np.full(q.shape, X) for p in S}, c, neutron); assert abs(St - X).max() < 1e-12 and abs(F - (sum(c[s] * neutron[s] for s in c)) ** 2 * (X - 1)).max() < 1e-12
    with pytest.raises(ValueError, match='null scattering'): total_sq(q, S, c, {'A': 5.0, 'B': -3.0})
    with pytest.raises(ValueError, match='no weight for species B'): total_sq(q, S, c, {'A': 1.0})


def test_ideal_gas_partials_are_one():
    rng = np.random.default_rng(11); L = 25.0; cell = np.eye(3) * L; sym = two_species(500, 1000); pos = rng.random((1500, 3)) * L; c = concentrations(sym)
    _, S, m = direct_partials(sym, pos, cell, 6.0, dq=0.25); half = m / 2
    sd = {('A', 'A'): 1 / c['A'], ('B', 'B'): 1 / c['B'], ('A', 'B'): 1 / np.sqrt(2 * c['A'] * c['B'])}
    for p, s in sd.items():
        z = (S[p] - 1) / (s / np.sqrt(half)); assert abs(z).max() < 5 and 0.6 < np.mean(z ** 2) < 1.5 and abs(np.sum(half * (S[p] - 1)) / np.sum(half)) < 4 * s / np.sqrt(half.sum())


def test_diamond_reflections_only_where_allowed():
    a = 3.567; sym, pos, cell = cubic('diamond', a, 2); N = len(sym); q, S, _ = direct_partials(sym, pos, cell, 9.0, dq=0.005); S = S[('A', 'A')]
    h = np.array(np.meshgrid(*[np.arange(-11, 12)] * 3, indexing='ij')).reshape(3, -1).T; Q = 2 * np.pi * np.linalg.norm(h, axis=1) / (2 * a); keep = (Q > 0) & (Q <= 9.0); h, Q = h[keep], Q[keep]
    hkl = h / 2; integer = np.all(h % 2 == 0, axis=1); odd = integer & np.all(np.abs(hkl) % 2 == 1, axis=1); even4 = integer & np.all(np.abs(hkl) % 2 == 0, axis=1) & (hkl.sum(1) % 4 == 0); weight = 0.5 * odd + 1.0 * even4
    shells = np.unique(np.round(Q, 9)); expect = np.array([N * weight[np.isclose(Q, s)].mean() for s in shells])
    assert np.allclose(q, shells, atol=1e-9) and abs(S - expect).max() < 1e-9
    assert abs(q[np.flatnonzero(S > 1e-6)[0]] - 2 * np.pi * np.sqrt(3) / a) < 1e-12 and abs(S[np.argmin(abs(q - 2 * np.pi * np.sqrt(3) / a))] - N / 2) < 1e-9 and abs(S[np.argmin(abs(q - 2 * np.pi * np.sqrt(8) / a))] - N) < 1e-9 and abs(S[np.argmin(abs(q - 4 * np.pi / a))]) < 1e-9
    sym2, _, _ = cubic('diamond', a, 2, two_species=True); q2, S2, _ = direct_partials(sym2, pos, cell, 9.0, dq=0.005); b = {'A': 1.7, 'B': -0.6}; F, _ = total_sq(q2, S2, concentrations(sym2), b); b2 = (b['A'] ** 2 + b['B'] ** 2) / 2
    at = lambda hkl: F[np.argmin(abs(q2 - 2 * np.pi * np.sqrt(hkl) / a))]
    assert abs(at(3) - (N / 4 * (b['A'] ** 2 + b['B'] ** 2) - b2)) < 1e-9 and abs(at(4) - (N / 4 * (b['A'] - b['B']) ** 2 - b2)) < 1e-9 and abs(at(8) - (N / 4 * (b['A'] + b['B']) ** 2 - b2)) < 1e-9


def test_long_wavelength_limit_follows_the_density_fluctuations():
    a = 2.5; sym, pos, cell = cubic('sc', a, 8); N = len(sym); q, S, m = direct_partials(sym, pos, cell, 2 * np.pi / a * 1.01, dq=0.01); S = S[('A', 'A')]; bragg = np.isclose(q, 2 * np.pi / a)
    assert abs(S[~bragg]).max() < 1e-10 and abs(S[bragg] - N * 6 / m[bragg]).max() < 1e-9
    rng = np.random.default_rng(2); sigma = 0.08; ratio = []
    for _ in range(4):
        _, Sd, md = direct_partials(sym, pos + rng.normal(0, sigma, pos.shape), cell, 2 * np.pi / a * 0.99, dq=0.01); ratio.append((md * Sd[('A', 'A')]).sum() / (md * (1 - np.exp(-(q[:len(md)] * sigma) ** 2))).sum())
        assert Sd[('A', 'A')][0] < 0.05
    assert abs(np.mean(ratio) - 1) < 0.1


def test_transform_is_the_direct_route_through_the_window():
    pos, cell = rsa(64, 9.0, 1.9, 1); sym = two_species(24, 40); V = 9.0 ** 3; R = 4.5; qs = np.array([1.0, 2.0, 3.0, 4.0, 6.0, 8.0])
    T = transform_partials(sym, pos, cell, qs, n_bins=4000, window='lorch'); q, S, m = direct_partials(sym, pos, cell, 30.0, dq=0.002); assert np.diff(q).min() > 0.005
    r = (np.arange(20000) + 0.5) * R / 20000; K = (np.sinc(np.outer(qs, r) / np.pi) * (4 * np.pi * r ** 2 * np.sinc(r / R) * R / 20000)) @ np.sinc(np.outer(r, q) / np.pi)
    for p in S: assert abs(T[p] - 1 - K @ (m * (S[p] - 1)) / V).max() < 1e-3
    rr, g = pair_correlations(sym, pos, cell, n_bins=4000); textbook = 1 + sine_transform(rr, g[('A', 'A')] - 1, qs, 64 / V, 'lorch')
    assert abs(textbook - 1 - K @ (m * (S[('A', 'A')] - 1)) / V).max() > 1e-2


def test_routes_approach_one_at_large_q():
    pos, cell = rsa(150, 12.0, 1.8, 7); sym = two_species(50, 100); c = concentrations(sym); b = {'A': 1.7, 'B': 0.6}
    q, S, m = direct_partials(sym, pos, cell, 20.0, dq=0.1); _, St = total_sq(q, S, c, b); hi = q > 15; qt = np.linspace(15, 20, 501); T = transform_partials(sym, pos, cell, qt, window='lorch'); _, Stt = total_sq(qt, T, c, b)
    assert all(abs(np.sum(m[hi] * (S[p][hi] - 1)) / m[hi].sum()) < 0.01 and abs(np.mean(T[p] - 1)) < 0.01 for p in S) and abs(np.sum(m[hi] * (St[hi] - 1)) / m[hi].sum()) < 0.01 and abs(np.mean(Stt - 1)) < 0.01


def test_real_space_round_trip():
    pos, cell = rsa(60, 9.0, 1.9, 4); sym = two_species(20, 40); c = concentrations(sym); b = {'A': 1.7, 'B': -0.6}; rho = 60 / 9.0 ** 3; R = 4.5; n = 500
    r, G = total_gr(sym, pos, cell, b, n_bins=n); Q = (np.arange(n) + 0.5) * np.pi / R; F = sine_transform(r, G, Q, rho)
    Fp, _ = total_sq(Q, transform_partials(sym, pos, cell, Q, n_bins=n), c, b); assert abs(F - Fp).max() < 1e-10 * abs(F).max()
    mean_b = sum(c[s] * b[s] for s in c); out = real_space(Q, F, r, rho, mean_b); assert abs(out['G'] - G).max() < 1e-10
    assert np.allclose(out['D'], 4 * np.pi * rho * r * out['G'], atol=1e-12) and np.allclose(out['T'], out['D'] + 4 * np.pi * rho * r * mean_b ** 2, atol=1e-12)
    low = r < 1.8; b2 = sum(c[s] * b[s] ** 2 for s in c); assert abs(G[low] - (-mean_b ** 2 + b2 / 60)).max() < 1e-12
    with pytest.raises(ValueError, match='exceeds half the shortest lattice vector'): transform_partials(sym, pos, cell, Q, r_max=4.6)
    with pytest.raises(ValueError, match="window must be None or 'lorch'"): sine_transform(r, G, Q, rho, window='hann')
    with pytest.raises(ValueError, match='one number per species'): total_gr(sym, pos, cell, {'A': lambda x: x, 'B': 1.0})


def test_ensembles_weight_cells_equally():
    cells = [(two_species(20, 40), *rsa(60, 9.0, 1.9, s)) for s in (1, 2)]; runs = [direct_partials(s, p, c, 5.0) for s, p, c in cells]
    q, S, c = ensemble_direct(cells, 5.0); assert abs(q - runs[0][0]).max() < 1e-12 and all(abs(S[p] - 0.5 * (runs[0][1][p] + runs[1][1][p])).max() < 1e-12 for p in S) and c == {'A': 1 / 3, 'B': 2 / 3}
    other = (two_species(25, 50), *rsa(75, 10.0, 1.9, 3)); q2, S2, _ = ensemble_direct([cells[0], other], 5.0); r2 = direct_partials(*other, 5.0)
    k0, k1 = np.floor(runs[0][0] / 0.05).astype(int), np.floor(r2[0] / 0.05).astype(int); common = np.intersect1d(k0, k1); i0, i1 = np.searchsorted(k0, common), np.searchsorted(k1, common)
    assert len(q2) == len(common) < min(len(k0), len(k1)) and all(abs(S2[p] - 0.5 * (runs[0][1][p][i0] + r2[1][p][i1])).max() < 1e-12 for p in S2)
    qt = np.linspace(0.5, 10, 20); T, _ = ensemble_transform(cells, qt); Ti = [transform_partials(s, p, cl, qt) for s, p, cl in cells]; assert all(abs(T[p] - 0.5 * (Ti[0][p] + Ti[1][p])).max() < 1e-12 for p in T)
    with pytest.raises(ValueError, match='do not share one set of species'): ensemble_direct([cells[0], (np.array(['A'] * 60), cells[1][1], cells[1][2])], 5.0)


def test_fsdp_of_a_gaussian_and_unresolved_peaks():
    q = np.arange(1, 201) * 0.02; S = 0.8 + 0.05 * q + 0.6 * np.exp(-(q - 1.6) ** 2 / (2 * 0.15 ** 2)); out = fsdp(q, S, (1.0, 2.2))
    assert abs(out['position_invA'] - 1.6019) < 2e-3 and abs(out['height_above_baseline'] - 0.6) < 2e-3 and abs(out['fwhm_invA'] - 2 * np.sqrt(2 * np.log(2)) * 0.15) < 3e-3
    with pytest.raises(ValueError, match='window edge'): fsdp(q, 0.8 + 0.05 * q, (1.0, 2.2))
    with pytest.raises(ValueError, match='single point above half height'): fsdp(q, 1.0 + (np.arange(200) == 79) * 0.5, (1.0, 2.2))
    with pytest.raises(ValueError, match='at least 5 are needed'): fsdp(q, S, (1.6, 1.65))


def test_transform_does_not_depend_on_the_basis():
    pos, cell = rsa(64, 9.0, 1.9, 1); sym = two_species(16, 48); qs = np.array([1.5, 2.5, 4.0]); skew = np.array([[1, 0, 0], [2, 1, 0], [0, 0, 1]]) @ cell
    T0, T1 = transform_partials(sym, pos, cell, qs, window='lorch'), transform_partials(sym, pos, skew, qs, window='lorch'); E0, _ = ensemble_transform([(sym, pos, cell)], qs); E1, _ = ensemble_transform([(sym, pos, skew)], qs)
    assert all(abs(T0[p] - T1[p]).max() < 1e-10 and abs(E0[p] - E1[p]).max() < 1e-10 for p in T0) and set(pair_correlations(sym, pos, skew)[1]) == set(T0)
    assert abs(pair_correlations(sym, pos, skew, n_bins=100)[0][-1] - 4.5 + 4.5 / 200) < 1e-12


def test_direct_partials_do_not_depend_on_the_blas_threads():
    code = ("import sys, hashlib; sys.path.insert(0, sys.argv[1]); from vfa.io.extxyz import read_extxyz; from vfa.structure.scattering import direct_partials; "
            "s, p, c = read_extxyz(sys.argv[2]); q, S, m = direct_partials(s, p, c, 8.0); print(hashlib.sha256(q.tobytes() + b''.join(S[k].tobytes() for k in sorted(S))).hexdigest())")
    root = Path(__file__).resolve().parents[1]; cell = root / 'example/data/structures/rho0.060_1.extxyz'
    out = {t: subprocess.run([sys.executable, '-c', code, str(root), str(cell)], env={**os.environ, 'OPENBLAS_NUM_THREADS': t, 'OMP_NUM_THREADS': t, 'MKL_NUM_THREADS': t}, capture_output=True, text=True, check=True).stdout for t in ('1', '4')}
    assert out['1'] == out['4'] and len(out['1'].strip()) == 64


def test_invalid_scattering_inputs_raise():
    pos, cell = rsa(20, 8.0, 1.6, 2); sym = two_species(8, 12); q = np.arange(1, 201) * 0.02; S = 0.8 + 0.6 * np.exp(-(q - 1.6) ** 2 / 0.045); bad = pos.copy(); bad[0, 0] = np.inf
    with pytest.raises(ValueError, match='fsdp: q and S'): fsdp(q[::-1], S[::-1], (1.0, 2.2))
    with pytest.raises(ValueError, match='fsdp: q and S'): fsdp(q, np.where(q == q[90], np.nan, S), (1.0, 2.2))
    with pytest.raises(ValueError, match='direct_partials: positions must be finite'): direct_partials(sym, bad, cell, 5.0)
    with pytest.raises(ValueError, match='direct_partials: one symbol per atom'): direct_partials(sym[:0], pos[:0], cell, 5.0)
    with pytest.raises(ValueError, match='transform_partials: positions must be finite'): transform_partials(sym, bad, cell, q)
    flat = np.diag([8.0, 8.0, 0.0]); b = {'A': 1.0, 'B': 2.0}
    for name, call in (('transform_partials', lambda: transform_partials(sym, pos, flat, q)), ('pair_correlations', lambda: pair_correlations(sym, pos, flat)), ('total_gr', lambda: total_gr(sym, pos, flat, b)),
                       ('direct_partials', lambda: direct_partials(sym, pos, flat, 5.0)), ('ensemble_direct', lambda: ensemble_direct([(sym, pos, flat)], 5.0)), ('ensemble_transform', lambda: ensemble_transform([(sym, pos, cell), (sym, pos, flat)], q))):
        with pytest.raises(ValueError, match=f'{name}: the cell must be a finite, non-singular'): call()


def test_total_sq_needs_each_species_pair_once():
    q = np.linspace(0.5, 10, 50); c = {'A': 0.4, 'B': 0.6}; b = {'A': 1.0, 'B': 2.0}; S = {('A', 'A'): 1 + 0.1 * np.sin(q), ('A', 'B'): 1 - 0.2 * np.cos(q), ('B', 'B'): 1 + 0.3 * np.sin(2 * q)}
    F, St = total_sq(q, S, c, b); F2, St2 = total_sq(q, {p: S[p] for p in reversed(list(S))}, {'B': 0.6, 'A': 0.4}, b); assert np.array_equal(F, F2) and np.array_equal(St, St2)
    for bad in ({p: S[p] for p in S if p != ('B', 'B')}, {**S, ('B', 'A'): S[('A', 'B')]}, {('A', 'A'): S[('A', 'A')], ('B', 'A'): S[('A', 'B')], ('B', 'B'): S[('B', 'B')]}, {**S, ('A', 'C'): S[('A', 'B')]}):
        with pytest.raises(ValueError, match='total_sq: partials must hold each species pair of conc once'): total_sq(q, bad, c, b)
    with pytest.raises(ValueError, match='total_sq: no weight for species B'): total_sq(q, S, c, {'A': 1.0})


def test_numeric_species_match_symbols_and_a_flat_topped_peak_is_resolved():
    _, pos, cell = cubic('sc', 3.0, 4); sym = np.where(np.arange(len(pos)) % 3 == 0, 'A', 'B'); g = pair_correlations(sym, pos, cell)[1]; gn = pair_correlations(np.where(sym == 'A', 14, 8), pos, cell)[1]
    assert len(g) == len(gn) == 3 and all(np.array_equal(gn[k], g[m]) for k, m in ((('14', '14'), ('A', 'A')), (('14', '8'), ('A', 'B')), (('8', '8'), ('B', 'B'))))
    q = np.linspace(1.0, 2.2, 121); S = 1.0 + 0.5 * (np.abs(q - 1.6) <= 0.015)
    assert abs(fsdp(q, S, (1.0, 2.2))['position_invA'] - 1.6) < 1e-12
    with pytest.raises(ValueError, match='number density must be positive'): real_space(q, S - 1, np.linspace(0.1, 5, 50), 0.0, 1.0)


def _explicit_shells(sym, pos, cell, q_max, dq):
    B = 2 * np.pi * np.linalg.inv(cell).T; H = np.ceil(q_max * np.linalg.norm(cell, axis=1) / (2 * np.pi)).astype(int) + 2; h = np.array(list(itertools.product(*[range(-k, k + 1) for k in H]))); G = h @ B; Q = np.linalg.norm(G, axis=1); keep = (Q > 0) & (Q <= q_max); G, Q = G[keep], Q[keep]
    E = np.exp(1j * pos @ G.T); c = concentrations(sym); sp = list(c); A = {s: E[sym == s].sum(0) for s in sp}; k = (Q / dq).astype(int); ks = np.unique(k); mean = lambda v: np.array([v[k == i].mean() for i in ks])
    S = {(a, b): 1 + (mean(abs(A[a]) ** 2) / len(sym) - c[a]) / c[a] ** 2 if a == b else 1 + mean((A[a] * A[b].conj()).real) / (len(sym) * c[a] * c[b]) for i, a in enumerate(sp) for b in sp[i:]}
    return mean(Q), S, np.array([(k == i).sum() for i in ks])


def test_direct_partials_take_every_vector_when_a_is_the_shortest_lattice_vector():
    rng = np.random.default_rng(6); cell = np.array([[4.6, 0, 0], [1.9, 7.4, 0], [-1.3, 2.2, 8.0]]); pos = rng.random((24, 3)) @ cell; sym = np.array(['A'] * 9 + ['B'] * 15)
    q, S, m = direct_partials(sym, pos, cell, 5.0, dq=0.25); qb, Sb, mb = _explicit_shells(sym, pos, cell, 5.0, 0.25)
    assert np.array_equal(m, mb) and abs(q - qb).max() < 1e-12 and all(abs(S[p] - Sb[p]).max() < 1e-10 for p in Sb)


def test_direct_partials_include_the_shell_at_q_max():
    for L in (4.3, 7.3, 8.6, 11.7, 2 * np.pi):
        q_max = 2 * np.pi * 3 / L; q, S, m = direct_partials(['A'], np.zeros((1, 3)), np.eye(3) * L, q_max, dq=0.05)
        assert m.sum() == 122 and m[-1] == 30 and abs(q[-1] - q_max) < 1e-12 and abs(S[('A', 'A')] - 1).max() < 1e-12


def test_ensemble_transform_stops_at_the_smallest_cell_limit():
    rng = np.random.default_rng(8); cells = [(np.array(['A'] * 10 + ['B'] * 20), rng.random((30, 3)) * 9.0, np.eye(3) * 9.0), (np.array(['A'] * 14 + ['B'] * 28), rng.random((42, 3)) * 10.0, np.eye(3) * 10.0)]; q = np.linspace(0.5, 10, 20)
    T, c = ensemble_transform(cells, q); one = [transform_partials(s, p, cl, q, r_max=4.5) for s, p, cl in cells]
    assert c == {'A': 1 / 3, 'B': 2 / 3} and all(abs(T[k] - 0.5 * (one[0][k] + one[1][k])).max() < 1e-12 for k in T)


def test_fsdp_of_a_parabolic_peak_on_a_curved_background():
    q = np.arange(1, 513) / 128; qc, k, H, c, q0 = 1.75, 0.05, 0.6, 6.0, 1.7 + 1 / 300; S = 1 - k * (q - qc) ** 2 + np.maximum(0, H - c * (q - q0) ** 2); out = fsdp(q, S, (1.0, 2.5))
    top = (k * qc + c * q0) / (k + c); p = int(np.argmax(S)); base = 1 - k * 0.75 ** 2; h = S[p] - base; y = S - base; L, R = int(np.flatnonzero(q == 1.0)[0]), int(np.flatnonzero(q == 2.5)[0])
    ql, qr = np.interp(h / 2, y[L:p + 1], q[L:p + 1]), np.interp(h / 2, y[p:R + 1][::-1], q[p:R + 1][::-1]); half = np.sqrt((1 - k * (top - qc) ** 2 + H - c * (top - q0) ** 2 - base - h / 2) / (k + c))
    assert abs(out['position_invA'] - top) < 1e-12 and out['baseline_invA'] == (1.0, 2.5) and abs(out['height_above_baseline'] - h) < 1e-12 and abs(out['height'] - S[p]) < 1e-15
    assert abs(out['half_height_invA'][0] - ql) < 1e-12 and abs(out['half_height_invA'][1] - qr) < 1e-12 and abs(out['fwhm_invA'] - (qr - ql)) < 1e-12 and abs(out['fwhm_invA'] - 2 * half) < 2e-4
