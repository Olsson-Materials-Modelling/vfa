"""Closed-form lattices, independent references for the pore network, periodic invariance and the probe sweep."""
import itertools, json, os, subprocess, sys
from pathlib import Path
import numpy as np
import pytest
from scipy import ndimage
from scipy.spatial import cKDTree
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lattices import cubic
from vfa.structure.cell import images
from vfa.voids.clearance import clearance_field, accessible_fraction, void_surface, probe_sweep, sample_clearance, clearance_cdf, exact_clearance
from vfa.io.extxyz import read_extxyz
from vfa.voids import network
from vfa.voids.network import voronoi_network, descriptors, free_sphere, pore_network, percolating_edges, _Offsets, _Space, _largest_clearance
from vfa.voids.sites import atom_free_volume, cavity_distance
from vfa.voids.radii import Radii, mass_density


def test_simple_cubic_closed_form():
    _, pos, cell = cubic('sc', n=4); a = 3.0; net = voronoi_network(pos, cell, np.zeros(len(pos))); d = descriptors(net)
    assert abs(d['Di_A'] - a * np.sqrt(3)) < 1e-9 and abs(d['Df_A'] - a * np.sqrt(2)) < 1e-9 and d['exact'] and d['percolates'] and (d['n_nodes'], d['n_edges']) == (64, 192)
    assert set(net) == {'n_nodes', 'canon_clear', 'canon_xyz', 'edges', 'neck', 'n_uncertified_nodes', 'n_uncertified_edges', 'complete', 'n_nodes_enumerated', 'n_nodes_added', 'n_isolated_edges'}
    assert d['complete'] is True and (d['n_nodes_enumerated'], d['n_nodes_added'], d['n_isolated_edges']) == (64, 0, 0)


def test_fcc_closed_form():
    _, pos, cell = cubic('fcc'); a = 3.0; d = descriptors(voronoi_network(pos, cell, np.zeros(len(pos))))
    assert abs(d['Di_A'] - a) < 1e-9 and abs(d['Df_A'] - 2 * a / np.sqrt(6)) < 1e-9 and d['exact'] and (d['n_nodes'], d['n_edges']) == (324, 864) and (d['n_nodes_enumerated'], d['n_nodes_added']) == (324, 0)


def test_radii_shrink_the_spheres():
    _, pos, cell = cubic('sc', n=4); r = 0.4; d0 = descriptors(voronoi_network(pos, cell, np.zeros(len(pos)))); d1 = descriptors(voronoi_network(pos, cell, np.full(len(pos), r)))
    assert abs((d0['Di_A'] - d1['Di_A']) - 2 * r) < 1e-9 and abs((d0['Df_A'] - d1['Df_A']) - 2 * r) < 1e-9


def test_cscl_unequal_radii_closed_form():
    a, n = 4.0, 3; g = np.array([(i, j, k) for i in range(n) for j in range(n) for k in range(n)], float) * a; pos = np.r_[g, g + a / 2]; cell = np.eye(3) * a * n
    for rA, rB, Di in ((1.5, 0.3, 2.7), (1.2, 0.2, 3.0 + 4 / 15)):
        c = a / 2 + rA - rB; t = (c * c - a * a / 2) / (2 * a * c); d = descriptors(voronoi_network(pos, cell, np.r_[np.full(len(g), rA), np.full(len(g), rB)]))
        assert abs(d['Di_A'] - 2 * (a * (0.5 - t) - rB)) < 1e-9 and abs(d['Di_A'] - Di) < 1e-9 and d['exact']


def test_dif_is_the_largest_sphere_on_the_percolating_component():
    a, m, c = 3.0, 8, 3; g = np.array([(i, j, k) for i in range(m) for j in range(m) for k in range(m)]); hole = np.all((g >= c) & (g <= c + 1), axis=1); wall = np.all((g >= c - 1) & (g <= c + 2), axis=1) & ~hole
    pos = g[~hole] * a + np.random.default_rng(0).normal(0, 1e-3, (int((~hole).sum()), 3)); d = descriptors(voronoi_network(pos, np.eye(3) * a * m, np.where(wall[~hole], 1.3, 0.5)))
    assert abs(d['Di_A'] - 2 * (a * np.sqrt(2.75) - 1.3)) < 0.01 and abs(d['Df_A'] - 2 * (a / np.sqrt(2) - 0.5)) < 0.01 and abs(d['Dif_A'] - 2 * (a * np.sqrt(3) / 2 - 0.5)) < 0.01 and d['exact']


def test_node_inside_an_atom_is_not_a_pore():
    _, pos, cell = cubic('sc', n=4); net = voronoi_network(pos, cell, np.full(len(pos), 3.0))
    assert np.allclose(net['canon_clear'], 1.5 * np.sqrt(3) - 3.0, atol=1e-9) and len(pore_network(net, 0.0)['pore_r']) == 0


def test_lattice_translations_and_basis_change_leave_the_network_unchanged():
    rng = np.random.default_rng(0); cell = np.eye(3) * 9.0; pos = rng.uniform(0, 9, (40, 3)); rad = rng.choice([0.7, 1.1], 40)
    a, b = descriptors(voronoi_network(pos, cell, rad)), descriptors(voronoi_network(pos + rng.integers(-3, 4, (40, 3)) @ cell, cell, rad))
    tri = np.array([[9.0, 0, 0], [3.0, 8.5, 0], [1.0, 2.0, 9.5]]); p2 = rng.random((60, 3)) @ tri; r2 = rng.choice([0.3, 1.2], 60); c2 = np.array([[1, 2, 0], [0, 1, 0], [0, 0, 1]]) @ tri
    e, f = descriptors(voronoi_network(p2, tri, r2)), descriptors(voronoi_network(p2 + rng.integers(-2, 3, (60, 3)) @ tri, c2, r2))
    for x, y in ((a, b), (e, f)): assert all(abs(x[k] - y[k]) < 1e-9 for k in ('Di_A', 'Df_A', 'Dif_A')) and (x['n_nodes'], x['n_edges']) == (y['n_nodes'], y['n_edges']) and x['exact'] and y['exact']


def test_union_find_carries_offsets_and_stays_shallow():
    n = 1024; T = np.random.default_rng(2).integers(-3, 4, (n, 3)); dsu = _Offsets(n)
    for i in range(n - 1, 0, -1): assert not dsu.union(i - 1, i, T[i] - T[i - 1])
    depth = 0
    for i in range(n):
        k, a = 0, i
        while dsu.p[a] != a: a = dsu.p[a]; k += 1
        depth = max(depth, k)
    root = [dsu.root(i) for i in range(n)]; assert depth <= 10 and len({r for r, _ in root}) == 1
    assert all(np.array_equal(np.asarray(root[v][1]) - np.asarray(root[u][1]), T[v] - T[u]) for u, v in np.random.default_rng(3).integers(0, n, (200, 2)))
    assert not dsu.union(5, 900, T[900] - T[5]) and dsu.union(5, 900, T[900] - T[5] + np.array([0, 1, 0]))


def _brute_clearance(rr, tree, x):
    d, i = tree.query(x, k=64); c = np.min(d - rr[i], axis=1); assert np.all(d[:, -1] - rr.max() >= c); return c


def _certified_max(pos, cell, rad, tol=1e-6):
    s = np.array(list(itertools.product((-1, 0, 1), repeat=3))) @ cell; rep = (pos[None] + s[:, None]).reshape(-1, 3); rr = np.tile(rad, 27); tree = cKDTree(rep); n = 16; E = cell / n; h = 0.5 * np.linalg.norm(E.sum(0))
    C = ((np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing='ij'), -1).reshape(-1, 3) + 0.5) / n) @ cell; f = _brute_clearance(rr, tree, C); best = f.max()
    while True:
        C = C[f + h > best + tol]
        if not len(C): return best, best + tol
        E = E / 2; h = h / 2; C = (C[:, None] + (np.array(list(itertools.product((-.5, .5), repeat=3))) @ E)[None]).reshape(-1, 3); f = _brute_clearance(rr, tree, C); best = max(best, f.max())


def _grid_percolates(M, conn):
    lab, _ = ndimage.label(M, structure=ndimage.generate_binary_structure(3, 1 if conn == 6 else 3)); n = np.array(M.shape); rows = []; p, off = {}, {}
    for d in itertools.product((-1, 0, 1), repeat=3):
        if any(d) and (conn == 26 or sum(map(abs, d)) == 1):
            for ax in np.flatnonzero(d):
                sl = [slice(None)] * 3; sl[ax] = n[ax] - 1 if d[ax] > 0 else 0; src = np.insert(np.argwhere(lab[tuple(sl)] > 0), ax, sl[ax], axis=1); tgt = src + d; b = lab[tuple((tgt % n).T)]; rows.append(np.c_[lab[tuple(src.T)], b, tgt // n][b > 0])

    def root(a):
        o = np.zeros(3, int)
        while p.get(a, a) != a: o = o + off[a]; a = p[a]
        return a, o
    for a, b, *w in np.unique(np.concatenate(rows), axis=0):
        (ra, oa), (rb, ob) = root(a), root(b)
        if ra == rb and np.any(ob - oa != w): return True
        if ra != rb: p[rb] = ra; off[rb] = oa + np.array(w) - ob
    return False


def _grid_free_radius(C, conn, tol=1e-4):
    lo, hi = float(C.min()), float(C.max())
    while hi - lo > tol: mid = (lo + hi) / 2; lo, hi = (mid, hi) if _grid_percolates(C >= mid, conn) else (lo, mid)
    return lo


def test_binary_packing_matches_independent_references():
    rng = np.random.default_rng(4); L = 8.0; cell = np.eye(3) * L; pos = rng.uniform(0, L, (60, 3)); rad = np.r_[np.full(12, 1.6), np.full(48, 0.3)]; d = descriptors(voronoi_network(pos, cell, rad))
    lo, hi = _certified_max(pos, cell, rad); C, n = clearance_field(pos, cell, rad, dx=0.08); s = L / n[0]
    assert d['exact'] and 2 * lo - 1e-9 <= d['Di_A'] <= 2 * hi and 2 * (_grid_free_radius(C, 6) - s / 2) <= d['Df_A'] <= 2 * (_grid_free_radius(C, 26) + np.sqrt(3) * s / 2)


def test_single_sphere_fraction_and_surface():
    cell = np.eye(3) * 10.0; pos = np.array([[5.0, 5.0, 5.0]]); R = 1.2; C, n = clearance_field(pos, cell, np.array([R]), dx=0.1); p = 0.8
    assert abs((1 - accessible_fraction(C, p)) - 4 / 3 * np.pi * (R + p) ** 3 / 1000) < 2e-3
    assert abs(void_surface(C, n, cell, p) / (4 * np.pi * (R + p) ** 2) - 1) < 0.02


def test_periodic_invariance():
    rng = np.random.default_rng(0); cell = np.eye(3) * 9.0; pos = rng.uniform(0, 9, (40, 3)); rad = rng.choice([0.7, 1.1], 40)
    shift = (pos + np.array([3.1, -2.2, 4.7])) % 9.0; a, b = descriptors(voronoi_network(pos, cell, rad)), descriptors(voronoi_network(shift, cell, rad))
    assert abs(a['Di_A'] - b['Di_A']) < 1e-6 and abs(a['Df_A'] - b['Df_A']) < 1e-6
    Ca, _ = clearance_field(pos, cell, rad, dx=0.3); Cb, _ = clearance_field(shift, cell, rad, dx=0.3); assert abs(Ca.max() - Cb.max()) < 0.3 and abs(accessible_fraction(Ca, 0.5) - accessible_fraction(Cb, 0.5)) < 0.02


def test_sweep_is_monotone_and_sites_partition():
    rng = np.random.default_rng(1); cell = np.eye(3) * 9.0; pos = rng.uniform(0, 9, (40, 3)); rad = np.full(40, 0.8); C, n = clearance_field(pos, cell, rad, dx=0.25); rows = probe_sweep(C, n, cell, (0.3, 0.6, 0.9))
    assert all(x['fraction'] >= y['fraction'] for x, y in zip(rows, rows[1:]))
    fv = atom_free_volume(C, n, cell, pos, rad, 0.3); assert abs(fv.sum() - rows[0]['volume_A3']) < 1e-9 and (fv >= 0).all()
    net = voronoi_network(pos, cell, rad); pores = pore_network(net, 0.3); assert len(pores['pore_r']) >= 1 and (pores['pore_r'] >= 0.3).all()
    assert np.isfinite(cavity_distance(pos, cell, pores['pore_xyz'])).all(); mask, probe = percolating_edges(net); assert mask.sum() > 0 and abs(probe - free_sphere(net)) < 1e-12


def test_radii_round_trip(tmp_path):
    r = Radii({'A': 1.1, 'B': 0.7}, 'covalent'); r.save(tmp_path / 'radii.json'); assert Radii.load(tmp_path / 'radii.json') == r and list(r.of(['B', 'A'])) == [0.7, 1.1]


def test_pointwise_and_grid_modes_agree():
    _, pos, cell = cubic('sc', n=3); a = 3.0; p = 1.0; exact = 1 - 4 / 3 * np.pi * p ** 3 / a ** 3
    C, _ = clearance_field(pos, cell, np.zeros(len(pos)), dx=0.1); s = sample_clearance(pos, cell, np.zeros(len(pos)), n=16384)
    assert abs(accessible_fraction(C, p) - exact) < 3e-3 and abs(accessible_fraction(s, p) - exact) < 5e-3 and abs(clearance_cdf(s, [p])[0] - (1 - exact)) < 5e-3


def test_radii_presets():
    r = Radii.covalent(['Si', 'O']); i = Radii.ionic(['Si', 'O']); assert r.name == 'covalent' and i.name == 'ionic' and r.table['Si'] > i.table['Si'] and i.table['O'] > r.table['O']
    assert abs(mass_density(['Si'] * 8, np.eye(3) * 5.43) - 2.33) < 0.01


def test_clearance_is_exact_with_unequal_radii():
    rng = np.random.default_rng(5); cell = np.eye(3) * 12.0; pos = rng.uniform(0, 12, (60, 3)); rad = np.where(rng.random(60) < 0.3, 2.0, 0.2); rep, r = images(pos, cell, rad); pts = rng.uniform(0, 12, (500, 3))
    brute = np.min(np.linalg.norm(pts[:, None] - rep[None], axis=2) - r[None], axis=1); assert abs(exact_clearance(cKDTree(rep), r, pts, k=2) - brute).max() < 1e-12
    C, _ = clearance_field(pos, cell, rad, dx=0.5, k=2); C2, _ = clearance_field(pos, cell, rad, dx=0.5, k=64); assert abs(C - C2).max() < 1e-12
    net = voronoi_network(pos, cell, rad); assert pore_network(net, 50.0)['pore_r'].size == 0


_CAPPED = """import json, os, sys
sys.path.insert(0, {root!r})
import numpy as np
from vfa.voids.network import voronoi_network, descriptors
try: import resource; resource.setrlimit(resource.RLIMIT_AS, ((next(int(l.split()[1]) for l in open('/proc/self/status') if l.startswith('VmSize')) << 10) + (os.cpu_count() or 1) * (8 << 20) + {extra}, resource.getrlimit(resource.RLIMIT_AS)[1]))
except (ImportError, OSError, ValueError, StopIteration): pass
pos, cell, rad = (np.asarray(x, float) for x in json.loads(sys.stdin.read()))
try: d = descriptors(voronoi_network(pos, cell, rad)); print(json.dumps([d['Di_A'], d['Df_A'], bool(d['exact'])]))
except RuntimeError as e:
    if not str(e).startswith('voronoi_network:'): raise
    print('null')
"""


def _capped_network(pos, cell, rad, extra=1 << 30, timeout=60):
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MALLOC_ARENA_MAX='2', PYTHONDONTWRITEBYTECODE='1')
    run = subprocess.run([sys.executable, '-c', _CAPPED.format(extra=extra, root=str(ROOT))], input=json.dumps([pos.tolist(), cell.tolist(), rad.tolist()]), capture_output=True, text=True, timeout=timeout, env=env)
    assert run.returncode == 0, run.stderr[-400:]; return json.loads(run.stdout)


def test_near_degenerate_unequal_radii_crystals_return_or_raise_within_memory():
    a = 5.0; g = np.array(list(itertools.product(range(4), repeat=3)), float); s = np.array(list(itertools.product(range(2), repeat=3)), float) * a
    nacl = (g * a / 2, np.eye(3) * 2 * a, np.where(g.sum(1) % 2, 0.4, 1.0)); cscl = (np.r_[s, s + a / 2], np.eye(3) * 2 * a, np.r_[np.full(8, 1.5), np.full(8, 0.3)])
    for pos, cell, rad in (nacl, cscl):
        ref = descriptors(voronoi_network(pos, cell, rad)); out = _capped_network(pos + np.random.default_rng(1).normal(0, 1e-8, pos.shape), cell, rad)
        assert ref['exact'] and (out is None or (abs(out[0] - ref['Di_A']) < 1e-6 and abs(out[1] - ref['Df_A']) < 1e-6))
    with pytest.raises(RuntimeError, match='reach'): voronoi_network(*cscl, max_images=100)


def test_arc_tracing_memory_stays_bounded_across_a_wide_vacuum():
    a, n, h, r = 3.0, 6, 40.0, 0.8; g = np.array(list(itertools.product(range(n), range(n), range(1))), float) * a; pos = g + np.random.default_rng(0).normal(0, 1e-3, g.shape)
    out = _capped_network(pos, np.diag([n * a, n * a, 2 * h]), np.full(len(pos), r), extra=600 << 20)
    assert out is not None and abs(out[0] - 2 * (np.hypot(a / np.sqrt(2), h) - r)) < 1e-2 and abs(out[1] - 2 * (np.hypot(a / 2, h) - r)) < 1e-2 and out[2]


def test_a_void_that_does_not_percolate_has_no_free_sphere():
    pos = np.random.default_rng(5).uniform(0, 9, (60, 3)); cell = np.eye(3) * 9.0; net = voronoi_network(pos, cell, np.full(60, 2.4)); d = descriptors(net); mask, probe = percolating_edges(net)
    assert d['percolates'] is False and np.isnan([d['Df_A'], d['Dif_A'], d['cavity_to_neck'], free_sphere(net), probe]).all() and d['Di_A'] > 0 and d['exact'] and not mask.any()
    assert len(pore_network(net, 0.0)['pore_r']) > 0 and not pore_network(net, 0.0)['pore_percolating'].any() and not percolating_edges(net, 0.0)[0].any()
    for f in (percolating_edges, pore_network):
        with pytest.raises(ValueError, match='probe'): f(net, -0.1)
    e = descriptors(voronoi_network(pos, cell, np.full(60, 1.8))); assert e['percolates'] is True and 0 < e['Df_A'] <= e['Dif_A'] <= e['Di_A']


def test_percolation_is_undecided_when_arcs_are_left_out_and_no_loop_closes():
    a = 3.0; pos = np.array(list(itertools.product(range(2), repeat=3)), float) * a; cell = np.eye(3) * 2 * a; r = np.full(8, 0.6); Df = 2 * (a / np.sqrt(2) - 0.6)
    for seed, eps in ((1, 1e-9), (18, 1e-9), (22, 1e-8)):
        p = pos + np.random.default_rng(seed).normal(0, eps, pos.shape); d = descriptors(voronoi_network(p, cell, r, verify=False))
        assert not d['exact'] and d['complete'] is None and d['percolates'] is None and np.isnan([d['Df_A'], d['Dif_A'], d['cavity_to_neck']]).all() and abs(d['Di_A'] - 2 * (a * np.sqrt(3) / 2 - 0.6)) < 1e-6
        v = descriptors(voronoi_network(p, cell, r)); assert not v['exact'] and v['complete'] and v['n_nodes_added'] > 0 and v['n_nodes'] == v['n_nodes_enumerated'] and v['percolates'] is True and abs(v['Df_A'] - Df) < 1e-6
    d = descriptors(voronoi_network(pos + np.random.default_rng(0).normal(0, 1e-9, pos.shape), cell, r)); assert not d['exact'] and d['percolates'] is True and abs(d['Df_A'] - Df) < 1e-6


def _arrays(net):
    return [*net['edges'], net['canon_clear'], net['canon_xyz'], net['neck'], net['n_nodes'], net['n_uncertified_nodes'], net['n_uncertified_edges'], net['complete'], net['n_nodes_enumerated'], net['n_nodes_added'], net['n_isolated_edges']]


def test_workers_and_pair_blocks_leave_the_network_unchanged():
    rng = np.random.default_rng(0); cell = np.eye(3) * 9.0; pos = rng.uniform(0, 9, (40, 3)); rad = rng.choice([0.7, 1.1], 40); ref = voronoi_network(pos, cell, rad)
    for kw in (dict(workers=1), dict(workers=-1, max_pairs=600)): assert all(np.array_equal(x, y) for x, y in zip(_arrays(ref), _arrays(voronoi_network(pos, cell, rad, **kw))))
    with pytest.raises(RuntimeError, match='voronoi_network: [0-9]+ atom images .* exceed max_pairs'): voronoi_network(pos, cell, rad, max_pairs=20)


def test_the_image_reach_stays_within_three_times_the_largest_clearance_and_radius(monkeypatch):
    seen = []; init = _Space.__init__

    def spy(self, *args): init(self, *args); seen.append(self.reach)
    monkeypatch.setattr(_Space, '__init__', spy)
    for h in (22.0, 25.0):
        g = np.array(list(itertools.product(range(2), range(2), range(1))), float) * 3.0; pos = g + np.random.default_rng(0).normal(0, 1e-3, g.shape); cell = np.diag([6.0, 6.0, 2 * h]); r = np.full(4, 0.8); seen.clear()
        assert descriptors(voronoi_network(pos, cell, r))['exact'] and len(seen) >= 2 and max(seen) <= max(3 * (36 * 2 * h / 4) ** (1 / 3) + 1.6, 3 * (_largest_clearance(pos, cell, r) + 0.8) + 1.0)


def test_invalid_input_raises_naming_the_network():
    _, pos, cell = cubic('sc', n=2); rad = np.zeros(len(pos)); p = pos.copy(); p[3, 0] = np.nan; c = cell.copy(); c[0, 1] = np.inf; flat = cell.copy(); flat[2] = flat[0] + flat[1]
    for args, name in (((p, cell, rad), 'positions'), ((pos, c, rad), 'cell'), ((pos, cell, np.r_[np.inf, rad[1:]]), 'radii'), ((pos[:, :2], cell, rad), 'positions'), ((pos, flat, rad), 'cell'), ((pos, cell, np.r_[-0.1, rad[1:]]), 'radii')):
        with pytest.raises(ValueError, match=f'voronoi_network: .*{name}'): voronoi_network(*args)
    with pytest.raises(ValueError, match='voronoi_network: max_pairs'): voronoi_network(pos, cell, rad, max_pairs=0)


def test_merge_fraction_sets_the_pore_partition_and_probes_must_be_finite():
    rng = np.random.default_rng(1); cell = np.eye(3) * 9.0; pos = rng.uniform(0, 9, (40, 3)); rad = np.full(40, 0.8); net = voronoi_network(pos, cell, rad)
    few, many = pore_network(net, 0.3, merge_frac=0.0), pore_network(net, 0.3, merge_frac=1.0); assert len(few['pore_r']) < len(many['pore_r']) and few['decided'] and many['decided']
    for call in (lambda: pore_network(net, np.nan), lambda: percolating_edges(net, np.nan), lambda: pore_network(net, -0.1)):
        with pytest.raises(ValueError, match='finite and non-negative'): call()


def test_pore_percolation_is_undecided_on_an_inexact_network_with_no_loop():
    rng = np.random.default_rng(1); cell = np.eye(3) * 9.0; net = voronoi_network(rng.uniform(0, 9, (40, 3)), cell, np.full(40, 0.8)); top = float(net['canon_clear'].max()); rf = free_sphere(net)
    assert pore_network(net, 0.5 * (rf + top))['decided'] and not pore_network(net, 0.5 * (rf + top))['pore_percolating'].any()
    for key, value in (('n_uncertified_edges', 1), ('n_uncertified_nodes', 1), ('complete', False), ('complete', None)):
        assert not pore_network(dict(net, **{key: value}), 0.5 * (rf + top))['decided'] and pore_network(dict(net, **{key: value}), 0.5 * rf)['decided']
        assert not pore_network(dict(net, **{key: value}), 2 * top)['decided'] and pore_network(net, 2 * top)['decided'] and not descriptors(dict(net, **{key: value}))['exact']


def test_an_edge_without_a_vertex_is_counted_and_the_network_stays_complete():
    cell = np.eye(3) * 20.0; big = [[6.5, 10, 10], [13.5, 10, 10]]
    for small in ([10, 10, 10], [10, 10.15, 10]):
        d = descriptors(voronoi_network(np.array(big + [small], float), cell, np.array([3.0, 3.0, 0.3])))
        assert d['n_isolated_edges'] == 1 and d['complete'] and d['exact'] and d['n_nodes'] == d['n_nodes_enumerated'] == 2
    d = descriptors(voronoi_network(np.array(big + [[10, 10.6, 10], [10, 9.4, 10.1]], float), cell, np.array([3.0, 3.0, 0.3, 0.32])))
    assert d['n_isolated_edges'] == 0 and d['complete'] and d['exact'] and d['n_nodes'] == d['n_nodes_enumerated'] == 4
    pos = np.array([[12.944, 11.566, 11.764], [13.132, 11.355, 11.623], [11.424, 14.578, 14.112]]); d = descriptors(voronoi_network(pos, np.eye(3) * 25.0, np.array([1.8, 1.8, 0.3])))
    assert d['n_isolated_edges'] == 0 and d['complete'] and d['exact']


def test_atoms_on_one_sphere_are_verified_once_per_node_and_the_work_is_capped():
    m = 24; i = np.arange(m) + 0.5; phi = np.arccos(1 - 2 * i / m); th = np.pi * (1 + 5 ** 0.5) * i; pos = 8.0 + 5.0 * np.c_[np.sin(phi) * np.cos(th), np.sin(phi) * np.sin(th), np.cos(phi)]; cell = np.eye(3) * 16.0
    d = descriptors(voronoi_network(pos, cell, np.ones(m))); assert d['complete'] and d['exact'] and d['n_nodes'] == d['n_nodes_enumerated'] and d['n_nodes_added'] == 0
    with pytest.raises(RuntimeError, match='voronoi_network: verifying the network needs [0-9]+ sets of four or three atoms, more than max_sets'): voronoi_network(pos, cell, np.ones(m), max_sets=1000)
    with pytest.raises(ValueError, match='voronoi_network: max_sets'): voronoi_network(pos, cell, np.ones(m), max_sets=0)


def test_vertices_the_trace_cannot_reach_are_added_from_the_enumeration(monkeypatch):
    cell = np.eye(3) * 20.0; pos = np.array([[6.5, 10, 10], [13.5, 10, 10], [10, 10.6, 10], [10, 9.4, 10.1]]); rad = np.array([3.0, 3.0, 0.3, 0.32]); ref = voronoi_network(pos, cell, rad); seeds = network._seeds
    monkeypatch.setattr(network, '_seeds', lambda sp, tol: (lambda s: s[np.linalg.norm(s - [10, 10, 10], axis=1) > 2])(seeds(sp, tol)))
    net = voronoi_network(pos, cell, rad); d, e = descriptors(net), descriptors(ref)
    assert d['n_nodes_added'] == 2 and e['n_nodes_added'] == 0 and d['complete'] and d['exact'] and all(d[k] == e[k] for k in ('n_nodes', 'n_edges', 'Di_A', 'Df_A', 'percolates'))
    assert descriptors(voronoi_network(pos, cell, rad, verify=False))['n_nodes'] == 2 and not descriptors(voronoi_network(pos, cell, rad, verify=False))['exact']


def test_a_vertex_is_traced_once_where_its_spheres_are_ill_conditioned():
    sym, pos, cell = read_extxyz(ROOT / 'example/data/structures/rho0.068_4.extxyz'); d = descriptors(voronoi_network(pos, cell, Radii.load(ROOT / 'example/data/radii.json').of(sym)))
    assert d['exact'] and d['complete'] and d['n_nodes'] == d['n_nodes_enumerated'] == 1520 and d['n_nodes_added'] == 0 and d['n_uncertified_nodes'] == 0


def test_verify_must_be_a_boolean():
    _, pos, cell = cubic('sc', n=2)
    with pytest.raises(ValueError, match='voronoi_network: verify'): voronoi_network(pos, cell, np.zeros(len(pos)), verify=1)


def test_a_flat_sheet_in_wide_vacuum_grows_the_reach_and_matches_the_closed_form():
    a, L, r = 2.46, 40.0, 0.7; cg = np.array([[a, 0, 0], [a / 2, a * 3 ** 0.5 / 2, 0], [0, 0, L]]); B = np.array([[0, 0, L / 2], [a / 2, a / (2 * 3 ** 0.5), L / 2]])
    pts = np.array([b + i * cg[0] + j * cg[1] for i in range(2) for j in range(2) for b in B]); C = cg * np.array([[2], [2], [1]]); d = descriptors(voronoi_network(pts, C, np.full(8, r), verify=False))
    assert abs(d['Di_A'] - 2 * (np.hypot(L / 2, a / 3 ** 0.5) - r)) < 1e-9


def test_network_options_are_checked():
    pos, cell = np.zeros((1, 3)), np.eye(3) * 4.0
    for kw in (dict(k=3), dict(k=2.5), dict(tol_A=0.0), dict(tol_A=np.nan)):
        with pytest.raises(ValueError, match='voronoi_network: k must be'): voronoi_network(pos, cell, np.ones(1), **kw)
    net = voronoi_network(pos, cell, np.ones(1))
    for m in (-0.1, 1.5, np.nan):
        with pytest.raises(ValueError, match='merge_frac'): pore_network(net, 0.1, merge_frac=m)


def _arc_minima(pos, cell, rad, net, step=4e-3, max_steps=5000):
    s = np.array(list(itertools.product(range(-2, 3), repeat=3))) @ cell; P = (pos[None] + s[:, None]).reshape(-1, 3); r = np.tile(rad, len(s)); dist = lambda x: np.linalg.norm(x[:, None] - P[None], axis=2) - r
    cu, cv, d = net['edges']; A, B = net['canon_xyz'][cu], net['canon_xyz'][cv] + d @ cell; touch = lambda x: [set(np.flatnonzero(g - g.min() < 1e-7)) for g in dist(x)]
    e, T, k = (np.array(v) for v in zip(*[(q, sorted(t), m) for q, (ta, tb) in enumerate(zip(touch(A), touch(B))) for t in itertools.combinations(sorted(ta & tb), 3) for m in sorted(ta - set(t))]))
    p, rr, b, x = P[T], r[T], B[e], A[e].copy(); live = np.ones(len(e), bool); hist = [x.copy()]

    def grads(y, i): D = y[:, None] - p[i]; n = np.linalg.norm(D, axis=2); return D / n[:, :, None], n - rr[i]

    def tang(y, i): g = grads(y, i)[0]; t = np.cross(g[:, 0] - g[:, 1], g[:, 0] - g[:, 2]); return t / np.linalg.norm(t, axis=1)[:, None]
    i = np.arange(len(e)); t = tang(x, i); t *= np.sign((((x - P[k]) / np.linalg.norm(x - P[k], axis=1)[:, None] - grads(x, i)[0][:, 0]) * t).sum(1))[:, None]
    for _ in range(max_steps):
        i = np.flatnonzero(live); fin = np.linalg.norm(x[i] - b[i], axis=1) <= step; j = i[~fin]; y = x[j] + step * t[j]
        for _ in range(3): g, dd = grads(y, j); J = np.stack([g[:, 0] - g[:, 1], g[:, 0] - g[:, 2]], 1); y -= np.einsum('nij,ni->nj', J, np.linalg.solve(J @ J.transpose(0, 2, 1), (dd[:, :1] - dd[:, 1:])[:, :, None])[:, :, 0])
        tn = tang(y, j); t[j] = tn * np.sign((tn * t[j]).sum(1))[:, None]; x[j] = y; x[i[fin]] = b[i[fin]]; live[i[fin]] = False; h = np.full(x.shape, np.nan); h[i] = x[i]; hist.append(h)
        if not live.any(): break
    H = np.array(hist); best = np.full(len(e), np.nan)
    for q in np.flatnonzero(~live):
        y = H[:, q][np.isfinite(H[:, q, 0])]; R = np.linalg.norm(y - p[q, 0], axis=1) - rr[q, 0]; m = int(np.argmin(R))
        if (R[::10] - dist(y[::10]).min(1)).max() > 1e-9 or R[m] - dist(y[m:m + 1]).min() > 1e-9: continue
        best[q] = R[m]; h0, h2 = (-np.linalg.norm(y[m - 1] - y[m]), np.linalg.norm(y[m + 1] - y[m])) if 0 < m < len(R) - 1 else (0.0, 0.0)
        if min(-h0, h2) > 1e-6: a = ((R[m + 1] - R[m]) / h2 - (R[m - 1] - R[m]) / h0) / (h2 - h0); c = (R[m + 1] - R[m]) / h2 - a * h2; best[q] = min(R[m], R[m] - c * c / (4 * a)) if a > 0 else R[m]
    return e, best


def test_edge_necks_are_the_least_clearance_along_their_arcs():
    for seed in (0, 3):
        rng = np.random.default_rng(seed); cell = np.diag([6.0, 6.6, 7.2]) + np.triu(rng.uniform(-1.5, 1.5, (3, 3)), 1); pos = rng.random((12, 3)) @ cell; rad = rng.choice([0.3, 0.9, 1.5], 12)
        net = voronoi_network(pos, cell, rad); e, best = _arc_minima(pos, cell, rad, net); assert net['n_uncertified_edges'] == 0 and net['complete']
        assert all(np.nanmin(np.abs(best[e == q] - net['neck'][q])) < 1e-7 for q in range(len(net['neck'])))


def test_half_diagonal_is_half_the_longest_diagonal_of_any_cell():
    V = np.array([[3.0, 0, 0], [1.2, 2.5, 0], [0.8, 0.9, 2.2]])
    for s in itertools.product((1, -1), repeat=3):
        E = np.diag(s) @ V; assert abs(network._half_diagonal(E) - 0.5 * max(np.linalg.norm(np.array(t) @ E) for t in itertools.product((1, -1), repeat=3))) < 1e-12


def test_one_atom_cubic_cell_has_one_node_and_three_edges_to_its_images():
    a, r = 4.0, 0.5; net = voronoi_network(np.zeros((1, 3)), np.eye(3) * a, np.array([r])); cu, cv, d = net['edges']; dd = descriptors(net)
    assert net['n_nodes'] == 1 and len(net['neck']) == 3 and (cu == 0).all() and (cv == 0).all() and sorted(map(tuple, np.abs(d).tolist())) == [(0, 0, 1), (0, 1, 0), (1, 0, 0)] and dd['exact']
    assert abs(net['canon_clear'][0] - (a * np.sqrt(3) / 2 - r)) < 1e-9 and abs(net['neck'] - (a / np.sqrt(2) - r)).max() < 1e-9 and abs(dd['cavity_to_neck'] - (a * np.sqrt(3) / 2 - r) / (a / np.sqrt(2) - r)) < 1e-9
    zero = descriptors(dict(net, neck=np.zeros(3))); assert zero['percolates'] is True and zero['Df_A'] == 0 and not np.isfinite(zero['cavity_to_neck'])


def test_pores_and_throats_of_a_hand_built_network():
    r = np.array([1.0, 0.95, 0.5, 1.2, 0.3, 1.0, 1.0]); xyz = np.arange(21.0).reshape(7, 3)
    e = np.array([[0, 1], [1, 2], [2, 3], [0, 3], [3, 1], [3, 4], [5, 6], [5, 0], [6, 6], [3, 5]]); neck = np.array([0.94, 0.45, 0.48, 0.6, 0.7, 0.25, 0.92, 0.45, 0.8, 0.5]); d = np.zeros((10, 3), np.int64); d[8, 0] = 1
    net = dict(n_nodes=7, canon_clear=r, canon_xyz=xyz, edges=(e[:, 0], e[:, 1], d), neck=neck, n_uncertified_nodes=0, n_uncertified_edges=0, complete=True); p = pore_network(net, 0.5)
    assert np.array_equal(p['pore_r'], [1.0, 0.5, 1.2, 1.0]) and np.array_equal(p['pore_xyz'], xyz[[0, 2, 3, 5]]) and p['throat_pairs'].tolist() == [[0, 2], [2, 3]] and p['throat_r'].tolist() == [0.7, 0.5]
    assert p['pore_percolating'].tolist() == [True, False, True, True] and p['decided'] is True
    none = pore_network(net, 1.5); assert none['pore_xyz'].shape == (0, 3) and none['pore_r'].shape == none['pore_percolating'].shape == none['throat_r'].shape == (0,) and none['throat_pairs'].shape == (0, 2)


def test_complete_needs_a_one_to_one_match_and_equal_counts(monkeypatch):
    pos = np.array(list(itertools.product(range(2), repeat=3)), float) * 3.0; cell = np.eye(3) * 6.0; rad = np.zeros(8); assert voronoi_network(pos, cell, rad)['complete'] is True; enumerated = network._enumerated
    for change in (lambda V: np.r_[V[:-1], V[:1]], lambda V: np.r_[V, V[:1]]):
        monkeypatch.setattr(network, '_enumerated', lambda sp, tol, tolV, max_sets, change=change: (lambda V, iso: (change(V), iso))(*enumerated(sp, tol, tolV, max_sets)))
        net = voronoi_network(pos, cell, rad); assert net['complete'] is False and net['n_nodes_added'] == 0 and net['n_nodes'] == 8 and not descriptors(net)['exact']


def test_cavity_to_neck_is_the_ratio_of_the_diameters_below_one_angstrom():
    a, r = 3.0, 1.2; d = descriptors(voronoi_network(np.zeros((1, 3)), np.eye(3) * a, np.full(1, r)))
    assert abs(d['Df_A'] - 2 * (a / 2 ** 0.5 - r)) < 1e-7 and d['Df_A'] < 2 and abs(d['cavity_to_neck'] - d['Di_A'] / d['Df_A']) < 1e-9
