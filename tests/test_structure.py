"""Bond graph, rings, pair correlations and the shortest lattice vector against known answers; the first minimum; input checks."""
import itertools, sys, tracemalloc
from pathlib import Path
import numpy as np, pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lattices import cubic, graphene, sizes
from vfa.io.extxyz import read_structure_files
from vfa.structure.cell import SHIFTS, minimum_image, shortest_vector, widths
from vfa.structure.graph import bond_graph, coordination, bond_vectors, angles, tetrahedral_order, planarity
from vfa.structure.rings import primitive_rings, rings_converged, ring_sizes, canonical, rings, king_rings, guttman_rings, ring_extents
from vfa.structure.rdf import partial_rdf, ensemble_rdf, running_coordination, first_minimum

ROOT = Path(__file__).resolve().parents[1]


def test_diamond_graph_and_angles():
    sym, pos, cell = cubic('diamond', 5.0, 2); e, o, l = bond_graph(sym, pos, cell, {('A', 'A'): 5.0 * np.sqrt(3) / 4 * 1.01}); z = coordination(len(sym), e)
    assert set(z) == {4} and len(e) == 2 * len(sym) and abs(l - 5.0 * np.sqrt(3) / 4).max() < 1e-9
    v = bond_vectors(pos, cell, e, o); assert abs(angles(v, range(len(sym))) - 109.4712).max() < 1e-3 and abs(tetrahedral_order(v, range(len(sym))) - 1).max() < 1e-9


def test_diamond_rings():
    sym, pos, cell = cubic('diamond', 5.0, 2); e, o, _ = bond_graph(sym, pos, cell, {('A', 'A'): 2.2}); rings, _ = rings_converged(len(sym), e, o, bipartite=False, start=8)
    assert len(rings) == 2 * len(sym) and set(ring_sizes(rings, bipartite=False)) == {6}
    sym2, _, _ = cubic('diamond', 5.0, 2, two_species=True); e2, o2, _ = bond_graph(sym2, pos, cell, {('A', 'B'): 2.2}); r2 = primitive_rings(len(sym2), e2, o2, 12)
    assert len(r2) == len(rings) and set(ring_sizes(r2)) == {3} and canonical((2, 3, 1)) == canonical((1, 2, 3)) == canonical((3, 2, 1))


def test_graphene_planarity():
    sym, pos, cell = graphene(3, 2); e, o, _ = bond_graph(sym, pos, cell, {('C', 'C'): 2.46 / np.sqrt(3) * 1.05}); v = bond_vectors(pos, cell, e, o)
    assert set(coordination(len(sym), e)) == {3} and abs(planarity(v, range(len(sym))) - 360).max() < 1e-6


def test_rdf_counts_neighbours():
    sym, pos, cell = cubic('diamond', 5.0, 2); r, cn = running_coordination(sym, pos, cell, ('A', 'A'), r_max=4.0, n_bins=400); assert abs(cn[np.searchsorted(r, 2.5)] - 4) < 1e-9
    r, g = partial_rdf(sym, pos, cell, ('A', 'A'), r_max=4.0, n_bins=400); fm = first_minimum(r, g, search=(1.5, 3.2)); assert abs(fm['peak_A'] - 5.0 * np.sqrt(3) / 4) < 0.05 and fm['minimum_A'] > fm['peak_A'] and not fm['resolved']
    fm = first_minimum(r, g, search=(1.5, 4.0)); first, second = 5.0 * np.sqrt(3) / 4, 5.0 / np.sqrt(2); assert abs(fm['peak_A'] - first) < 0.01 and abs(fm['basin_middle_A'] - 0.5 * (first + second)) < 0.01 and fm['minimum_g'] == 0 and fm['resolved']


def test_ring_criteria_differ_on_a_cubic_lattice():
    a = 3.0; sym, pos, cell = cubic('sc', a, 3)
    e, o, _ = bond_graph(sym, pos, cell, {('A', 'A'): a * 1.01})
    prim = rings(len(sym), e, o, 8, 'primitive', bipartite=False); king = king_rings(len(sym), e, o, 8, bipartite=False); gut = guttman_rings(len(sym), e, o, 8, bipartite=False)
    assert sizes(prim) == {4: 81, 6: 108} and sizes(king) == {4: 81, 6: 162} and sizes(gut) == {4: 81}
    assert not {canonical(x) for x in prim if len(x) == 6} & {canonical(x) for x in king if len(x) == 6}


def test_ring_criteria_agree_on_diamond():
    sym, pos, cell = cubic('diamond', 5.0, 2); e, o, _ = bond_graph(sym, pos, cell, {('A', 'A'): 2.2})
    counts = {c: len(rings(len(sym), e, o, 10, c, bipartite=False)) for c in ('primitive', 'king', 'guttman')}; assert set(counts.values()) == {2 * len(sym)}


def test_bond_graph_matches_brute_force():
    rng = np.random.default_rng(4); cell = np.array([[9, 0, 0], [2, 9, 0], [1, 1, 9.0]]); pos = rng.random((50, 3)) @ cell; sym = np.array(['A', 'B'] * 25); cut = {('A', 'B'): 2.4, ('B', 'B'): 1.9}
    e, o, l = bond_graph(sym, pos, cell, cut); found = {(int(i), int(j), tuple(int(x) for x in s)): x for (i, j), s, x in zip(e, o, l)}; brute = {}
    for s in SHIFTS.astype(int):
        d = np.linalg.norm((pos + s @ cell)[None] - pos[:, None], axis=2)
        for i in range(50):
            for j in range(50):
                c = cut.get(tuple(sorted((sym[i], sym[j]))))
                if c is None or d[i, j] > c or (not s.any() and i >= j): continue
                if (j, i, tuple(-s)) not in brute: brute[(i, j, tuple(s))] = d[i, j]
    assert len(found) == len(e) == len(brute) and all(abs(x - brute.get(k, brute.get((k[1], k[0], tuple(-y for y in k[2])), np.inf))) < 1e-12 for k, x in found.items())
    assert np.array_equal(np.lexsort((o[:, 2], o[:, 1], o[:, 0], e[:, 1], e[:, 0])), np.arange(len(e)))


def test_ring_extents_on_diamond():
    sym, pos, cell = cubic('diamond', 5.0, 2); e, o, _ = bond_graph(sym, pos, cell, {('A', 'A'): 2.2}); r = rings(len(sym), e, o, 10, 'primitive', bipartite=False)
    ex = ring_extents(r, pos, cell); assert abs(ex - ex[0]).max() < 1e-9 and 4.0 < ex[0] < 5.0


def test_minimum_image_is_exact_in_skewed_cells():
    rng = np.random.default_rng(2)
    for cell in (np.array([[10, 0, 0], [-5, 5 * np.sqrt(3), 0], [0, 0, 7.0]]), np.array([[10, 0, 0], [20, 10, 0], [0, 0, 10.0]]), np.array([[0, 5, 5], [5, 0, 5], [5, 5, 0.0]])):
        v = rng.uniform(-25, 25, (2000, 3)); m = minimum_image(v, cell); best = np.full(len(v), np.inf)
        for s in itertools.product(range(-8, 9), repeat=3): best = np.minimum(best, np.linalg.norm(v + np.array(s) @ cell, axis=1))
        lat = (m - v) @ np.linalg.inv(cell); assert abs(np.linalg.norm(m, axis=1) - best).max() < 1e-9 and abs(lat - np.rint(lat)).max() < 1e-9


def test_graph_and_rdf_do_not_depend_on_lattice_translations_of_atoms():
    rng = np.random.default_rng(3); cell = np.array([[9, 0, 0], [2, 9, 0], [1, 1, 9.0]]); pos = rng.random((60, 3)) @ cell; sym = np.array(['A', 'B'] * 30); cut = {('A', 'B'): 2.4}
    moved = pos + rng.integers(-3, 4, (60, 3)) @ cell; e0, o0, l0 = bond_graph(sym, pos, cell, cut); e1, o1, l1 = bond_graph(sym, moved, cell, cut)
    assert np.array_equal(e0, e1) and abs(l0 - l1).max() < 1e-9
    v0, v1 = bond_vectors(pos, cell, e0, o0), bond_vectors(moved, cell, e1, o1); assert max(abs(np.sort(a, 0) - np.sort(b, 0)).max() for a, b in zip(v0, v1) if len(a)) < 1e-9
    assert abs(partial_rdf(sym, pos, cell, ('A', 'B'), n_bins=200)[1] - partial_rdf(sym, moved, cell, ('A', 'B'), n_bins=200)[1]).max() == 0


def test_rdf_stops_at_half_the_shortest_lattice_vector():
    sym, pos, cell = cubic('diamond', 5.0, 2); half = 0.5 * np.linalg.norm(shortest_vector(cell)); r, _ = partial_rdf(sym, pos, cell, ('A', 'A'), n_bins=100); assert abs(r[-1] + 0.5 * (r[1] - r[0]) - half) < 1e-12
    with pytest.raises(ValueError, match='exceeds half the shortest lattice vector'): partial_rdf(sym, pos, cell, ('A', 'A'), r_max=half + 0.1)


def test_unresolved_first_minimum_keeps_every_key():
    r = np.linspace(0.05, 5, 100); fm = first_minimum(r, r.copy(), search=(1.0, 4.5), smoothing_A=0); full = first_minimum(*partial_rdf(*cubic('diamond', 5.0, 2), ('A', 'A'), r_max=4.0, n_bins=400), search=(1.5, 3.2))
    assert set(fm) == set(full) and not fm['resolved'] and np.isnan(fm['minimum_A'])


def test_shortest_vector_is_exact_in_any_basis():
    rng = np.random.default_rng(8); hexagonal = np.array([[6.0, 0, 0], [-3.0, 3 * np.sqrt(3), 0], [0, 0, 9.0]]); cells = [rng.normal(0, 4, (3, 3)) + np.eye(3) * 6 for _ in range(20)]
    for cell in cells + [hexagonal, np.array([[1, 0, 0], [7, 1, 0], [-5, 3, 1]]) @ hexagonal, np.array([[1, 0, 0], [20, 1, 0], [0, 0, 1]]) @ np.eye(3) * 9.0]:
        n = np.array(list(itertools.product(range(-25, 26), repeat=3))); n = n[np.any(n != 0, axis=1)]; best = np.linalg.norm(n @ cell, axis=1).min(); v = shortest_vector(cell); f = v @ np.linalg.inv(cell)
        assert abs(np.linalg.norm(v) - best) < 1e-9 * best and abs(f - np.rint(f)).max() < 1e-9 and np.any(np.rint(f) != 0)
    assert abs(np.linalg.norm(shortest_vector(hexagonal)) - 6.0) < 1e-12 and 0.5 * widths(hexagonal).min() < 0.45 * 6.0


def test_rdf_range_and_values_do_not_depend_on_the_basis():
    rng = np.random.default_rng(9); cell = np.eye(3) * 9.0; pos = rng.random((60, 3)) @ cell; sym = np.array(['A', 'B', 'B'] * 20)
    for U in (np.array([[1, 0, 0], [2, 1, 0], [0, 0, 1]]), np.array([[1, 0, 0], [1, 1, 0], [0, -2, 1]])):
        r0, g0 = partial_rdf(sym, pos, cell, ('A', 'B'), n_bins=300); r1, g1 = partial_rdf(sym, pos, U @ cell, ('A', 'B'), n_bins=300)
        assert np.array_equal(r0, r1) and abs(r0[-1] + 0.5 * (r0[1] - r0[0]) - 4.5) < 1e-12 and abs(g0 - g1).max() < 1e-12
    assert abs(partial_rdf(sym, pos, np.array([[1, 0, 0], [2, 1, 0], [0, 0, 1]]) @ cell, ('B', 'B'), r_max=4.4, n_bins=300)[1] - partial_rdf(sym, pos, cell, ('B', 'B'), r_max=4.4, n_bins=300)[1]).max() < 1e-12


def bumps(r, *shells):
    """Sum of Gaussian shells (centre, width, height) on the grid r."""
    return sum(h * np.exp(-0.5 * ((r - c) / w) ** 2) for c, w, h in shells)


def test_first_minimum_takes_the_first_prominent_peak():
    r = np.linspace(0.005, 6, 1200); g = bumps(r, (2.35, 0.06, 0.6), (3.05, 0.12, 3.0)) + np.where(r > 3.5, 1 - np.exp(-(r - 3.5) / 0.3), 0); fm = first_minimum(r, g); cut = first_minimum(r, g, search=(1.0, 2.9))
    assert abs(fm['peak_A'] - 2.35) < 0.01 and abs(fm['basin_middle_A'] - 2.59) < 0.01 and fm['resolved'] and cut['basin_middle_A'] == fm['basin_middle_A'] and not cut['resolved']
    r = (np.arange(600) + 0.5) * 0.01; split = first_minimum(r, bumps(r, (2.0, 0.04, 3.0), (2.16, 0.04, 6.0), (3.2, 0.15, 1.5))); assert abs(split['peak_A'] - 2.16) < 0.01 and 2.3 < split['basin_middle_A'] < 2.9 and split['resolved']


def test_first_minimum_basin_ends_at_the_next_peak():
    r = (np.arange(800) + 0.5) * 0.0075; g = np.clip(0.6 + bumps(r, (1.6, 0.08, 4.0), (2.45, 0.10, 0.8), (4.4, 0.15, 0.5)) - 0.6 * (r < 1.25), 0, None); fm = first_minimum(r, g)
    assert fm['basin_A'][1] < 2.45 and abs(fm['basin_middle_A'] - 2.0285) < 0.01 and abs(fm['minimum_g'] - 0.6) < 1e-3 and fm['resolved']
    r = (np.arange(600) + 0.5) * 0.01; k = np.arange(600); g = np.where(k <= 167, np.maximum(0, 30 * (1 - ((r - 1.6) / 0.08) ** 2)), 0.0); g[185:190] = 0.5; g[235:] = 1.0 + np.exp(-(r[235:] - 2.6) ** 2 / 0.02)
    fm = first_minimum(r, g); assert abs(fm['basin_A'][0] - 1.735) < 1e-9 and abs(fm['basin_A'][1] - 2.295) < 1e-9 and fm['minimum_g'] == 0.0 and fm['resolved']


def test_first_minimum_does_not_sink_the_last_bins():
    out = []
    for r_max in (6.0, 4.2, 4.0):
        r = (np.arange(int(round(r_max / 0.01))) + 0.5) * 0.01; g = np.where(r < 2.2, 0, bumps(r, (2.7, 0.22, 2.6)) + (1 - bumps(r, (3.6, 0.35, 0.5))) / (1 + np.exp(-(r - 2.6) / 0.08))); out.append(first_minimum(r, g))
    assert all(o == out[0] for o in out) and abs(out[0]['basin_middle_A'] - 3.61) < 0.01 and not out[0]['resolved']


def test_first_minimum_is_stable_against_the_binning():
    _, cells = read_structure_files(ROOT / 'example/data/structures')
    for pair, middle in ((('A', 'B'), 2.043), (('B', 'B'), 3.087)):
        curves = [ensemble_rdf(cells, pair, r_max=rm, n_bins=nb) for rm, nb in ((None, 400), (None, 1000), (6.0, 600), (6.0, 2000))]; fms = [first_minimum(r, g, search=(1.0, 5.5)) for r, g in curves]; mids = [f['basin_middle_A'] for f in fms]
        assert max(mids) - min(mids) < 0.02 and max(abs(m - middle) for m in mids) < 0.015 and all(f['resolved'] for f in fms)
        if pair == ('A', 'B'): assert abs(first_minimum(*curves[2])['basin_middle_A'] - 2.045) < 0.005


def test_running_coordination_counts_within_the_returned_r():
    rng = np.random.default_rng(12); cell = np.array([[9, 0, 0], [1, 9, 0], [0, 2, 9.0]]); pos = rng.random((80, 3)) @ cell; sym = np.array(['A', 'B', 'B', 'B'] * 20)
    for pair in (('A', 'B'), ('B', 'B')):
        r, cn = running_coordination(sym, pos, cell, pair, n_bins=300); per = (2 if pair[0] == pair[1] else 1) / (sym == pair[0]).sum()
        assert abs(r[-1] - 0.5 * np.linalg.norm(shortest_vector(cell))) < 1e-12 and all(abs(cn[k] - per * len(bond_graph(sym, pos, cell, {pair: r[k]})[0])) < 1e-9 for k in range(2, 300, 7))


def test_bond_graph_memory_does_not_grow_as_the_square_of_the_atoms():
    rng = np.random.default_rng(10); cell = np.eye(3) * 30.0; pos = rng.random((2000, 3)) * 30.0; sym = np.array(['A', 'B'] * 1000)
    tracemalloc.start(); e, _, l = bond_graph(sym, pos, cell, {('A', 'B'): 2.0}); peak = tracemalloc.get_traced_memory()[1]; tracemalloc.stop()
    assert peak < 3e7 and len(e) > 0 and l.max() <= 2.0


def test_invalid_structure_inputs_raise():
    sym, pos, cell = cubic('diamond', 5.0, 2); bad = pos.copy(); bad[3, 1] = np.nan
    for call in (lambda: bond_graph(sym, bad, cell, {('A', 'A'): 2.2}), lambda: partial_rdf(sym, bad, cell, ('A', 'A')), lambda: minimum_image(bad - bad[0], cell), lambda: bond_vectors(bad, cell, np.zeros((0, 2), int), np.zeros((0, 3), int)), lambda: bond_graph(sym, pos, cell * np.nan, {('A', 'A'): 2.2})):
        with pytest.raises(ValueError, match='finite'): call()
    with pytest.raises(ValueError, match='bond_graph: cutoffs'): bond_graph(sym, pos, cell, {})
    with pytest.raises(ValueError, match='bond_graph: each cutoff'): bond_graph(sym, pos, cell, {('A', 'A'): -1.0})
    with pytest.raises(ValueError, match='absent from the symbols'): bond_graph(sym, pos, cell, {('A', 'Si'): 2.4})
    for key in ('AB', 'ABC', ('A',), ('A', 'B', 'B'), 5):
        with pytest.raises(ValueError, match='pair of species'): bond_graph(sym, pos, cell, {key: 2.4})
    e, o, l = bond_graph([], np.zeros((0, 3)), cell, {('A', 'B'): 2.4}); assert e.shape == (0, 2) and o.shape == (0, 3) and l.shape == (0,)
    with pytest.raises(ValueError, match='given twice'): bond_graph(sym, pos, cell, {('A', 'B'): 2.4, ('B', 'A'): 2.0})
    with pytest.raises(ValueError, match='partial_rdf: n_bins'): partial_rdf(sym, pos, cell, ('A', 'A'), n_bins=0)
    with pytest.raises(ValueError, match='ensemble_rdf: n_bins'): ensemble_rdf([(sym, pos, cell)], ('A', 'A'), n_bins=0)
    with pytest.raises(ValueError, match='first_minimum'): first_minimum(np.arange(5.0), np.zeros(4))
    with pytest.raises(ValueError, match='first_minimum: smoothing_A'): first_minimum(np.arange(5.0), np.zeros(5), smoothing_A=-0.1)
    with pytest.raises(ValueError, match='first_minimum: smoothing_A'): first_minimum(np.arange(5.0), np.zeros(5), prominence=np.nan)
    with pytest.raises(ValueError, match='ensemble_rdf: the cell'): ensemble_rdf([(sym, pos, np.diag([10.0, 10.0, 0.0]))], ('A', 'A'))
    with pytest.raises(ValueError, match='running_coordination: the cell'): running_coordination(sym, pos, np.diag([10.0, 10.0, 0.0]), ('A', 'A'))
    with pytest.raises(ValueError, match='non-singular'): partial_rdf(sym, pos, np.diag([10.0, 10.0, 0.0]), ('A', 'A'))


def test_first_minimum_reports_the_peak_that_closes_the_basin():
    r = np.linspace(0.01, 5.0, 1000); gauss = lambda c, w, h: h * np.exp(-0.5 * ((r - c) / w) ** 2)
    f = first_minimum(r, gauss(1.75, 0.04, 20) + gauss(2.3, 0.05, 3) + gauss(3.2, 0.15, 2)); assert abs(f['peak_A'] - 1.75) < 0.01 and abs(f['next_peak_A'] - 2.3) < 0.01 and f['basin_A'][1] < 2.3 and f['resolved']
    g = gauss(1.75, 0.04, 20) + gauss(3.2, 0.15, 2); bump = first_minimum(r, g + gauss(2.3, 0.05, 0.15)); clean = first_minimum(r, g)
    assert clean['basin_max_g'] < 0.02 and bump['basin_max_g'] > 0.1 and abs(bump['basin_middle_A'] - 2.3) < 0.15


def test_numeric_species_read_as_their_string_and_an_absent_species_raises():
    _, pos, cell = cubic('sc', 3.0, 4); sym = np.where(np.arange(len(pos)) % 3 == 0, 'A', 'B'); num = np.where(sym == 'A', 14, 8)
    for pair, names in (((14, 8), ('A', 'B')), ((8, 8), ('B', 'B'))): assert np.array_equal(partial_rdf(num, pos, cell, pair)[1], partial_rdf(sym, pos, cell, names)[1])
    with pytest.raises(ValueError, match='species Si is not in the cell'): partial_rdf(sym, pos, cell, ('Si', 'A'))


def test_minimum_image_is_exact_and_cheap_in_a_skewed_basis():
    b = np.diag([20.0, 20.0, 20.0]); s = np.array([[1, 0, 0], [3, 1, 0], [9, 3, 1]], float) @ b; v = np.random.default_rng(2).normal(size=(200, 3)) * 25
    got = np.linalg.norm(minimum_image(v, s), axis=1); want = np.linalg.norm(minimum_image(v, b), axis=1); assert np.allclose(got, want, atol=1e-9)
