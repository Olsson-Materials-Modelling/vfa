"""Bond graph, rings and pair correlations on lattices with known answers."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vfa.structure.graph import bond_graph, coordination, bond_vectors, angles, tetrahedral_order, planarity
from vfa.structure.rings import primitive_rings, rings_converged, ring_sizes, canonical
from vfa.structure.rdf import partial_rdf, running_coordination, first_minimum


def diamond(a=5.0, n=2, two_species=False):
    """Diamond (or zincblende when two_species) lattice; the two sublattices bond at a sqrt(3)/4."""
    fcc = [(0, 0, 0), (.5, .5, 0), (.5, 0, .5), (0, .5, .5)]; pos, sym = [], []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for x, y, z in fcc:
                    pos.append(((i + x) * a, (j + y) * a, (k + z) * a)); sym.append('A'); pos.append(((i + x + .25) * a, (j + y + .25) * a, (k + z + .25) * a)); sym.append('B' if two_species else 'A')
    return np.array(sym), np.array(pos), np.eye(3) * a * n


def test_diamond_graph_and_angles():
    sym, pos, cell = diamond(); e, o, l = bond_graph(sym, pos, cell, {('A', 'A'): 5.0 * np.sqrt(3) / 4 * 1.01}); z = coordination(len(sym), e)
    assert set(z) == {4} and len(e) == 2 * len(sym) and abs(l - 5.0 * np.sqrt(3) / 4).max() < 1e-9
    v = bond_vectors(pos, cell, e, o); assert abs(angles(v, range(len(sym))) - 109.4712).max() < 1e-3 and abs(tetrahedral_order(v, range(len(sym))) - 1).max() < 1e-9


def test_diamond_rings():
    sym, pos, cell = diamond(); e, o, _ = bond_graph(sym, pos, cell, {('A', 'A'): 2.2}); rings, cap = rings_converged(len(sym), e, o, bipartite=False, start=8)
    assert len(rings) == 2 * len(sym) and set(ring_sizes(rings, bipartite=False)) == {6}                  # two six-rings per atom
    sym2, _, _ = diamond(two_species=True); e2, o2, _ = bond_graph(sym2, pos, cell, {('A', 'B'): 2.2}); r2 = primitive_rings(len(sym2), e2, o2, 12)
    assert len(r2) == len(rings) and set(ring_sizes(r2)) == {3} and canonical((2, 3, 1)) == canonical((1, 2, 3)) == canonical((3, 2, 1))


def test_graphene_planarity():
    a = 2.46; cell = np.array([[a * 3, 0, 0], [0, a * np.sqrt(3) * 2, 0], [0, 0, 20.0]]); pos = []
    for i in range(3):
        for j in range(2):
            x0, y0 = i * a, j * a * np.sqrt(3); pos += [(x0, y0, 10.0), (x0 + a / 2, y0 + a * np.sqrt(3) / 6, 10.0), (x0 + a / 2, y0 + a * np.sqrt(3) / 2, 10.0), (x0, y0 + 2 * a * np.sqrt(3) / 3, 10.0)]
    pos = np.array(pos); sym = np.array(['C'] * len(pos)); e, o, l = bond_graph(sym, pos, cell, {('C', 'C'): a / np.sqrt(3) * 1.05}); v = bond_vectors(pos, cell, e, o)
    assert set(coordination(len(sym), e)) == {3} and abs(planarity(v, range(len(sym))) - 360).max() < 1e-6


def test_rdf_counts_neighbours():
    sym, pos, cell = diamond(); r, cn = running_coordination(sym, pos, cell, ('A', 'A'), r_max=4.0, n_bins=400); assert abs(cn[np.searchsorted(r, 2.5)] - 4) < 1e-9
    r, g = partial_rdf(sym, pos, cell, ('A', 'A'), r_max=4.0, n_bins=400); fm = first_minimum(r, g, search=(1.5, 3.2)); assert abs(fm['peak_A'] - 5.0 * np.sqrt(3) / 4) < 0.05 and fm['minimum_A'] > fm['peak_A']      # the smoothing window is 0.09 A wide


def test_ring_criteria_differ_on_a_cubic_lattice():
    from vfa.structure.rings import rings, king_rings, guttman_rings, canonical
    a = 3.0; n = 3; pos = np.array([(i * a, j * a, k * a) for i in range(n) for j in range(n) for k in range(n)], float); cell = np.eye(3) * a * n; sym = np.array(['A'] * len(pos))
    e, o, _ = bond_graph(sym, pos, cell, {('A', 'A'): a * 1.01})
    prim = rings(len(sym), e, o, 8, 'primitive', bipartite=False); king = king_rings(len(sym), e, o, 8, bipartite=False); gut = guttman_rings(len(sym), e, o, 8, bipartite=False)
    sizes = lambda r: dict(zip(*np.unique([len(x) for x in r], return_counts=True)))
    assert sizes(prim) == {4: 81, 6: 108} and sizes(king) == {4: 81, 6: 162} and sizes(gut) == {4: 81}       # skew hexagons are primitive; 2x1 rectangles are King's for collinear neighbours; Guttman sees the squares only
    assert not {canonical(x) for x in prim if len(x) == 6} & {canonical(x) for x in king if len(x) == 6}


def test_ring_criteria_agree_on_diamond():
    from vfa.structure.rings import rings
    sym, pos, cell = diamond(); e, o, _ = bond_graph(sym, pos, cell, {('A', 'A'): 2.2})
    counts = {c: len(rings(len(sym), e, o, 10, c, bipartite=False)) for c in ('primitive', 'king', 'guttman')}; assert set(counts.values()) == {2 * len(sym)}


def test_bond_graph_matches_brute_force():
    from vfa.structure.cell import SHIFTS
    rng = np.random.default_rng(4); cell = np.array([[9, 0, 0], [2, 9, 0], [1, 1, 9.0]]); pos = rng.random((50, 3)) @ cell; sym = np.array(['A', 'B'] * 25); cut = {('A', 'B'): 2.4, ('B', 'B'): 1.9}
    e, o, l = bond_graph(sym, pos, cell, cut); found = {(int(i), int(j), tuple(int(x) for x in s)) for (i, j), s in zip(e, o)}; brute = set()
    for s in SHIFTS.astype(int):
        d = np.linalg.norm((pos + s @ cell)[None] - pos[:, None], axis=2)
        for i in range(50):
            for j in range(50):
                c = cut.get(tuple(sorted((sym[i], sym[j]))))
                if c is None or d[i, j] > c or (not s.any() and i >= j): continue
                key = (i, j, tuple(s)); alt = (j, i, tuple(-s))
                if alt not in brute: brute.add(key)
    assert len(found) == len(brute) and all(k in brute or (k[1], k[0], tuple(-x for x in k[2])) in brute for k in found)


def test_ring_extents_on_diamond():
    from vfa.structure.rings import rings, ring_extents
    sym, pos, cell = diamond(); e, o, _ = bond_graph(sym, pos, cell, {('A', 'A'): 2.2}); r = rings(len(sym), e, o, 10, 'primitive', bipartite=False)
    ex = ring_extents(r, pos, cell); assert abs(ex - ex[0]).max() < 1e-9 and 4.0 < ex[0] < 5.0        # every six-ring of diamond has the same chair extent, well inside the 10 A cell
