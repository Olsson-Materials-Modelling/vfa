"""Rings: the bipartite check, rings through the periodic images, convergence of the cap, and fixed ring sets of the three criteria on an example cell."""
import sys, hashlib
from pathlib import Path
import numpy as np, pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lattices import cubic, graphene, sizes
from vfa.structure.graph import bond_graph
from vfa.structure.rings import rings, rings_converged, ring_sizes, ring_extents, canonical, CRITERIA, primitive_rings, king_rings, guttman_rings
from vfa.io.extxyz import read_extxyz

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = {'primitive': ({6: 9, 8: 8, 10: 12, 12: 4, 14: 10, 16: 4, 18: 1, 20: 5, 22: 3, 24: 1}, '9fa551c2bdfff927977627aa610a469de171f4011cc66624422c27f84caa3dd9'),
             'king': ({6: 9, 8: 8, 10: 15, 12: 14, 14: 21, 16: 19, 18: 20, 20: 18, 22: 8, 24: 7}, '7b2caa5f5551022354717e3fe51c4baaaa9499d7d0955a9c8b970e1f0056ee83'),
             'guttman': ({6: 9, 8: 8, 10: 12, 12: 3, 14: 9, 16: 4, 20: 3}, 'e9e445633882d2b3ead8e8d85ded4839dae016866e1575108daaa14a453e3dd8')}


def alpha_quartz():
    """Alpha-quartz primitive cell (P3_221), Si on the two-fold axis at z = 1/6: three Si and six O."""
    a, c = 4.9134, 5.4052; cell = np.array([[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, c]])
    ops = [lambda x, y, z: (x, y, z), lambda x, y, z: (-y, x - y, z + 2 / 3), lambda x, y, z: (-x + y, -x, z + 1 / 3), lambda x, y, z: (y, x, -z), lambda x, y, z: (x - y, -y, -z + 1 / 3), lambda x, y, z: (-x, -x + y, -z + 2 / 3)]
    fr, sym = [], []
    for s, p in (('Si', (0.4697, 0.0, 1 / 6)), ('O', (0.4135, 0.2669, 0.1191 + 1 / 6))):
        for op in ops:
            q = np.array(op(*p)) % 1.0
            if not any(np.allclose((q - r + 0.5) % 1 - 0.5, 0, atol=1e-6) for r in fr): fr.append(q); sym.append(s)
    return np.array(sym), np.array(fr) @ cell, cell


def test_bipartite_default_is_checked():
    sym, pos, cell = graphene(stone_wales=True); e, o, _ = bond_graph(sym, pos, cell, {('C', 'C'): 1.6})
    for crit in ('primitive', 'king', 'guttman'):
        assert sizes(rings(len(sym), e, o, 14, crit, bipartite=False)) == {5: 2, 6: 44, 7: 2}
        with pytest.raises(ValueError, match='not bipartite'): rings(len(sym), e, o, 14, crit)
    with pytest.raises(ValueError, match='5 atoms'): ring_sizes(rings(len(sym), e, o, 14, bipartite=False))
    with pytest.raises(ValueError, match='not bipartite'): rings_converged(len(sym), e, o, start=8)
    sym, pos, cell = graphene(); e, o, _ = bond_graph(sym, pos, cell, {('C', 'C'): 1.6}); r = rings(len(sym), e, o, 14)
    assert sizes(r) == {6: 48} and set(ring_sizes(r)) == {3}


def test_odd_cycle_through_the_images_is_not_bipartite():
    sym, pos, cell = cubic('sc', 3.0, (3, 2, 2)); e, o, _ = bond_graph(sym, pos, cell, {('A', 'A'): 3.03})
    assert sizes(rings(len(sym), e, o, 4, bipartite=False)) == {4: 36}
    with pytest.raises(ValueError, match='not bipartite'): rings(len(sym), e, o, 4)
    sym, pos, cell = cubic('sc', 3.0, (1, 1, 1)); e, o, _ = bond_graph(sym, pos, cell, {('A', 'A'): 3.03})
    with pytest.raises(ValueError, match='not bipartite'): rings(len(sym), e, o, 4)


def test_rings_through_the_images_are_distinct():
    sym, pos, cell = cubic('sc', 3.0, (2, 2, 2)); e, o, _ = bond_graph(sym, pos, cell, {('A', 'A'): 3.03})
    for crit in ('primitive', 'king', 'guttman'): assert sizes(rings(len(sym), e, o, 4, crit, bipartite=False)) == {4: 24}
    r = rings(len(sym), e, o, 6, bipartite=False); ri = rings(len(sym), e, o, 6, bipartite=False, images=True)
    assert sizes(r) == {4: 24, 6: 32} and r == rings(len(sym), e, o, 6) and [tuple(a for a, _ in x) for x in ri] == r
    assert len({canonical(x) for x in ri}) == 56 and len({canonical(x) for x in r}) == 10
    assert [(len(x), canonical(x)) for x in r] == sorted((len(x), canonical(x)) for x in r)
    ext = ring_extents(ri, pos, cell); assert np.allclose(ext[:24], 3 * np.sqrt(2)) and np.allclose(ext[24:], 3 * np.sqrt(3))


def test_ring_extents_beyond_the_cell():
    sym, pos, cell = cubic('sc', 3.0, (1, 1, 1)); e, o, _ = bond_graph(sym, pos, cell, {('A', 'A'): 3.03})
    ri = rings(len(sym), e, o, 6, bipartite=False, images=True); ext = ring_extents(ri, pos, cell)
    assert sizes(ri) == {4: 3, 6: 4} and np.allclose(ext, [3 * np.sqrt(2)] * 3 + [3 * np.sqrt(3)] * 4) and ext.min() > 3.0
    assert np.allclose(ring_extents([tuple(a for a, _ in x) for x in ri], pos, cell), 0.0)


def test_canonical_with_images():
    ring = ((0, (0, 0, 0)), (1, (0, 0, 0)), (3, (0, 1, 0)), (2, (0, 1, 0)))
    moved = [(a, (t[0] + 1, t[1] - 2, t[2] + 3)) for a, t in ring]; turned = (moved[2:] + moved[:2])[::-1]
    assert canonical(turned) == canonical(ring) and canonical(((0, (0, 0, 0)), (1, (0, 0, 0)), (3, (0, 0, 0)), (2, (0, 1, 0)))) != canonical(ring)
    assert canonical([a for a, _ in turned]) == canonical([a for a, _ in ring]) == (0, 1, 3, 2)


def test_quartz_converges_from_a_small_cap():
    sym, pos, cell = alpha_quartz(); e, o, _ = bond_graph(sym, pos, cell, {('Si', 'O'): 1.8})
    r, cap = rings_converged(len(sym), e, o, start=4, step=4)
    assert sizes(r) == {12: 3, 16: 15} and cap == 32 and set(ring_sizes(r)) == {6, 8}
    assert sizes(rings(len(sym), e, o, 16, 'king')) == {12: 3, 16: 21} and sizes(rings(len(sym), e, o, 16, 'guttman')) == {12: 3}
    with pytest.raises(RuntimeError, match='no primitive ring'): rings_converged(len(sym), e, o, start=4, step=4, hard_cap=8)
    with pytest.raises(RuntimeError, match='not confirmed'): rings_converged(len(sym), e, o, start=4, step=4, hard_cap=24)
    with pytest.raises(ValueError, match='margin'): rings_converged(len(sym), e, o, margin=0.5)


def test_regression_rings_of_the_example_cell_are_not_confirmed_within_the_hard_cap():
    sym, pos, cell = read_extxyz(ROOT / 'example/data/structures/rho0.060_1.extxyz'); e, o, _ = bond_graph(sym, pos, cell, {('A', 'B'): 2.045})
    r = rings(len(sym), e, o, 24, images=True)
    assert sizes(r) == REFERENCE['primitive'][0] and ring_extents(r, pos, cell).max() < np.linalg.norm(cell, axis=1).min()
    with pytest.raises(RuntimeError, match='not confirmed'): rings_converged(len(sym), e, o, start=16)


def test_regression_criteria_reproduce_fixed_ring_sets_on_the_example_cell():
    sym, pos, cell = read_extxyz(ROOT / 'example/data/structures/rho0.060_1.extxyz'); e, o, _ = bond_graph(sym, pos, cell, {('A', 'B'): 2.045})
    for crit, (hist, digest) in REFERENCE.items():
        found = rings(len(sym), e, o, 24, crit); keys = sorted({tuple(int(x) for x in canonical(r)) for r in found})
        assert sizes(found) == hist and len(keys) == len(found) and hashlib.sha256(repr(keys).encode()).hexdigest() == digest


def test_every_component_of_the_bond_graph_is_two_coloured():
    e = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 4]]); o = np.zeros((7, 3), int)
    for crit in CRITERIA:
        assert sizes(rings(7, e, o, 6, crit, bipartite=False)) == {3: 1, 4: 1}
        with pytest.raises(ValueError, match='not bipartite'): rings(7, e, o, 6, crit)
    with pytest.raises(ValueError, match='not bipartite'): rings_converged(7, e, o, start=6)
    for f in (primitive_rings, king_rings, guttman_rings):
        r = f(4, e[:4], o[:4], 6); assert [canonical(x) for x in r] == [(0, 1, 2, 3)] and all(np.isscalar(a) for a in r[0])
        with pytest.raises(ValueError, match='not bipartite'): f(7, e, o, 6)


def test_three_atom_rings_are_kept_by_every_criterion():
    e = np.array([[0, 1], [1, 2], [2, 0]]); o = np.zeros((3, 3), int)
    for crit in CRITERIA: assert [canonical(r) for r in rings(3, e, o, 3, crit, bipartite=False)] == [(0, 1, 2)] and rings(3, e, o, 2, crit, bipartite=False) == []


def test_no_criterion_returns_a_ring_longer_than_the_cap():
    for n in (5, 6, 7):
        e = np.c_[np.arange(n), (np.arange(n) + 1) % n]; o = np.zeros((n, 3), int)
        for crit in CRITERIA: assert rings(n, e, o, n - 1, crit, bipartite=False) == [] and [canonical(r) for r in rings(n, e, o, n, crit, bipartite=False)] == [tuple(range(n))]


def test_ring_extents_unwrap_an_atom_tuple_from_any_starting_atom():
    L, s = 7.0, 1.4; ang = np.arange(6) * np.pi / 3; pos = np.c_[s * np.cos(ang) + 0.3, s * np.sin(ang) - 0.2, np.full(6, 6.9)] % L; cell = np.eye(3) * L
    for k in range(6):
        ring = tuple(np.roll(np.arange(6), -k)); assert abs(ring_extents([ring, ring[::-1]], pos, cell) - 2 * s).max() < 1e-12
