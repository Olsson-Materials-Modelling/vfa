"""The bond graph of a periodic cell from per-pair cutoffs: edges with their image offsets, coordination, bond vectors and angles."""
import numpy as np
from scipy.spatial import cKDTree
from .cell import _valid, folded, shell


def _key(a, b):
    return (a, b) if a <= b else (b, a)


def bond_graph(symbols, positions, cell, cutoffs):
    """Bonds within the pair cutoffs, each once: edges (i, j), image offsets of j relative to the input positions, lengths (A)."""
    if not all(isinstance(k, tuple) and len(k) == 2 and all(isinstance(x, str) for x in k) for k in dict(cutoffs)): raise ValueError(f"bond_graph: each cutoff key is a pair of species such as ('A', 'B'), not {list(dict(cutoffs))}")
    pos, n0 = folded(positions, cell, 'bond_graph'); cell = np.asarray(cell, float); sym = np.asarray(symbols); n = len(pos); pairs = [_key(*k) for k in dict(cutoffs)]; cut = {_key(*k): float(v) for k, v in dict(cutoffs).items()}
    if sym.shape != (n,): raise ValueError(f'bond_graph: one symbol per atom is needed ({n} atoms), not shape {sym.shape}')
    if not cut: raise ValueError('bond_graph: cutoffs must name at least one species pair')
    if not all(np.isfinite(v) and v > 0 for v in cut.values()): raise ValueError(f'bond_graph: each cutoff must be a positive, finite distance (A), not {dict(cutoffs)}')
    if len(pairs) != len(cut): raise ValueError(f'bond_graph: a species pair is given twice (in both orders) in {dict(cutoffs)}')
    absent = sorted({s for k in cut for s in k} - set(sym.tolist()))
    if n and absent: raise ValueError(f'bond_graph: the cutoffs name species absent from the symbols: {absent}; present: {sorted(set(sym.tolist()))}')
    if n == 0: return np.zeros((0, 2), np.int64), np.zeros((0, 3), np.int64), np.zeros(0)
    H = shell(cell, max(cut.values())).astype(np.int64); H = H[[tuple(s) >= tuple(-s) for s in H]]; P = np.concatenate([pos + s @ cell for s in H])
    lists = cKDTree(pos).query_ball_point(P, max(cut.values()) * (1 + 1e-9) + 1e-12); p = np.repeat(np.arange(len(P)), [len(x) for x in lists]); i = np.fromiter((a for x in lists for a in x), np.int64, len(p)); k, j = p // n, p % n
    keep = np.any(H[k] != 0, axis=1) | (i < j); p, i, j, k = p[keep], i[keep], j[keep], k[keep]; d = np.linalg.norm(P[p] - pos[i], axis=1)
    species, code = np.unique(sym, return_inverse=True); M = np.array([[cut.get(_key(a, b), -np.inf) for b in species] for a in species]); keep = d <= M[code[i], code[j]]
    e, o, l = np.c_[i[keep], j[keep]].astype(np.int64), H[k[keep]], d[keep]; o = o + n0[e[:, 0]] - n0[e[:, 1]]
    order = np.lexsort((o[:, 2], o[:, 1], o[:, 0], e[:, 1], e[:, 0])); return e[order], o[order], l[order]


def coordination(n_atoms, edges):
    """Bonds at each atom."""
    z = np.zeros(n_atoms, np.int64); np.add.at(z, edges.ravel(), 1); return z


def bond_vectors(positions, cell, edges, offsets):
    """Bond vectors from each atom to each of its neighbours, as a list per atom."""
    pos, cell = _valid(positions, cell, 'bond_vectors'); out = [[] for _ in range(len(pos))]
    for (i, j), o in zip(edges, offsets):
        v = pos[j] + o @ cell - pos[i]; out[i].append(v); out[j].append(-v)
    return [np.array(v).reshape(-1, 3) for v in out]


def angles(vectors, atoms):
    """Bond angles (degrees) at the given atoms, every pair of bonds."""
    out = []
    for i in atoms:
        v = vectors[i]
        for a in range(len(v)):
            for b in range(a + 1, len(v)): out.append(np.degrees(np.arccos(np.clip(v[a] @ v[b] / (np.linalg.norm(v[a]) * np.linalg.norm(v[b])), -1, 1))))
    return np.array(out)


def tetrahedral_order(vectors, atoms):
    """q of the four shortest bonds: 1 for a regular tetrahedron, NaN below four bonds."""
    q = np.full(len(vectors), np.nan)
    for i in atoms:
        v = vectors[i]
        if len(v) < 4: continue
        v = v[np.argsort(np.linalg.norm(v, axis=1), kind='stable')[:4]]; u = v / np.linalg.norm(v, axis=1)[:, None]; c = [u[a] @ u[b] for a in range(4) for b in range(a + 1, 4)]; q[i] = 1 - 3 / 8 * sum((x + 1 / 3) ** 2 for x in c)
    return q


def planarity(vectors, atoms):
    """Sum of the three bond angles (degrees) at three-bonded atoms; NaN otherwise."""
    p = np.full(len(vectors), np.nan)
    for i in atoms:
        v = vectors[i]
        if len(v) != 3: continue
        u = v / np.linalg.norm(v, axis=1)[:, None]; p[i] = sum(np.degrees(np.arccos(np.clip(u[a] @ u[b], -1, 1))) for a in range(3) for b in range(a + 1, 3))
    return p
