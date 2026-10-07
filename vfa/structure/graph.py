"""The bond graph of a periodic cell from per-pair cutoffs: edges with their image offsets, coordination, bond vectors and angles."""
import numpy as np


def _key(a, b):
    return (a, b) if a <= b else (b, a)


def bond_graph(symbols, positions, cell, cutoffs):
    """Edges (i, j), image offsets of j and bond lengths for every species pair with a cutoff (A); each bond once."""
    sym = np.asarray(symbols); pos = np.asarray(positions, float); cell = np.asarray(cell, float); cut = {_key(*k): float(v) for k, v in cutoffs.items()}; rmax = max(cut.values())
    reps = np.ceil(rmax * np.linalg.norm(np.linalg.inv(cell), axis=0)).astype(int); n = len(sym); edges, offsets, lengths = [], [], []
    for shift in np.array([[a, b, c] for a in range(-reps[0], reps[0] + 1) for b in range(-reps[1], reps[1] + 1) for c in range(-reps[2], reps[2] + 1)]):
        if tuple(shift) < tuple(-shift): continue
        d = np.linalg.norm((pos + shift @ cell)[None] - pos[:, None], axis=2); near = d <= rmax
        if not shift.any(): near &= np.triu(np.ones((n, n), bool), 1)
        for i, j in zip(*np.nonzero(near)):
            c = cut.get(_key(sym[i], sym[j]))
            if c is not None and d[i, j] <= c: edges.append((i, j)); offsets.append(shift); lengths.append(d[i, j])
    e, o, l = np.array(edges, np.int64).reshape(-1, 2), np.array(offsets, np.int64).reshape(-1, 3), np.array(lengths, float)
    order = np.lexsort((o[:, 2], o[:, 1], o[:, 0], e[:, 1], e[:, 0])) if len(e) else np.zeros(0, int); return e[order], o[order], l[order]


def coordination(n_atoms, edges):
    """Bonds at each atom."""
    z = np.zeros(n_atoms, np.int64); np.add.at(z, edges.ravel(), 1); return z


def bond_vectors(positions, cell, edges, offsets):
    """Bond vectors from each atom to each of its neighbours, as a list per atom."""
    pos = np.asarray(positions, float); cell = np.asarray(cell, float); out = [[] for _ in range(len(pos))]
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
        v = v[np.argsort(np.linalg.norm(v, axis=1))[:4]]; u = v / np.linalg.norm(v, axis=1)[:, None]; c = [u[a] @ u[b] for a in range(4) for b in range(a + 1, 4)]; q[i] = 1 - 3 / 8 * sum((x + 1 / 3) ** 2 for x in c)
    return q


def planarity(vectors, atoms):
    """Sum of the three bond angles (degrees) at three-bonded atoms; NaN otherwise."""
    p = np.full(len(vectors), np.nan)
    for i in atoms:
        v = vectors[i]
        if len(v) != 3: continue
        u = v / np.linalg.norm(v, axis=1)[:, None]; p[i] = sum(np.degrees(np.arccos(np.clip(u[a] @ u[b], -1, 1))) for a in range(3) for b in range(a + 1, 3))
    return p
