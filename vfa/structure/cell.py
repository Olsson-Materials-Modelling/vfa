"""Periodic geometry of one cell: widths, the shortest lattice vector, folding, image shells, the exact minimum image, separations, distances."""
import itertools
import numpy as np

SHIFTS = np.array(list(itertools.product((-1, 0, 1), repeat=3)), float)


def _valid(positions, cell, name):
    c = np.asarray(cell, float); p = np.asarray(positions, float)
    if c.shape != (3, 3) or not np.all(np.isfinite(c)) or abs(np.linalg.det(c)) <= 1e-12 * np.prod(np.linalg.norm(c, axis=1)): raise ValueError(f'{name}: the cell must be a finite, non-singular 3 x 3 matrix of lattice vectors (A)')
    if p.ndim != 2 or p.shape[1] != 3: raise ValueError(f'{name}: positions must have shape (n_atoms, 3), not {p.shape}')
    if not np.all(np.isfinite(p)): raise ValueError(f'{name}: positions must be finite ({int((~np.isfinite(p)).any(1).sum())} atoms are not)')
    return p, c


def volume(cell):
    """Cell volume (A3)."""
    return float(abs(np.linalg.det(np.asarray(cell, float))))


def fractional(positions, cell):
    """Fractional coordinates."""
    return np.asarray(positions, float) @ np.linalg.inv(np.asarray(cell, float))


def widths(cell):
    """Perpendicular widths (A): the distance between each pair of opposite faces."""
    return 1.0 / np.linalg.norm(np.linalg.inv(np.asarray(cell, float)), axis=0)


def _reduced(c):
    U = np.eye(3, dtype=np.int64); b = c.copy(); changed = True
    while changed:
        changed = False
        for i, j in itertools.permutations(range(3), 2):
            mu = int(np.rint(b[i] @ b[j] / (b[j] @ b[j])))
            if mu == 0: continue
            u = U[i] - mu * U[j]; v = u @ c
            if v @ v < (1 - 1e-12) * (b[i] @ b[i]): U[i] = u; b[i] = v; changed = True
    return b, U


def shortest_vector(cell):
    """Shortest nonzero lattice vector (A), exact in any basis."""
    c = _valid(np.zeros((0, 3)), cell, 'shortest_vector')[1]; b, U = _reduced(c)
    m = np.floor(np.linalg.norm(b, axis=1).min() / widths(b) * (1 + 1e-9)).astype(int); n = np.array(list(itertools.product(*[range(-k, k + 1) for k in m])), np.int64)
    t = (n[np.any(n != 0, axis=1)] @ U) @ c; return t[np.argmin(np.linalg.norm(t, axis=1))]


def folded(positions, cell, name='folded'):
    """Positions moved into the cell by lattice translations n, and n (positions = folded + n @ cell)."""
    pos, cell = _valid(positions, cell, name); n = np.floor(fractional(pos, cell)).astype(np.int64); return pos - n @ cell, n


def wrap(positions, cell, name='wrap'):
    """Positions folded into the cell."""
    return folded(positions, cell, name)[0]


def shell(cell, reach):
    """Lattice translations whose images of folded positions hold every atom within reach (A) of any point of the cell."""
    m = np.maximum(np.ceil(float(reach) / widths(cell)).astype(int), 0); return np.array(list(itertools.product(*[range(-k, k + 1) for k in m])), float)


def images(positions, cell, values=None, reach=None, name='images'):
    """Folded positions replicated over shell(cell, reach) (26 neighbours when None) as rows k * n_atoms + i, with values alongside."""
    cell = np.asarray(cell, float); pos = wrap(positions, cell, name); sh = SHIFTS if reach is None else shell(cell, reach); rep = (pos[None] + (sh @ cell)[:, None]).reshape(-1, 3)
    if values is None: return rep
    v = np.asarray(values, float)
    if v.shape != (len(pos),): raise ValueError(f'{name}: one value per atom is needed ({len(pos)} atoms), not shape {v.shape}')
    return rep, np.tile(v, len(sh))


def orthogonal(cell, tol=1e-12):
    """Whether the lattice vectors are mutually perpendicular."""
    c = np.asarray(cell, float); g = c @ c.T; d = np.sqrt(np.diag(g)); return bool(np.all(np.abs(g - np.diag(np.diag(g))) <= tol * np.outer(d, d)))


def minimum_image(v, cell):
    """Vectors moved to their shortest periodic image, exact in any cell."""
    cell = _valid(np.zeros((0, 3)), cell, 'minimum_image')[1]; v = np.asarray(v, float)
    if not np.all(np.isfinite(v)): raise ValueError('minimum_image: the vectors must be finite')
    if orthogonal(cell): f = v @ np.linalg.inv(cell); return (f - np.rint(f)) @ cell
    b = _reduced(cell)[0]; f = v @ np.linalg.inv(b); flat = ((f - np.rint(f)) @ b).reshape(-1, 3); L = np.linalg.norm(flat, axis=1); w = widths(b); hard = np.flatnonzero(L >= 0.5 * w.min())
    if hard.size:
        m = np.floor(2 * L[hard].max() / w).astype(int); best = flat[hard]; bl = L[hard]
        for s in itertools.product(*[range(-k, k + 1) for k in m]):
            c = flat[hard] + np.asarray(s, float) @ b; cl = np.linalg.norm(c, axis=1); better = cl < bl - 1e-12; best = np.where(better[:, None], c, best); bl = np.where(better, cl, bl)
        flat = flat.copy(); flat[hard] = best
    return flat.reshape(v.shape)


def separations(positions, cell):
    """Minimum-image vectors between all atoms."""
    pos = np.asarray(positions, float); return minimum_image(pos[None] - pos[:, None], cell)


def distances(positions, cell):
    """Minimum-image distances between all atoms."""
    return np.linalg.norm(separations(positions, cell), axis=2)
