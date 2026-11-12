"""Cells with known answers for the tests (cubic lattices, diamond and zincblende, graphene with or without a Stone-Wales defect) and a ring-size count."""
import numpy as np

FCC = [(0, 0, 0), (.5, .5, 0), (.5, 0, .5), (0, .5, .5)]


def cubic(kind, a=3.0, n=3, two_species=False):
    """Symbols, positions and cell of an sc, fcc or diamond supercell of n (or n[0] x n[1] x n[2]) cubic cells of side a."""
    m = (n, n, n) if np.isscalar(n) else tuple(n); sites = [(0, 0, 0)] if kind == 'sc' else FCC; pos, sym = [], []
    for i in range(m[0]):
        for j in range(m[1]):
            for k in range(m[2]):
                for x, y, z in sites:
                    pos.append(((i + x) * a, (j + y) * a, (k + z) * a)); sym.append('A')
                    if kind == 'diamond': pos.append(((i + x + .25) * a, (j + y + .25) * a, (k + z + .25) * a)); sym.append('B' if two_species else 'A')
    return np.array(sym), np.array(pos, float), np.diag(np.array(m, float) * a)


def graphene(nx=6, ny=4, a=2.46, stone_wales=False):
    """Symbols, positions and cell of an orthorhombic graphene supercell, one bond turned by 90 degrees when stone_wales."""
    b = a / np.sqrt(3); base = np.array([[0, 0], [a / 2, b / 2], [a / 2, 1.5 * b], [0, 2 * b]])
    pos = np.array([[i * a + x, j * 3 * b + y, 10.0] for i in range(nx) for j in range(ny) for x, y in base]); cell = np.diag([nx * a, ny * 3 * b, 20.0])
    if stone_wales:
        c0 = np.array([nx // 2 * a, ny // 2 * 3 * b + 1.5 * b, 10.0]); i = int(np.argmin(np.linalg.norm(pos - c0, axis=1))); d = np.linalg.norm(pos - pos[i], axis=1); d[i] = np.inf; j = int(np.argmin(d))
        m = (pos[i] + pos[j]) / 2; v = pos[j] - pos[i]; r = np.array([-v[1], v[0], 0.0]); pos[i], pos[j] = m - r / 2, m + r / 2
    return np.array(['C'] * len(pos)), pos, cell


def sizes(found):
    """Number of rings of each length, as a dict."""
    return dict(zip(*[x.tolist() for x in np.unique([len(r) for r in found], return_counts=True)])) if len(found) else {}
