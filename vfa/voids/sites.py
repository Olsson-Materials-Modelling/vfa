"""The void space seen from each atom: its share of the accessible volume and its distance to the nearest pore."""
import numpy as np
from ..structure.cell import _valid, minimum_image, volume
from .clearance import _probe, _centres, _covered, _field


def atom_free_volume(C, n, cell, positions, radii, probe, k=12, chunk=1 << 18, workers=-1):
    """Accessible volume (A3) of each atom: grid points with clearance >= probe, each owned by the atom with the nearest surface."""
    pos, cell = _valid(positions, cell, 'atom_free_volume'); C = _field(C, 'atom_free_volume'); n = tuple(int(x) for x in n); probe = _probe(probe, 'atom_free_volume')
    if C.shape != n: raise ValueError(f'atom_free_volume: the field must have the grid shape {n}, not {C.shape}')
    free = np.flatnonzero(C.ravel() >= probe)
    c, own = _covered(pos, cell, radii, lambda: (_centres(free[s:s + chunk], n, cell) for s in range(0, free.size, chunk)), k, chunk, 0.0, workers, 'atom_free_volume', owners=True)
    if c.size and np.abs(c - C.ravel()[free]).max() > 1e-6: raise ValueError('atom_free_volume: the field is not the clearance of these positions and radii')
    return np.bincount(own, minlength=len(pos)) * volume(cell) / C.size


def cavity_distance(positions, cell, pore_xyz):
    """Distance (A) from each atom to the nearest pore centre, minimum image."""
    pos, cell = _valid(positions, cell, 'cavity_distance'); pores = np.asarray(pore_xyz, float).reshape(-1, 3)
    if not np.all(np.isfinite(pores)): raise ValueError('cavity_distance: the pore centres must be finite')
    return np.array([np.linalg.norm(minimum_image(pores - p, cell), axis=1).min() for p in pos]) if len(pores) else np.full(len(pos), np.nan)
