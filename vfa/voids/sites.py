"""What the void space looks like from each atom: its share of the accessible volume and its distance to the nearest pore."""
import numpy as np
from scipy.spatial import cKDTree
from ..structure.cell import images, minimum_image, volume


def atom_free_volume(C, n, cell, positions, probe):
    """Accessible volume (A3) nearest to each atom: grid points with clearance >= probe, assigned to the closest atom centre."""
    cell = np.asarray(cell, float); f = np.stack(np.meshgrid(*[(np.arange(m) + 0.5) / m for m in n], indexing='ij'), -1).reshape(-1, 3); free = C.ravel() >= probe
    rep = images(positions, cell); _, i = cKDTree(rep).query((f @ cell)[free]); out = np.zeros(len(positions)); np.add.at(out, i % len(positions), 1.0)
    return out * volume(cell) / C.size


def cavity_distance(positions, cell, pore_xyz):
    """Distance (A) from each atom to the nearest pore centre, minimum image."""
    pos = np.asarray(positions, float); return np.array([np.linalg.norm(minimum_image(pore_xyz - p, cell), axis=1).min() for p in pos]) if len(pore_xyz) else np.full(len(pos), np.nan)
