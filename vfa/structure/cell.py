"""Periodic geometry of one cell: images, minimum image, separations, distances."""
import numpy as np

SHIFTS = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], float)


def volume(cell):
    """Cell volume (A3)."""
    return float(abs(np.linalg.det(np.asarray(cell, float))))


def fractional(positions, cell):
    """Fractional coordinates."""
    return np.asarray(positions, float) @ np.linalg.inv(np.asarray(cell, float))


def wrap(positions, cell):
    """Positions folded into the cell."""
    f = fractional(positions, cell) % 1.0; return f @ np.asarray(cell, float)


def images(positions, cell, values=None):
    """Positions replicated into the cell and its 26 neighbours, with per-atom values repeated alongside."""
    pos = np.asarray(positions, float); cell = np.asarray(cell, float); rep = (pos[None] + (SHIFTS @ cell)[:, None]).reshape(-1, 3)
    return rep if values is None else (rep, np.tile(np.asarray(values, float), len(SHIFTS)))


def minimum_image(v, cell):
    """Vectors folded into the nearest periodic image (general cell)."""
    cell = np.asarray(cell, float); f = np.asarray(v, float) @ np.linalg.inv(cell); return (f - np.rint(f)) @ cell


def separations(positions, cell):
    """Minimum-image vectors between all atoms."""
    pos = np.asarray(positions, float); return minimum_image(pos[None] - pos[:, None], cell)


def distances(positions, cell):
    """Minimum-image distances between all atoms."""
    return np.linalg.norm(separations(positions, cell), axis=2)
