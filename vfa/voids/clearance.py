"""The clearance (distance to the nearest atom surface) at points of the cell: on a grid, which also gives the void surface, or at sampled points."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import qmc
from skimage.measure import marching_cubes
from ..structure.cell import images, volume


def exact_clearance(tree, rad, pts, k=12):
    """Clearance at each point from a tree of atom centres and their radii: the k nearest centres, doubled until no omitted centre can be closer in surface distance."""
    pts = np.asarray(pts, float); out = np.full(len(pts), np.inf); pending = np.arange(len(pts)); rmax = rad.max()
    while pending.size:
        k = min(k, tree.n); d, i = tree.query(pts[pending], k=k, workers=-1); d, i = np.atleast_2d(d.T).T, np.atleast_2d(i.T).T; c = np.min(d - rad[i], axis=1); out[pending] = c
        sure = (d[:, -1] - rmax >= c) | (k == tree.n); pending = pending[~sure]; k *= 2
    return out


def _tree(positions, cell, radii, k):
    rep, rad = images(positions, cell, radii); tree = cKDTree(rep); return lambda pts: exact_clearance(tree, rad, pts, k)


def clearance_field(positions, cell, radii, dx=0.15, k=12):
    """Grid mode: the clearance at the centres of a fractional grid of spacing about dx; returns the field and the grid shape."""
    cell = np.asarray(cell, float); n = np.maximum(np.round(np.linalg.norm(cell, axis=1) / dx).astype(int), 8)
    f = np.stack(np.meshgrid(*[(np.arange(m) + 0.5) / m for m in n], indexing='ij'), -1).reshape(-1, 3)
    return _tree(positions, cell, radii, k)(f @ cell).reshape(tuple(n)), n


def sample_clearance(positions, cell, radii, n=8192, sampler='sobol', seed=0, k=12):
    """Pointwise mode: the clearance at n Sobol points (n a power of two, scrambled with the seed) or uniform random points."""
    cell = np.asarray(cell, float)
    if sampler == 'sobol': f = qmc.Sobol(3, scramble=True, seed=seed).random(n)
    elif sampler == 'random': f = np.random.default_rng(seed).random((n, 3))
    else: raise ValueError(sampler)
    return _tree(positions, cell, radii, k)(f @ cell)


def accessible_fraction(c, probe):
    """Fraction of the cell a sphere of the probe radius can occupy, isolated cavities included."""
    return float(np.mean(np.asarray(c) >= probe))


def void_volume(c, cell, probe):
    """Accessible volume (A3)."""
    return accessible_fraction(c, probe) * volume(cell)


def clearance_cdf(c, r):
    """Fraction of the cell with clearance below each r: the cumulative distribution of the clearance."""
    return np.searchsorted(np.sort(np.asarray(c).ravel()), r, side='right') / np.asarray(c).size


def void_surface(C, n, cell, probe):
    """Grid mode: area of the isosurface C = probe by marching cubes on the twice-tiled field, each triangle counted once by its centroid."""
    try: verts, faces, _, _ = marching_cubes(np.tile(C, (2, 2, 2)), level=probe)
    except (ValueError, RuntimeError): return 0.0
    frac = verts / np.asarray(n, float); keep = np.all((frac[faces].mean(1) >= 0) & (frac[faces].mean(1) < 1), axis=1); f = faces[keep]
    if not len(f): return 0.0
    x = frac @ np.asarray(cell, float); a, b = x[f[:, 1]] - x[f[:, 0]], x[f[:, 2]] - x[f[:, 0]]; return float(0.5 * np.linalg.norm(np.cross(a, b), axis=1).sum())


def probe_sweep(C, n, cell, probes):
    """Grid mode: accessible fraction, accessible volume and specific surface (A2 per 1000 A3) at each probe radius."""
    V = volume(cell); return [dict(probe_A=p, fraction=accessible_fraction(C, p), volume_A3=accessible_fraction(C, p) * V, surface_A2=void_surface(C, n, cell, p), specific_surface=1000 * void_surface(C, n, cell, p) / V) for p in probes]


def clearance_statistics(c, thresholds=(0.25, 0.5, 0.75, 1.0, 1.25, 1.5)):
    """Largest clearance, upper percentiles, open fraction and the fraction above each threshold, for either mode."""
    c = np.asarray(c).ravel(); open_ = c[c > 0]
    out = dict(largest_A=float(c.max()), p99_A=float(np.quantile(c, .99)), p95_A=float(np.quantile(c, .95)), open_fraction=float(np.mean(c > 0)), mean_open_A=float(open_.mean()) if open_.size else np.nan)
    out.update({f'above_{t:g}': float(np.mean(c >= t)) for t in thresholds}); return out


def sampling_convergence(positions, cell, radii, probe, counts=(1024, 2048, 4096, 8192, 16384), sampler='sobol', seed=0):
    """Accessible fraction at the probe against the number of sample points."""
    return [dict(n=n, fraction=accessible_fraction(sample_clearance(positions, cell, radii, n, sampler, seed), probe)) for n in counts]


def grid_convergence(positions, cell, radii, probe, spacings=(0.3, 0.2, 0.15, 0.1)):
    """Accessible fraction and specific surface at the probe against the grid spacing."""
    out = []
    for dx in spacings:
        C, n = clearance_field(positions, cell, radii, dx); out.append(dict(dx=dx, fraction=accessible_fraction(C, probe), specific_surface=1000 * void_surface(C, n, cell, probe) / volume(cell)))
    return out
