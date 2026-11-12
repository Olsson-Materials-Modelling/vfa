"""The clearance (distance to the nearest atom surface) on a grid or at sampled points, its statistics and the void surface."""
import os
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import qmc
from skimage.measure import marching_cubes
from ..structure.cell import _valid, folded, images, volume, widths


def threads(workers, name='threads'):
    """Threads for the neighbour queries: -1 means every CPU this process may use."""
    if isinstance(workers, bool) or not isinstance(workers, (int, np.integer)) or (workers < 1 and workers != -1): raise ValueError(f'{name}: workers must be -1 or a positive integer, not {workers!r}')
    if workers == -1:
        try: return len(os.sched_getaffinity(0))
        except AttributeError: return os.cpu_count() or 1
    return int(workers)


def _probe(probe, name):
    if not np.isfinite(probe): raise ValueError(f'{name}: the probe radius must be finite, not {probe}')
    return float(probe)


def _radii(radii, n, name):
    r = np.asarray(radii, float)
    if n == 0: raise ValueError(f'{name}: no atoms')
    if r.shape != (n,): raise ValueError(f'{name}: radii must hold one value per atom ({n} atoms), not shape {r.shape}')
    if not np.all(np.isfinite(r)) or np.any(r < 0): raise ValueError(f'{name}: radii must be finite and non-negative')
    return r


def _field(c, name):
    c = np.asarray(c, float)
    if c.size == 0: raise ValueError(f'{name}: the clearance is empty')
    if not np.all(np.isfinite(c)): raise ValueError(f'{name}: the clearance must be finite ({int((~np.isfinite(c)).sum())} values are not)')
    return c


def _nearest(tree, rad, pts, k, chunk, workers, label=None):
    pts = np.asarray(pts, float); out = np.empty(len(pts)); own = None if label is None else np.empty(len(pts), np.int64); rmax = rad.max(); top = np.iinfo(np.int64).max
    for s in range(0, len(pts), chunk):
        p = pts[s:s + chunk]; o = np.empty(len(p)); w = np.empty(len(p), np.int64); pending = np.arange(len(p)); kk = k
        while pending.size:
            kk = min(kk, tree.n); d, i = tree.query(p[pending], k=kk, workers=threads(workers)); d, i = np.atleast_2d(d.T).T, np.atleast_2d(i.T).T; sd = d - rad[i]; c = np.min(sd, axis=1); o[pending] = c
            if label is None: sure = d[:, -1] - rmax >= c
            else: w[pending] = np.where(sd == c[:, None], label[i], top).min(1); sure = d[:, -1] - rmax > c
            sure |= kk == tree.n; pending = pending[~sure]; kk *= 2
        out[s:s + chunk] = o
        if label is not None: own[s:s + chunk] = w
    return out if label is None else (out, own)


def exact_clearance(tree, rad, pts, k=12, chunk=1 << 18, workers=-1):
    """Exact clearance at each point from a tree of atom centres and their radii, the neighbour search widened as needed."""
    rad = np.asarray(rad, float)
    if tree.n == 0 or rad.shape != (tree.n,): raise ValueError(f'exact_clearance: one radius per centre of a non-empty tree is needed ({tree.n} centres), not shape {rad.shape}')
    pts = np.atleast_2d(np.asarray(pts, float))
    if pts.ndim != 2 or pts.shape[1] != 3 or not np.all(np.isfinite(pts)): raise ValueError(f'exact_clearance: points must be finite, shape (n, 3), not {pts.shape}')
    return _nearest(tree, rad, pts, k, chunk, threads(workers, 'exact_clearance'))


def _points(points, name):
    p = np.asarray(points, float).reshape(-1, 3)
    if not np.all(np.isfinite(p)): raise ValueError(f'{name}: the points must be finite')
    return p


def _covered(positions, cell, radii, points, k, chunk, margin, workers, name, owners=False):
    pos, cell = _valid(positions, cell, name); rad0 = _radii(radii, len(pos), name); workers = threads(workers, name); blocks = points if callable(points) else (lambda: [_points(points, name)]); reach = float(widths(cell).min())
    while True:
        rep, rad = images(pos, cell, rad0, reach, name=name); tree = cKDTree(rep); lab = np.arange(len(rep)) % len(pos) if owners else None; parts = [_nearest(tree, rad, folded(p, cell, name)[0], k, chunk, workers, lab) for p in blocks()]
        c = np.concatenate([q[0] if owners else q for q in parts]) if parts else np.zeros(0)
        if not c.size or c.max() + rad0.max() + margin <= reach: return (c, np.concatenate([q[1] for q in parts]) if parts else np.zeros(0, np.int64)) if owners else c
        reach = float(c.max() + rad0.max() + margin)


def covered_clearance(positions, cell, radii, points, k=12, chunk=1 << 18, margin=0.0, workers=-1):
    """Exact clearance at points anywhere (folded into the cell), given as an array or a callable yielding blocks."""
    return _covered(positions, cell, radii, points, k, chunk, margin, workers, 'covered_clearance')


def _centres(index, n, cell):
    return ((np.stack(np.unravel_index(index, tuple(n)), -1) + 0.5) / np.asarray(n, float)) @ cell


def _grid(n, cell, chunk):
    m = int(np.prod(n))
    for s in range(0, m, chunk): yield _centres(np.arange(s, min(s + chunk, m)), n, cell)


def clearance_field(positions, cell, radii, dx=0.15, k=12, chunk=1 << 18, workers=-1):
    """Grid mode: the clearance at the centres of a fractional grid of spacing about dx (A), and the grid shape."""
    cell = _valid(positions, cell, 'clearance_field')[1]
    if not (np.isfinite(dx) and dx > 0): raise ValueError(f'clearance_field: dx must be positive and finite, not {dx!r}')
    n = np.maximum(np.round(np.linalg.norm(cell, axis=1) / dx).astype(int), 8)
    return _covered(positions, cell, radii, lambda: _grid(n, cell, chunk), k, chunk, 0.0, workers, 'clearance_field').reshape(tuple(n)), n


def sample_clearance(positions, cell, radii, n=8192, sampler='sobol', seed=0, k=12, workers=-1):
    """Pointwise mode: the clearance at n scrambled Sobol points (n a power of two) or n uniform random points."""
    cell = _valid(positions, cell, 'sample_clearance')[1]
    if isinstance(n, bool) or not float(n).is_integer() or n < 1: raise ValueError(f'sample_clearance: n must be a positive integer, not {n!r}')
    n = int(n)
    if sampler == 'sobol':
        if n & (n - 1): raise ValueError(f'sample_clearance: Sobol points come in powers of two, not n = {n}')
        f = qmc.Sobol(3, scramble=True, seed=seed).random(n)
    elif sampler == 'random': f = np.random.default_rng(seed).random((n, 3))
    else: raise ValueError(f"sample_clearance: sampler must be 'sobol' or 'random', not {sampler!r}")
    return _covered(positions, cell, radii, f @ cell, k, 1 << 18, 0.0, workers, 'sample_clearance')


def accessible_fraction(c, probe):
    """Fraction of the cell accessible to the centre of a probe sphere (clearance at least the probe radius), isolated cavities included."""
    return float(np.mean(_field(c, 'accessible_fraction') >= _probe(probe, 'accessible_fraction')))


def void_volume(c, cell, probe):
    """Volume accessible to the probe centre (A3)."""
    return float(np.mean(_field(c, 'void_volume') >= _probe(probe, 'void_volume'))) * volume(cell)


def clearance_cdf(c, r):
    """Fraction of the cell with clearance at most each r: the cumulative distribution of the clearance."""
    c = _field(c, 'clearance_cdf'); return np.searchsorted(np.sort(c.ravel()), r, side='right') / c.size


def void_surface(C, n, cell, probe):
    """Grid mode: area (A2) of the periodic isosurface C = probe by marching cubes; zero outside the field's range."""
    C = _field(C, 'void_surface'); n = tuple(int(x) for x in n); probe = _probe(probe, 'void_surface')
    if C.ndim != 3 or C.shape != n: raise ValueError(f'void_surface: the field must be three-dimensional with the grid shape {n}, not {C.shape}')
    if not C.min() < probe < C.max(): return 0.0
    verts, faces, _, _ = marching_cubes(np.pad(C, ((0, 1),) * 3, mode='wrap'), level=probe)
    x = (verts / np.asarray(n, float)) @ np.asarray(cell, float); a, b = x[faces[:, 1]] - x[faces[:, 0]], x[faces[:, 2]] - x[faces[:, 0]]; return float(0.5 * np.linalg.norm(np.cross(a, b), axis=1).sum())


def probe_sweep(C, n, cell, probes):
    """Grid mode: accessible fraction, accessible volume (A3), surface area (A2) and specific surface (m2 cm-3) at each probe radius."""
    V = volume(cell); out = []
    for p in probes: s = void_surface(C, n, cell, p); out.append(dict(probe_A=p, fraction=accessible_fraction(C, p), volume_A3=void_volume(C, cell, p), surface_A2=s, surface_m2_cm3=1e4 * s / V))
    return out


def clearance_statistics(c, thresholds=(0.25, 0.5, 0.75, 1.0, 1.25, 1.5)):
    """Largest clearance, upper percentiles, open fraction and the fraction above each threshold, for either mode."""
    c = _field(c, 'clearance_statistics').ravel(); open_ = c[c > 0]
    out = dict(largest_A=float(c.max()), p99_A=float(np.quantile(c, .99)), p95_A=float(np.quantile(c, .95)), open_fraction=float(np.mean(c > 0)), mean_open_A=float(open_.mean()) if open_.size else np.nan)
    out.update({f'above_{t:g}': float(np.mean(c >= t)) for t in thresholds}); return out


def sampling_convergence(positions, cell, radii, probe, counts=(1024, 2048, 4096, 8192, 16384), sampler='sobol', seed=0, workers=-1):
    """Accessible fraction at the probe against the number of sample points."""
    return [dict(n=n, fraction=accessible_fraction(sample_clearance(positions, cell, radii, n, sampler, seed, workers=workers), probe)) for n in counts]


def grid_convergence(positions, cell, radii, probe, spacings=(0.3, 0.2, 0.15, 0.1), workers=-1):
    """Accessible fraction and specific surface (m2 cm-3) at the probe against the grid spacing."""
    out = []
    for dx in spacings:
        C, n = clearance_field(positions, cell, radii, dx, workers=workers); out.append(dict(dx=dx, fraction=accessible_fraction(C, probe), surface_m2_cm3=1e4 * void_surface(C, n, cell, probe) / volume(cell)))
    return out
