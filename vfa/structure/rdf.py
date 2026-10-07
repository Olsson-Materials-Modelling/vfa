"""Partial radial distribution functions with explicit image expansion, and the first minimum after the first peak."""
import numpy as np


def partial_rdf(symbols, positions, cell, pair, r_max=10.0, n_bins=1000):
    """g(r) of ordered pairs of two species, normalised by the density of the second; returns bin centres and g."""
    sym = np.asarray(symbols); pos = np.asarray(positions, float); cell = np.asarray(cell, float); a, b = pair; ia, ib = np.flatnonzero(sym == a), np.flatnonzero(sym == b)
    edges = np.linspace(0, r_max, n_bins + 1); r = 0.5 * (edges[1:] + edges[:-1]); h = np.zeros(n_bins, np.int64)
    if ia.size == 0 or ib.size == 0: return r, np.zeros(n_bins)
    reps = np.ceil(r_max * np.linalg.norm(np.linalg.inv(cell), axis=0)).astype(int)
    for shift in np.array([[x, y, z] for x in range(-reps[0], reps[0] + 1) for y in range(-reps[1], reps[1] + 1) for z in range(-reps[2], reps[2] + 1)]):
        d = np.linalg.norm((pos[ib] + shift @ cell)[None] - pos[ia][:, None], axis=2)
        if a == b and not shift.any(): np.fill_diagonal(d, np.inf)
        h += np.histogram(d.ravel(), bins=edges)[0]
    V = abs(np.linalg.det(cell)); rho = (ib.size - 1) / V if a == b else ib.size / V; shell = 4 * np.pi * r ** 2 * (edges[1] - edges[0])
    with np.errstate(divide='ignore', invalid='ignore'): g = h / (ia.size * shell * rho)
    g[~np.isfinite(g)] = 0.0; return r, g


def ensemble_rdf(cells, pair, r_max=10.0, n_bins=1000):
    """Cell-weighted mean of the partial RDF over (symbols, positions, cell) triples."""
    acc = np.zeros(n_bins); r = None
    for sym, pos, cell in cells: r, g = partial_rdf(sym, pos, cell, pair, r_max, n_bins); acc += g
    return r, acc / max(len(cells), 1)


def running_coordination(symbols, positions, cell, pair, r_max=10.0, n_bins=1000):
    """Mean number of atoms of the second species within r of an atom of the first."""
    sym = np.asarray(symbols); ia, ib = np.flatnonzero(sym == pair[0]), np.flatnonzero(sym == pair[1]); r, g = partial_rdf(symbols, positions, cell, pair, r_max, n_bins)
    V = abs(np.linalg.det(np.asarray(cell, float))); rho = (ib.size - 1) / V if pair[0] == pair[1] else ib.size / V; dr = r[1] - r[0]; return r, np.cumsum(g * 4 * np.pi * r ** 2 * dr * rho)


def first_minimum(r, g, search=(1.0, 4.5), smoothing_bins=9, basin_tolerance=1.10):
    """First peak (maximum of the smoothed curve in the window), the minimum after it, and the basin where g stays within the tolerance of that minimum."""
    gs = np.convolve(g, np.ones(smoothing_bins) / smoothing_bins, mode='same') if smoothing_bins > 1 else g.copy(); idx = np.flatnonzero((r >= search[0]) & (r <= search[1]))
    if idx.size == 0: raise ValueError(f'first_minimum: no bins inside the search window {search}')
    peak = idx[np.argmax(gs[idx])]; after = idx[idx > peak]
    if after.size == 0: return dict(peak_A=float(r[peak]), minimum_A=np.nan, resolved=False)
    m = after[np.argmin(gs[after])]; lo = hi = int(m)
    while lo - 1 >= idx[0] and gs[lo - 1] <= gs[m] * basin_tolerance: lo -= 1
    while hi + 1 <= idx[-1] and gs[hi + 1] <= gs[m] * basin_tolerance: hi += 1
    return dict(peak_A=float(r[peak]), peak_g=float(gs[peak]), minimum_A=float(r[m]), minimum_g=float(gs[m]), basin_A=(float(r[lo]), float(r[hi])), resolved=bool(m < idx[-1] - smoothing_bins))
