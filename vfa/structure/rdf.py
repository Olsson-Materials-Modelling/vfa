"""Partial radial distribution functions up to half the shortest lattice vector, and the first minimum after the first peak."""
import numpy as np
from scipy.signal import find_peaks, peak_prominences
from .cell import _valid, shell, shortest_vector, wrap


def _r_max(cell, r_max, name):
    half = 0.5 * float(np.linalg.norm(shortest_vector(_valid(np.zeros((0, 3)), cell, name)[1])))
    if r_max is None: return half
    if not (np.isfinite(r_max) and r_max > 0): raise ValueError(f'{name}: r_max must be positive and finite, not {r_max!r}')
    if r_max > half + 1e-9: raise ValueError(f'{name}: r_max {r_max:g} A exceeds half the shortest lattice vector ({half:.3f} A): beyond it two images of one atom fall inside r')
    return float(r_max)


def _bins(n_bins, name):
    if isinstance(n_bins, bool) or not float(n_bins).is_integer() or n_bins < 1: raise ValueError(f'{name}: n_bins must be a positive integer, not {n_bins!r}')
    return int(n_bins)


def partial_rdf(symbols, positions, cell, pair, r_max=None, n_bins=1000):
    """Bin centres and g(r) of an ordered species pair up to r_max (default and limit: half the shortest lattice vector)."""
    sym = np.asarray(symbols).astype(str); pos = wrap(positions, cell, 'partial_rdf'); cell = np.asarray(cell, float); n_bins = _bins(n_bins, 'partial_rdf'); r_max = _r_max(cell, r_max, 'partial_rdf'); a, b = str(pair[0]), str(pair[1])
    if sym.shape != (len(pos),): raise ValueError(f'partial_rdf: one symbol per atom is needed ({len(pos)} atoms), not shape {sym.shape}')
    ia, ib = np.flatnonzero(sym == a), np.flatnonzero(sym == b); edges = np.linspace(0, r_max, n_bins + 1); r = 0.5 * (edges[1:] + edges[:-1]); h = np.zeros(n_bins, np.int64)
    if ia.size == 0 or ib.size == 0: raise ValueError(f"partial_rdf: species {a if ia.size == 0 else b} is not in the cell ({', '.join(sorted(set(sym.tolist())))})")
    for shift in shell(cell, r_max):
        d = np.linalg.norm((pos[ib] + shift @ cell)[None] - pos[ia][:, None], axis=2)
        if a == b and not shift.any(): np.fill_diagonal(d, np.inf)
        h += np.histogram(d.ravel(), bins=edges)[0]
    V = abs(np.linalg.det(cell)); rho = (ib.size - 1) / V if a == b else ib.size / V; dv = 4 * np.pi * r ** 2 * (edges[1] - edges[0])
    with np.errstate(divide='ignore', invalid='ignore'): g = h / (ia.size * dv * rho)
    g[~np.isfinite(g)] = 0.0; return r, g


def ensemble_rdf(cells, pair, r_max=None, n_bins=1000):
    """Cell-weighted mean partial RDF over (symbols, positions, cell) triples, up to r_max (default: the smallest limit of any cell)."""
    cells = list(cells)
    if not cells: raise ValueError('ensemble_rdf: no cells')
    for _, pos, c in cells: _valid(pos, c, 'ensemble_rdf')
    n_bins = _bins(n_bins, 'ensemble_rdf'); r_max = min(_r_max(c, r_max, 'ensemble_rdf') for _, _, c in cells); acc = np.zeros(n_bins); r = None
    for sym, pos, cell in cells: r, g = partial_rdf(sym, pos, cell, pair, r_max, n_bins); acc += g
    return r, acc / len(cells)


def running_coordination(symbols, positions, cell, pair, r_max=None, n_bins=1000):
    """Upper bin edges r (A) and the mean number of atoms of the second species within r of an atom of the first."""
    pos, cell = _valid(positions, cell, 'running_coordination'); n_bins = _bins(n_bins, 'running_coordination'); r_max = _r_max(cell, r_max, 'running_coordination'); sym = np.asarray(symbols).astype(str); ib = np.flatnonzero(sym == str(pair[1]))
    r, g = partial_rdf(symbols, pos, cell, pair, r_max, n_bins); edges = np.linspace(0, r_max, n_bins + 1); rho = (ib.size - 1) / abs(np.linalg.det(cell)) if str(pair[0]) == str(pair[1]) else ib.size / abs(np.linalg.det(cell))
    return edges[1:], np.cumsum(g * 4 * np.pi * r ** 2 * (edges[1] - edges[0]) * rho)


def _smooth(r, g, width):
    h = 0.5 * width * (1 + 1e-9); lo = np.searchsorted(r, r - h); hi = np.searchsorted(r, r + h, 'right'); return np.add.reduceat(np.r_[g, 0.0], np.ravel(np.c_[lo, hi]))[::2] / (hi - lo)


def first_minimum(r, g, search=(1.0, 4.5), smoothing_A=0.1, prominence=0.25, basin_tolerance=1.10, basin_floor=0.01):
    """First prominent peak of the smoothed g, the next peak, the basin between them with its minimum and middle; resolved when both peaks lie in search."""
    r = np.asarray(r, float); g = np.asarray(g, float)
    if r.ndim != 1 or r.shape != g.shape or r.size < 2 or np.any(np.diff(r) <= 0) or not np.all(np.isfinite(g)): raise ValueError(f'first_minimum: r and g must be finite one-dimensional arrays of one shape, r strictly increasing (shapes {r.shape}, {g.shape})')
    if not (np.isfinite(smoothing_A) and smoothing_A >= 0 and np.isfinite(prominence) and prominence >= 0 and basin_tolerance >= 1 and basin_floor >= 0): raise ValueError(f'first_minimum: smoothing_A ({smoothing_A!r}) and prominence ({prominence!r}) must be finite and non-negative, basin_tolerance ({basin_tolerance!r}) at least 1 and basin_floor ({basin_floor!r}) non-negative')
    win = np.flatnonzero((r >= search[0]) & (r <= search[1]))
    if win.size == 0: raise ValueError(f'first_minimum: no bins inside the search window {search}')
    gs = _smooth(r, g, float(smoothing_A)); k = find_peaks(gs)[0]; k = k[peak_prominences(gs, k)[0] >= prominence * np.maximum(gs[k], 1.0)]; first = k[(k >= win[0]) & (k <= win[-1])]
    out = dict(peak_A=np.nan, peak_g=np.nan, next_peak_A=np.nan, minimum_A=np.nan, minimum_g=np.nan, basin_A=(np.nan, np.nan), basin_middle_A=np.nan, basin_max_g=np.nan, resolved=False)
    if first.size == 0: return out
    p = int(first[0]); later = k[k > p]; resolved = bool(later.size and later[0] <= win[-1]); span = np.arange(p + 1, int(later[0]) if resolved else int(win[-1]) + 1); out.update(peak_A=float(r[p]), peak_g=float(gs[p]), next_peak_A=float(r[later[0]]) if later.size else np.nan)
    if span.size == 0: return out
    m = int(span[np.argmin(gs[span])]); inside = span[gs[span] <= gs[m] + (basin_tolerance - 1) * abs(gs[m]) + basin_floor]; lo, hi = int(inside[0]), int(inside[-1])
    out.update(minimum_A=float(r[m]), minimum_g=float(gs[m]), basin_A=(float(r[lo]), float(r[hi])), basin_middle_A=0.5 * (float(r[lo]) + float(r[hi])), basin_max_g=float(gs[lo:hi + 1].max()), resolved=resolved); return out
