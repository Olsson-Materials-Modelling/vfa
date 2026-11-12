"""Total scattering of a cell: Faber-Ziman partials by transform or reciprocal sum, weighted totals, real-space functions, first sharp diffraction peak."""
import numpy as np
from .cell import _valid, volume, fractional
from .rdf import _r_max, partial_rdf


def concentrations(symbols):
    """Atomic fraction of each species, keyed in sorted order."""
    sym, n = np.unique(np.asarray(symbols).astype(str), return_counts=True); return {str(s): float(k / n.sum()) for s, k in zip(sym, n)}


def _pairs(species):
    return [(a, b) for i, a in enumerate(species) for b in species[i:]]


def _grid(x, window, name):
    x = np.asarray(x, float)
    if x.ndim != 1 or x.size < 2 or np.any(np.diff(x) <= 0) or x[0] < 0: raise ValueError(f'{name}: the grid must be one-dimensional, non-negative and strictly increasing with at least two points')
    edges = np.concatenate([[max(0.0, x[0] - 0.5 * (x[1] - x[0]))], 0.5 * (x[1:] + x[:-1]), [x[-1] + 0.5 * (x[-1] - x[-2])]])
    if window is None: m = np.ones_like(x)
    elif window == 'lorch': m = np.sinc(x / edges[-1])
    else: raise ValueError(f"{name}: window must be None or 'lorch', not {window!r}")
    return x, np.diff(edges), m


def sine_transform(r, h, q, rho_A3, window=None):
    """rho * integral of 4 pi r^2 h(r) sin(Qr)/(Qr) M(r) dr over the bin centres, M the Lorch window or 1."""
    r, w, m = _grid(r, window, 'sine_transform'); h = np.asarray(h, float)
    if h.shape != r.shape: raise ValueError(f'sine_transform: h has shape {h.shape}, the r grid {r.shape}')
    return rho_A3 * (np.sinc(np.outer(np.asarray(q, float), r) / np.pi) @ (4 * np.pi * r ** 2 * h * m * w))


def inverse_sine_transform(q, H, r, rho_A3, window=None):
    """1/(2 pi^2 rho) * integral of Q^2 H(Q) sin(Qr)/(Qr) M(Q) dQ: the inverse of sine_transform on the Q grid."""
    q, w, m = _grid(q, window, 'inverse_sine_transform'); H = np.asarray(H, float)
    if H.shape != q.shape: raise ValueError(f'inverse_sine_transform: H has shape {H.shape}, the Q grid {q.shape}')
    return (np.sinc(np.outer(np.asarray(r, float), q) / np.pi) @ (q ** 2 * H * m * w)) / (2 * np.pi ** 2 * rho_A3)


def pair_correlations(symbols, positions, cell, r_max=None, n_bins=2000):
    """Bin centres and partial_rdf of every species pair, keyed by sorted pairs, up to r_max (default and limit as partial_rdf)."""
    _valid(positions, cell, 'pair_correlations'); sym = np.asarray(symbols).astype(str); out = {}; r = None
    for p in _pairs(list(concentrations(sym))): r, out[p] = partial_rdf(sym, positions, cell, p, r_max, n_bins)
    return r, out


def transform_partials(symbols, positions, cell, q, r_max=None, n_bins=2000, window=None):
    """Faber-Ziman S_ab(Q) by the sine transform of g_ab - 1 up to r_max, keyed by sorted species pairs."""
    _valid(positions, cell, 'transform_partials'); sym = np.asarray(symbols).astype(str); c = concentrations(sym); n = {s: int((sym == s).sum()) for s in c}; rho = len(sym) / volume(cell); r, g = pair_correlations(sym, positions, cell, r_max, n_bins)
    return {(a, b): 1 + sine_transform(r, g[(a, b)] - 1, q, rho * (n[a] - 1) / n[a] if a == b else rho, window) for a, b in g}


def _reciprocal(cell, q_max):
    B = 2 * np.pi * np.linalg.inv(cell).T; H = np.floor(q_max * np.linalg.norm(cell, axis=1) / (2 * np.pi) + 1e-9).astype(int); h2, h3 = np.meshgrid(np.arange(-H[1], H[1] + 1), np.arange(-H[2], H[2] + 1), indexing='ij'); h2, h3 = h2.ravel(), h3.ravel()
    for h1 in range(H[0] + 1):
        hk = np.stack([np.full(h2.size, h1), h2, h3], 1)
        if h1 == 0: hk = hk[(h2 > 0) | ((h2 == 0) & (h3 > 0))]
        Q = np.linalg.norm(hk @ B, axis=1); keep = Q <= q_max * (1 + 1e-12); yield hk[keep], Q[keep]


def _amplitudes(T, h, bounds):
    E = T[0][:, h[:, 0]] * T[1][:, h[:, 1]] * T[2][:, h[:, 2]]; return [E[bounds[a]:bounds[a + 1]].sum(0) for a in range(len(bounds) - 1)]


def _direct(symbols, positions, cell, q_max, dq, chunk, name):
    pos, cell = _valid(positions, cell, name); sym = np.asarray(symbols).astype(str); N = len(pos)
    if N == 0 or sym.shape != (N,): raise ValueError(f'{name}: one symbol per atom of at least one is needed ({N} atoms), not shape {sym.shape}')
    if not (q_max > 0 and dq > 0 and np.isfinite(q_max) and np.isfinite(dq)): raise ValueError(f'{name}: q_max ({q_max:g}) and dq ({dq:g}) must be positive and finite')
    c = concentrations(sym); species = list(c); code = np.unique(sym, return_inverse=True)[1].ravel(); order = np.argsort(code, kind='stable'); bounds = np.r_[0, np.cumsum(np.bincount(code))]
    f = fractional(pos[order], cell); pairs = _pairs(species); nb = int(np.floor(q_max / dq)) + 1
    H = np.floor(q_max * np.linalg.norm(cell, axis=1) / (2 * np.pi) + 1e-9).astype(int); T = [np.exp(2j * np.pi * np.outer(f[:, k], np.arange(-H[k], H[k] + 1))) for k in range(3)]
    acc = {p: np.zeros(nb) for p in pairs}; cnt = np.zeros(nb, np.int64); qs = np.zeros(nb); step = max(1, int(chunk) // max(N, 1))
    for hk, Q in _reciprocal(cell, q_max):
        for s in range(0, len(Q), step):
            h = hk[s:s + step] + H; k = np.minimum((Q[s:s + step] / dq).astype(int), nb - 1); A = _amplitudes(T, h, bounds)
            cnt += np.bincount(k, minlength=nb); qs += np.bincount(k, Q[s:s + step], minlength=nb)
            for (a, b) in pairs: i, j = species.index(a), species.index(b); acc[(a, b)] += np.bincount(k, (A[i] * A[j].conj()).real, minlength=nb)
    ok = cnt > 0; S = {}
    for a, b in pairs:
        mean = acc[(a, b)][ok] / (cnt[ok] * N); S[(a, b)] = 1 + (mean - c[a]) / c[a] ** 2 if a == b else 1 + mean / (c[a] * c[b])
    return np.flatnonzero(ok), qs[ok] / cnt[ok], S, 2 * cnt[ok]


def direct_partials(symbols, positions, cell, q_max, dq=0.05, chunk=1 << 22):
    """Mean |G| of each non-empty shell of width dq, the Faber-Ziman partials over 0 < |G| <= q_max, vectors per shell."""
    _, q, S, m = _direct(symbols, positions, cell, q_max, dq, chunk, 'direct_partials'); return q, S, m


def _same_species(cells, name):
    sets = [tuple(concentrations(s)) for s, _, _ in cells]
    if not cells: raise ValueError(f'{name}: no cells')
    for _, p, cl in cells: _valid(p, cl, name)
    if len(set(sets)) > 1: raise ValueError(f'{name}: the cells do not share one set of species: {sorted(set(sets))}')
    return {s: float(np.mean([concentrations(sym)[s] for sym, _, _ in cells])) for s in sets[0]}


def ensemble_direct(cells, q_max, dq=0.05, chunk=1 << 22):
    """Mean over cells of direct_partials on the shells present in every cell."""
    cells = list(cells); c = _same_species(cells, 'ensemble_direct'); runs = [_direct(s, p, cl, q_max, dq, chunk, 'ensemble_direct') for s, p, cl in cells]; common = runs[0][0]
    for k, _, _, _ in runs[1:]: common = np.intersect1d(common, k)
    pick = [np.searchsorted(k, common) for k, _, _, _ in runs]; q = np.mean([r[1][i] for r, i in zip(runs, pick)], axis=0)
    return q, {p: np.mean([r[2][p][i] for r, i in zip(runs, pick)], axis=0) for p in runs[0][2]}, c


def ensemble_transform(cells, q, r_max=None, n_bins=2000, window=None):
    """Mean over cells of transform_partials, and the mean concentrations."""
    cells = list(cells); c = _same_species(cells, 'ensemble_transform'); r_max = min(_r_max(cl, r_max, 'ensemble_transform') for _, _, cl in cells)
    runs = [transform_partials(s, p, cl, q, r_max, n_bins, window) for s, p, cl in cells]; return {p: np.mean([r[p] for r in runs], axis=0) for p in runs[0]}, c


def _weights(weights, species, q, name):
    out = {}
    for s in species:
        if s not in weights: raise ValueError(f'{name}: no weight for species {s}')
        w = weights[s]; w = np.asarray(w(q), float) if callable(w) else np.asarray(w, float)
        if w.ndim and w.shape != q.shape: raise ValueError(f'{name}: the weight of {s} has shape {w.shape}, the Q grid {q.shape}')
        out[s] = np.broadcast_to(w, q.shape)
    return out


def total_sq(q, partials, conc, weights):
    """F(Q) = sum c_a c_b w_a w_b [S_ab - 1] and S(Q) = 1 + F / (sum_a c_a w_a)^2 from the Faber-Ziman partials; w_a a number, array or callable."""
    q = np.asarray(q, float); w = _weights(weights, list(conc), q, 'total_sq'); F = np.zeros_like(q); pairs = _pairs(sorted(conc)); extra = sorted(set(partials) - set(pairs), key=str); missing = [p for p in pairs if p not in partials]
    if missing or extra: raise ValueError(f'total_sq: partials must hold each species pair of conc once, keyed in sorted order {pairs}: missing {missing}, not expected {extra}')
    for a, b in pairs: F = F + (1 if a == b else 2) * conc[a] * conc[b] * w[a] * w[b] * (np.asarray(partials[(a, b)], float) - 1)
    mean = sum(conc[s] * w[s] for s in conc)
    if np.any(mean == 0): raise ValueError('total_sq: the mean weight sum_a c_a w_a is zero on the grid (null scattering); S(Q) is undefined, use F(Q)')
    return F, 1 + F / mean ** 2


def total_gr(symbols, positions, cell, b, r_max=None, n_bins=2000):
    """Bin centres and the neutron G(r) = sum_ab c_a c_b b_a b_b [g_ab - 1] of the cell, like pairs scaled by (N_a - 1)/N_a."""
    _valid(positions, cell, 'total_gr'); sym = np.asarray(symbols).astype(str); c = concentrations(sym); n = {s: int((sym == s).sum()) for s in c}
    if any(s not in b or callable(b[s]) or np.ndim(b[s]) for s in c): raise ValueError('total_gr: b needs one number per species (Q-independent scattering lengths); with X-ray form factors pass S(Q) - 1 from total_sq to real_space with mean_b = 1')
    r, g = pair_correlations(sym, positions, cell, r_max, n_bins)
    return r, sum((1 if a == e else 2) * c[a] * c[e] * float(b[a]) * float(b[e]) * ((n[a] - 1) / n[a] if a == e else 1.0) * (g[(a, e)] - 1) for a, e in g)


def real_space(q, F, r, rho_A3, mean_b, window=None):
    """Dict of G(r), the inverse sine transform of F(Q), D(r) = 4 pi rho r G and T(r) = D + 4 pi rho r mean_b^2."""
    if not (np.isfinite(rho_A3) and rho_A3 > 0): raise ValueError(f'real_space: the number density must be positive and finite, not {rho_A3}')
    r = np.asarray(r, float); G = inverse_sine_transform(q, F, r, rho_A3, window); D = 4 * np.pi * rho_A3 * r * G; return dict(G=G, D=D, T=D + 4 * np.pi * rho_A3 * r * float(mean_b) ** 2)


def fsdp(q, S, window):
    """First sharp diffraction peak in the Q window: position, height, height above the baseline and full width at half that height."""
    q = np.asarray(q, float); S = np.asarray(S, float)
    if q.ndim != 1 or q.shape != S.shape or q.size < 2 or np.any(np.diff(q) <= 0) or not np.all(np.isfinite(q)) or not np.all(np.isfinite(S)): raise ValueError(f'fsdp: q and S must be finite one-dimensional arrays of one shape, q strictly increasing (shapes {q.shape}, {S.shape})')
    idx = np.flatnonzero((q >= window[0]) & (q <= window[1]))
    if idx.size < 5: raise ValueError(f'fsdp: {idx.size} points inside the window {tuple(window)}; at least 5 are needed')
    p = int(idx[np.argmax(S[idx])])
    if p == idx[0] or p == idx[-1]: raise ValueError(f'fsdp: no peak resolved in {tuple(window)}: the maximum is at the window edge (Q = {q[p]:.4f})')
    lo = idx[idx < p]; hi = idx[idx > p]; L = int(lo[np.argmin(S[lo])]); R = int(hi[np.argmin(S[hi])]); base = lambda x: S[L] + (S[R] - S[L]) * (x - q[L]) / (q[R] - q[L]); h = S[p] - base(q[p]); y = S - base(q)
    i = next(k for k in range(p, L, -1) if y[k - 1] <= 0.5 * h); j = next(k for k in range(p, R) if y[k + 1] <= 0.5 * h)
    if i == p and j == p: raise ValueError(f'fsdp: no peak resolved in {tuple(window)}: the maximum at Q = {q[p]:.4f} is a single point above half height')
    ql = q[i - 1] + (0.5 * h - y[i - 1]) * (q[i] - q[i - 1]) / (y[i] - y[i - 1]); qr = q[j] + (0.5 * h - y[j]) * (q[j + 1] - q[j]) / (y[j + 1] - y[j])
    (x0, x1, x2), (z0, z1, z2) = q[p - 1:p + 2], S[p - 1:p + 2]; den = (x1 - x0) * (z1 - z2) - (x1 - x2) * (z1 - z0); pos = x1 - 0.5 * ((x1 - x0) ** 2 * (z1 - z2) - (x1 - x2) ** 2 * (z1 - z0)) / den if den > 0 else x1
    top = p + int(np.argmin(S[p:] == S[p])) if np.any(S[p:] != S[p]) else len(S)
    if top - p > 1: pos = 0.5 * (q[p] + q[top - 1])
    return dict(position_invA=float(pos), height=float(S[p]), height_above_baseline=float(h), fwhm_invA=float(qr - ql), half_height_invA=(float(ql), float(qr)), baseline_invA=(float(q[L]), float(q[R])))
