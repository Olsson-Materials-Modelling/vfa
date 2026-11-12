"""Readings of one-electron spectra given as tables of energies, occupations, species weights and participation."""
import numpy as np


def _ascending(E, name):
    E = np.asarray(E, float)
    if E.ndim != 1 or not np.all(np.isfinite(E)) or np.any(np.diff(E) < 0): raise ValueError(f'{name}: the energies must be finite and in ascending order')
    return E


def character(weights, species):
    """Fraction of each state's weight on one species; NaN where the total is zero."""
    tot = sum(np.asarray(w, float) for w in weights.values()); w = np.asarray(weights[species], float)
    with np.errstate(invalid='ignore', divide='ignore'): return np.where(tot > 0, w / tot, np.nan)


def occupied_count(occupation, threshold=0.5):
    """Occupied states of one spin channel."""
    return int((np.asarray(occupation) > threshold).sum())


def semicore_reference(E, n_occ, min_gap=1.5):
    """Count and centroid (eV) of the deep band below the single gap wider than min_gap in the lower half of the occupied states; (0, nan) otherwise."""
    E = _ascending(E, 'semicore_reference'); half = n_occ // 2; big = np.flatnonzero(np.diff(E[:half]) > min_gap)
    if big.size != 1: return 0, float('nan')
    k = int(big[0]) + 1; return k, float(E[:k].mean())


def align(E, reference):
    """Energies relative to a reference level of the same calculation."""
    return np.asarray(E, float) - reference


def band_ladder(E, n_occ, ks=(0, 1, 2, 3, 5, 10)):
    """Gap from the highest occupied state to the k-th empty state, for each k."""
    E = _ascending(E, 'band_ladder')
    if not 0 < n_occ < E.size: raise ValueError(f'band_ladder: n_occ {n_occ} leaves no occupied or no empty state among {E.size}')
    return {k: dict(gap_eV=float(E[n_occ + k] - E[n_occ - 1]), empty_eV=float(E[n_occ + k])) for k in ks if n_occ + k < E.size}


def tail_spread(E, n_occ, n_tail=5):
    """Spread and mean spacing of the first n_tail empty states; NaN when fewer are listed."""
    E = _ascending(E, 'tail_spread')
    if not 0 < n_occ <= E.size: raise ValueError(f'tail_spread: n_occ {n_occ} outside 1..{E.size}')
    t = E[n_occ:n_occ + n_tail]
    if t.size < n_tail: return dict(spread_eV=np.nan, spacing_eV=np.nan)
    return dict(spread_eV=float(t[-1] - t[0]), spacing_eV=float((t[-1] - t[0]) / (n_tail - 1)))


def bulk_character(f, n_occ, window=(20, 120)):
    """Median character of the valence and conduction states between window[0] and window[1] from the gap."""
    f = np.asarray(f, float); a, b = (int(x) for x in window)
    if not 0 <= a < b: raise ValueError(f'bulk_character: window {window} must satisfy 0 <= start < stop')
    if not b <= n_occ <= f.size - b: raise ValueError(f'bulk_character: window {window} needs {b} occupied and {b} empty states, the spectrum has {n_occ} and {f.size - n_occ}')
    v, c = f[n_occ - b:n_occ - a], f[n_occ + a:n_occ + b]
    if np.isnan(v).all() or np.isnan(c).all(): raise ValueError('bulk_character: no state of the window has a character')
    return dict(valence=float(np.nanmedian(v)), conduction=float(np.nanmedian(c)))


def trap_level(E, f, n_occ, valence_character, window=30):
    """The occupied state with the largest excess of the species' character over the valence band, with the runner-up margin."""
    E, f = _ascending(E, 'trap_level'), np.asarray(f, float); idx = np.arange(max(0, n_occ - window), n_occ)
    if n_occ > E.size: raise ValueError(f'trap_level: n_occ {n_occ} exceeds the {E.size} states')
    if idx.size < 2: raise ValueError('trap_level: fewer than two occupied states')
    ex = f[idx] - valence_character; o = np.argsort(-ex, kind='stable'); j, j2 = int(idx[o[0]]), int(idx[o[1]])
    return dict(index=j, energy_eV=float(E[j]), character=float(f[j]), excess=float(ex[o[0]]), margin=float(ex[o[0]] - ex[o[1]]), runner_up_gap_eV=float(abs(E[j] - E[j2])), is_homo=bool(j == n_occ - 1), levels_above=int(n_occ - 1 - j))


def delocalisation_onset(E, atoms, n_occ, n_atoms, fractions=(5, 4, 3)):
    """Per fraction, the empty and occupied states nearest the gap spreading over at least n_atoms / fraction atoms."""
    E, at = _ascending(E, 'delocalisation_onset'), np.asarray(atoms, float); i = np.arange(E.size)
    if not 0 < n_occ < E.size: raise ValueError('delocalisation_onset: the spectrum needs occupied and empty states')
    out = dict(lumo_eV=float(E[n_occ]), homo_eV=float(E[n_occ - 1]), lumo_atoms=float(at[n_occ]), homo_atoms=float(at[n_occ - 1]))
    for fr in fractions:
        c = (i >= n_occ) & (at >= n_atoms / fr); v = (i < n_occ) & (at >= n_atoms / fr)
        out[f'conduction_onset_{fr}'] = dict(index=int(i[c].min()), above_lumo_eV=float(E[c].min() - E[n_occ])) if c.any() else dict(index=-1, above_lumo_eV=np.nan)
        out[f'valence_onset_{fr}'] = dict(index=int(i[v].max()), below_homo_eV=float(E[n_occ - 1] - E[v].max())) if v.any() else dict(index=-1, below_homo_eV=np.nan)
    return out


def internal_depth(E, atoms, n_occ, n_atoms, fractions=(5, 4, 3)):
    """Depth of the highest occupied state below the conduction onset of the same spectrum, per fraction."""
    E, at = _ascending(E, 'internal_depth'), np.asarray(atoms, float); i = np.arange(E.size)
    if not 0 < n_occ < E.size: raise ValueError('internal_depth: the spectrum needs occupied and empty states')
    out = dict(state_eV=float(E[n_occ - 1]), state_atoms=float(at[n_occ - 1]))
    for fr in fractions:
        c = (i >= n_occ) & (at >= n_atoms / fr); out[f'depth_{fr}_eV'] = float(E[c].min() - E[n_occ - 1]) if c.any() else np.nan
    return out
