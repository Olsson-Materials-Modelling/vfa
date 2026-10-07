"""One-electron spectra as tables (energies, occupations, species weights, participation) and what is read from them; the conventions are in docs/conventions.md."""
import numpy as np


def character(weights, species):
    """Fraction of each state's weight on one species; NaN where the total is zero."""
    tot = sum(np.asarray(w, float) for w in weights.values()); w = np.asarray(weights[species], float)
    with np.errstate(invalid='ignore', divide='ignore'): return np.where(tot > 0, w / tot, np.nan)


def occupied_count(occupation, threshold=0.5):
    """Occupied states of one spin channel."""
    return int((np.asarray(occupation) > threshold).sum())


def semicore_reference(E, n_occ, min_gap=1.5):
    """Alignment reference: the states below the single gap wider than min_gap in the lower half of the occupied spectrum, as (count, centroid eV); (0, nan) unless exactly one such gap."""
    E = np.asarray(E, float); half = n_occ // 2; big = np.flatnonzero(np.diff(E[:half]) > min_gap)
    if big.size != 1: return 0, float('nan')
    k = int(big[0]) + 1; return k, float(E[:k].mean())


def align(E, reference):
    """Energies relative to a reference level of the same calculation."""
    return np.asarray(E, float) - reference


def band_ladder(E, n_occ, ks=(0, 1, 2, 3, 5, 10)):
    """Gap from the highest occupied state to the k-th empty state, for each k."""
    E = np.asarray(E, float); return {k: dict(gap_eV=float(E[n_occ + k] - E[n_occ - 1]), empty_eV=float(E[n_occ + k])) for k in ks if n_occ + k < E.size}


def tail_spread(E, n_occ, n_tail=5):
    """Spread and mean spacing of the first n_tail empty states."""
    t = np.asarray(E, float)[n_occ:n_occ + n_tail]
    if t.size < n_tail: return dict(spread_eV=np.nan, spacing_eV=np.nan)
    return dict(spread_eV=float(t[-1] - t[0]), spacing_eV=float((t[-1] - t[0]) / (n_tail - 1)))


def bulk_character(f, n_occ, window=(20, 120)):
    """Median character of the valence and conduction bands between window[0] and window[1] states from the gap."""
    f = np.asarray(f, float); a, b = window; return dict(valence=float(np.nanmedian(f[max(0, n_occ - b):n_occ - a])), conduction=float(np.nanmedian(f[n_occ + a:n_occ + b])))


def trap_level(E, f, n_occ, valence_character, window=30):
    """The occupied state with the largest excess of the species' character over the valence band among the top window occupied states, with the runner-up margin."""
    E, f = np.asarray(E, float), np.asarray(f, float); idx = np.arange(max(0, n_occ - window), n_occ)
    if idx.size < 2: raise ValueError('trap_level: fewer than two occupied states')
    ex = f[idx] - valence_character; o = np.argsort(-ex); j, j2 = int(idx[o[0]]), int(idx[o[1]])
    return dict(index=j, energy_eV=float(E[j]), character=float(f[j]), excess=float(ex[o[0]]), margin=float(ex[o[0]] - ex[o[1]]), runner_up_gap_eV=float(abs(E[j] - E[j2])), is_homo=bool(j == n_occ - 1), levels_above=int(n_occ - 1 - j))


def delocalisation_onset(E, atoms, n_occ, n_atoms, fractions=(5, 4, 3)):
    """Per fraction: the lowest empty and the highest occupied state spreading over at least n_atoms / fraction atoms, with their distances from the band edges; NaN when none."""
    E, at = np.asarray(E, float), np.asarray(atoms, float); i = np.arange(E.size)
    if not 0 < n_occ < E.size: raise ValueError('delocalisation_onset: the spectrum needs occupied and empty states')
    out = dict(lumo_eV=float(E[n_occ]), homo_eV=float(E[n_occ - 1]), lumo_atoms=float(at[n_occ]), homo_atoms=float(at[n_occ - 1]))
    for fr in fractions:
        c = (i >= n_occ) & (at >= n_atoms / fr); v = (i < n_occ) & (at >= n_atoms / fr)
        out[f'conduction_onset_{fr}'] = dict(index=int(i[c].min()), above_lumo_eV=float(E[c].min() - E[n_occ])) if c.any() else dict(index=-1, above_lumo_eV=np.nan)
        out[f'valence_onset_{fr}'] = dict(index=int(i[v].max()), below_homo_eV=float(E[n_occ - 1] - E[v].max())) if v.any() else dict(index=-1, below_homo_eV=np.nan)
    return out


def internal_depth(E, atoms, n_occ, n_atoms, fractions=(5, 4, 3)):
    """Depth of the highest occupied state below the conduction onset of the same spectrum, per fraction."""
    E, at = np.asarray(E, float), np.asarray(atoms, float); i = np.arange(E.size)
    if not 0 < n_occ < E.size: raise ValueError('internal_depth: the spectrum needs occupied and empty states')
    out = dict(state_eV=float(E[n_occ - 1]), state_atoms=float(at[n_occ - 1]))
    for fr in fractions:
        c = (i >= n_occ) & (at >= n_atoms / fr); out[f'depth_{fr}_eV'] = float(E[c].min() - E[n_occ - 1]) if c.any() else np.nan
    return out


def participation_from_weights(p):
    """Effective atoms of a state: 1 / sum p_i^2 of its normalised per-atom weights."""
    p = np.asarray(p, float); p = p / p.sum(); return float(1 / (p ** 2).sum())
