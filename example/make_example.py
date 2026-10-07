#!/usr/bin/env python3
"""Synthetic example data: network glasses (four-fold A bridged by two-fold B, 204 atoms) at six densities, five cells each, one trapping event per cell; deterministic, about twenty minutes.
    python3 example/make_example.py [out_dir]     or     python3 example/make_example.py electronic [out_dir]   (the electronic data only)"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vfa.structure.cell import minimum_image, separations, distances
from vfa.io.extxyz import write_extxyz
from vfa.voids.radii import Radii

SEED = 11; N_A, N_B = 68, 136; R0 = 1.62; DENSITIES = (0.052, 0.056, 0.060, 0.064, 0.068, 0.072); PER_DENSITY = 5; DEFECTS = 2    # atoms per A3; three-fold A atoms per cell
COS0 = {'A': -1 / 3, 'B': -0.82}; RADII = Radii({'A': 1.1, 'B': 0.7}, 'covalent'); RADII_IONIC = Radii({'A': 0.3, 'B': 1.4}, 'ionic')        # the example's two conventions: a bonding radius per species, and a small cation with a large anion
ARGS = [a for a in sys.argv[1:] if a != 'electronic']; OUT = Path(ARGS[0]) if ARGS else Path(__file__).resolve().parent / 'data'


def bonds_of(pos, cell, role, want):
    """Bond list (closest A-B pairs first, while both atoms have free valence) and the angle triplets at every atom."""
    n = len(pos); d = distances(pos, cell); ii, jj = np.nonzero(np.triu(role[:, None] != role[None, :])); order = np.argsort(d[ii, jj]); free = want.copy(); b = []
    partners = [set() for _ in range(n)]                                                                                     # atoms bonded so far, to refuse a bond that would close a two-membered ring
    for k in order:
        i, j = ii[k], jj[k]; a, c = (i, j) if role[i] else (j, i)
        if free[i] > 0 and free[j] > 0 and d[i, j] < 3.6 and not partners[c] & {x for y in partners[a] for x in partners[y]} - {a}:
            b.append((i, j)); free[i] -= 1; free[j] -= 1; partners[a].add(c); partners[c].add(a)
    for i in set(range(n)) - set(np.array(b).ravel()):
        opp = np.flatnonzero(role != role[i]); j = int(opp[np.argmin(d[i, opp])]); b.append((min(i, j), max(i, j)))
    b = np.array(sorted(b)); tri = []
    for i in range(n):
        nb = sorted(set(b[b[:, 0] == i, 1]) | set(b[b[:, 1] == i, 0])); tri += [(i, a, c) for x, a in enumerate(nb) for c in nb[x + 1:]]
    return b, np.array(tri)


def energy(x, cell, role, b, tri, cos0, ka, kb=12.0, kr=4.0):
    """Model energy and gradient: bond springs, non-bonded repulsion, angle springs (stiff at A, soft at the bridging B)."""
    n = len(role); p = x.reshape(n, 3); v = separations(p, cell); d = np.linalg.norm(v, axis=2) + np.eye(n); g = np.zeros((n, 3)); e = 0.0
    r = d[b[:, 0], b[:, 1]]; u = v[b[:, 0], b[:, 1]] / r[:, None]; f = 2 * kb * (r - R0); e += kb * ((r - R0) ** 2).sum()
    np.add.at(g, b[:, 0], -f[:, None] * u); np.add.at(g, b[:, 1], f[:, None] * u)
    rep = np.where(role[:, None] & role[None, :], 2.9, np.where(role[:, None] | role[None, :], 2.4, 2.5)); close = (d < rep) & ~np.eye(n, dtype=bool); close[b[:, 0], b[:, 1]] = close[b[:, 1], b[:, 0]] = False
    ii, jj = np.nonzero(np.triu(close)); r = d[ii, jj]; u = v[ii, jj] / r[:, None]; f = -2 * kr * (rep[ii, jj] - r); e += kr * ((rep[ii, jj] - r) ** 2).sum()
    np.add.at(g, ii, -f[:, None] * u); np.add.at(g, jj, f[:, None] * u)
    i, a, c = tri.T; va, vc = v[i, a], v[i, c]; ra, rc = d[i, a], d[i, c]; cs = (va * vc).sum(1) / (ra * rc); dif = cs - cos0[i]; e += (ka[i] * dif ** 2).sum()
    dca = vc / (ra * rc)[:, None] - (cs / ra ** 2)[:, None] * va; dcc = va / (ra * rc)[:, None] - (cs / rc ** 2)[:, None] * vc; f = (2 * ka[i] * dif)[:, None]
    np.add.at(g, a, f * dca); np.add.at(g, c, f * dcc); np.add.at(g, i, -f * (dca + dcc))
    return e, g.ravel()


def network(rng, density, inflate=1.35):
    """One relaxed network grown in an inflated box and compressed to the number density; DEFECTS A atoms stay three-fold."""
    n = N_A + N_B; sym = np.array(['A'] * N_A + ['B'] * N_B); role = sym == 'A'; L = (n / density) ** (1 / 3)
    want = np.where(role, 4, 2); want[rng.choice(N_A, DEFECTS, replace=False)] = 3; cos0 = np.where(role, COS0['A'], COS0['B']); ka = np.where(role, 0.8, 0.3)
    cell = np.eye(3) * L * inflate; pos = np.empty((n, 3))
    for i in range(n):
        while True:
            p = rng.uniform(0, cell[0, 0], 3)
            if i == 0 or np.linalg.norm(minimum_image(pos[:i] - p, cell), axis=1).min() > 1.8: pos[i] = p; break

    def relax(pos, cell, rounds):
        for _ in range(rounds):
            b, tri = bonds_of(pos, cell, role, want); res = minimize(energy, pos.ravel(), args=(cell, role, b, tri, cos0, ka), jac=True, method='L-BFGS-B', options=dict(maxiter=200)); pos = res.x.reshape(n, 3) % np.diag(cell)
        return pos

    pos = relax(pos, cell, 10)
    for f in np.linspace(inflate, 1.0, 7)[1:]:
        new = np.eye(3) * L * f; pos = pos * (np.diag(new) / np.diag(cell)); cell = new; pos = relax(pos, cell, 4)
    return sym, relax(pos, cell, 8), cell


def main():
    rng = np.random.default_rng(SEED); (OUT / 'structures').mkdir(parents=True, exist_ok=True); rows = []
    for density in DENSITIES:
        for k in range(PER_DENSITY):
            name = f'rho{density:.3f}_{k + 1}'; sym, pos, cell = network(rng, density); write_extxyz(OUT / 'structures' / f'{name}.extxyz', sym, pos, cell)
            rows.append(dict(cell=name, number_density=density, volume_A3=float(np.linalg.det(cell)), n_atoms=len(sym))); print(name, flush=True)
    pd.DataFrame(rows).to_csv(OUT / 'cells.csv', index=False); RADII.save(OUT / 'radii.json'); RADII_IONIC.save(OUT / 'radii_ionic.json')



# ---- synthetic electronic data: one trapping event per cell, tied to the void space the structure has
GAP = 8.0; N_VALENCE, N_CONDUCTION, N_SEMICORE = 180, 160, 20; HARTREE_EV = 27.211386245988


def spectrum(rng, n_atoms, depth, gap_shift):
    """Neutral and charged spectra: a semicore band, a B-character valence band, an A-character conduction band, and an A-character state empty in the gap before trapping and occupied below it after."""
    E_v = np.sort(rng.uniform(-6.5, 0, N_VALENCE)); E_c = np.sort(rng.uniform(GAP - gap_shift, GAP - gap_shift + 5, N_CONDUCTION)); E_s = np.sort(rng.uniform(-22, -20, N_SEMICORE))
    tail = lambda n, lo, hi: np.minimum(n_atoms, lo + (hi - lo) * np.linspace(0, 1, n) ** 0.5)                               # participation rising from the band tail into the band
    at_v, at_c, at_s = tail(N_VALENCE, 8, 0.7 * n_atoms)[::-1], tail(N_CONDUCTION, 6, 0.9 * n_atoms), np.full(N_SEMICORE, 0.5 * n_atoms)
    fA_v, fA_c, fA_s = rng.normal(0.15, 0.03, N_VALENCE), rng.normal(0.62, 0.03, N_CONDUCTION), np.full(N_SEMICORE, 0.5)
    pre = dict(E=E_c[0] - 0.3, atoms=rng.uniform(15, 40), fA=0.6); post = dict(E=depth, atoms=rng.uniform(3, 6), fA=0.6)             # the precursor below the conduction edge; the trap deep in the gap
    shift = {'neutral': 0.0, 'charged': 0.6 + rng.normal(0, 0.05)}                                                                     # a charged cell's levels are shifted by its compensating background: the alignment removes it
    out = {}
    for tag, st, occ_state in (('neutral', pre, 0.0), ('charged', post, 1.0)):
        E = np.concatenate([E_s, E_v, [st['E']], E_c]) + shift[tag]; o = np.argsort(E); occ = np.concatenate([np.ones(N_SEMICORE + N_VALENCE), [occ_state], np.zeros(N_CONDUCTION)])[o]
        wA = np.concatenate([fA_s, fA_v, [st['fA']], fA_c])[o]; at = np.concatenate([at_s, at_v, [st['atoms']], at_c])[o]
        out[tag] = dict(energy_eV=E[o], occupation=occ, weight_A=wA, weight_B=1 - wA, atoms=at)
    return out


def electronic(out):
    """Spectra, total energies and corrector summaries for every structure in out/structures/."""
    import json
    from vfa.io.extxyz import read_structure_files
    from vfa.voids.clearance import clearance_field
    from vfa.voids.sites import atom_free_volume
    rng = np.random.default_rng(SEED + 1); names, cells = read_structure_files(out / 'structures'); meta = pd.read_csv(out / 'cells.csv').set_index('cell'); (out / 'spectra').mkdir(exist_ok=True); (out / 'corrections').mkdir(exist_ok=True); rows = []
    for name, (sym, pos, cell) in zip(names, cells):
        C, n = clearance_field(pos, cell, RADII.of(sym), dx=0.3); fv = atom_free_volume(C, n, cell, pos, 0.8); isA = sym == 'A'; trap = int(np.flatnonzero(isA)[np.argmax(fv[isA])])
        depth = 1.0 + 0.06 * fv[trap] + rng.normal(0, 0.1); e_relax = 0.5 + 0.03 * fv[trap] + rng.normal(0, 0.05); gap_shift = 40 * (meta.number_density[name] - 0.062)     # planted: the trap deepens with the free volume at its atom; the gap narrows with density
        sp = spectrum(rng, len(sym), depth, gap_shift); np.savez_compressed(out / 'spectra' / f'{name}.npz', **{f'{t}_{k}': v for t, d in sp.items() for k, v in d.items()})
        e0 = -500.0 + rng.normal(0, 0.01); e_vert = e0 - (3.0 + 0.2 * gap_shift) / HARTREE_EV; e_rel = e_vert - e_relax / HARTREE_EV
        corr = dict(vertical=0.30 + rng.normal(0, 0.02), relaxed=0.33 + rng.normal(0, 0.02))
        for leg in ('vertical', 'relaxed'):
            c = corr[leg]; pot = {'schemes': {'FNV_planar_gaussian': {'E_corr_eV': round(c, 4), 'recommended': True, 'status': 'ok'}, 'LZ_scaled_PC_only': {'E_corr_eV': round(c * 0.9, 4)}, 'PC_IIC': {'E_corr_eV': round(c * 0.85, 4)}}, 'ks': {'delta_epsilon_eV': round(-0.4 * c, 4)}}
            dens = {'energies': {'E_corr_eV': round(c * 0.85, 4)}, 'neutral_pipeline': {'components': {'CD': {'selected_minus_q_median_eV': round(c * 0.15, 4), 'selected_quality': {'ok': True}}}}, 'state_corrections': [{'energy_expectation_eV': round(c * 0.7, 4), 'eigenvalue_shift_status': 'ok', 'numerical_reliability': {'ok': True}}]}
            (out / 'corrections' / f'{name}.{leg}.potential.json').write_text(json.dumps(pot, indent=1)); (out / 'corrections' / f'{name}.{leg}.density.json').write_text(json.dumps(dens, indent=1))
        rows.append(dict(cell=name, trap_atom=trap + 1, free_volume_trap_A3=fv[trap], e_neutral_Ha=e0, e_charged_vertical_Ha=e_vert, e_charged_relaxed_Ha=e_rel, kind='deep' if depth > 1.5 else 'shallow'))
    pd.DataFrame(rows).to_csv(out / 'events.csv', index=False)
    (out / 'README.md').write_text('# Example data\n\nSynthetic network glasses: 68 four-fold A atoms bridged by 136 two-fold B atoms, bond 1.62 A, no two A atoms sharing two B atoms, relaxed under bond, angle and repulsion springs at six number densities, five cells each (`cells.csv`); two A atoms per cell are three-fold by valence and the growth leaves further under-bonded atoms. `radii.json` and `radii_ionic.json` are the example\'s two radii conventions. No real material is represented.\n\n`spectra/<cell>.npz`: a neutral and a charged spectrum per cell (energies, occupations, species weights, participation) with a localised state, empty in the gap before trapping and occupied below it after. `events.csv`: the trap atom and the three total energies. `corrections/`: corrector-style summaries of each charged endpoint. By construction the trap deepens with the free volume at its atom and the gap narrows with density.\n')


if __name__ == '__main__':
    if sys.argv[1:2] == ['electronic']: electronic(OUT)
    else: main(); electronic(OUT)
