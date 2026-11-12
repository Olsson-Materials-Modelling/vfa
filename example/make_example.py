#!/usr/bin/env python3
"""Synthetic example data: thirty network-glass cells at six densities with one trapping event each, written to example/data."""
import json, textwrap, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vfa.units import HARTREE_EV
from vfa.structure.cell import minimum_image, separations, distances
from vfa.io.extxyz import write_extxyz, read_structure_files
from vfa.voids.radii import Radii
from vfa.voids.clearance import clearance_field
from vfa.voids.sites import atom_free_volume

SEED = 11; N_A, N_B = 68, 136; R0 = 1.62; DENSITIES = (0.052, 0.056, 0.060, 0.064, 0.068, 0.072); PER_DENSITY = 5; DEFECTS = 2
COS0 = {'A': -1 / 3, 'B': -0.82}; RADII = Radii({'A': 1.1, 'B': 0.7}, 'covalent'); RADII_IONIC = Radii({'A': 0.3, 'B': 1.4}, 'ionic')
GAP = 8.0; N_VALENCE, N_CONDUCTION, N_SEMICORE = 180, 160, 20; PRECURSOR_eV = 0.3; DEPTH0_eV, DEPTH_eV_PER_A3, DEPTH_SD_eV = 1.0, 0.04, 0.1; DEEP_eV = 2.0; PROBE = 0.8; DX = 0.3
STRUCTURE_VERSIONS = 'numpy 2.4 and scipy 1.17'; EPS_INF, EPS_0 = 4.0, 8.0; CORRECTOR = {'tool_name': 'cp2k_charge_corrections', 'code_version': '0.8.0', 'schema_version': '4.0'}
OUT = Path(__file__).resolve().parent / 'data'


def bonds_of(pos, cell, role, want):
    """Bond list (closest A-B pairs first, while both atoms have free valence) and the angle triplets at every atom."""
    n = len(pos); d = distances(pos, cell); ii, jj = np.nonzero(np.triu(role[:, None] != role[None, :])); order = np.argsort(d[ii, jj], kind='stable'); free = want.copy(); b = []
    partners = [set() for _ in range(n)]
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


def relax(pos, cell, rounds, role, want, cos0, ka):
    """Positions after rounds of bond-list refresh and spring minimisation in a cubic cell."""
    for _ in range(rounds):
        b, tri = bonds_of(pos, cell, role, want); res = minimize(energy, pos.ravel(), args=(cell, role, b, tri, cos0, ka), jac=True, method='L-BFGS-B', options=dict(maxiter=200)); pos = res.x.reshape(len(pos), 3) % np.diag(cell)
    return pos


def network(rng, density, inflate=1.35):
    """One relaxed network grown in an inflated box and compressed to the number density; DEFECTS A atoms stay three-fold."""
    n = N_A + N_B; sym = np.array(['A'] * N_A + ['B'] * N_B); role = sym == 'A'; L = (n / density) ** (1 / 3)
    want = np.where(role, 4, 2); want[rng.choice(N_A, DEFECTS, replace=False)] = 3; cos0 = np.where(role, COS0['A'], COS0['B']); ka = np.where(role, 0.8, 0.3)
    cell = np.eye(3) * L * inflate; pos = np.empty((n, 3))
    for i in range(n):
        while True:
            p = rng.uniform(0, cell[0, 0], 3)
            if i == 0 or np.linalg.norm(minimum_image(pos[:i] - p, cell), axis=1).min() > 1.8: pos[i] = p; break
    pos = relax(pos, cell, 10, role, want, cos0, ka)
    for f in np.linspace(inflate, 1.0, 7)[1:]:
        new = np.eye(3) * L * f; pos = pos * (np.diag(new) / np.diag(cell)); cell = new; pos = relax(pos, cell, 4, role, want, cos0, ka)
    return sym, relax(pos, cell, 8, role, want, cos0, ka), cell


def spectrum(rng, n_atoms, depth, gap_shift):
    """Neutral and charged spectra of one cell: semicore, valence and conduction bands, the empty precursor and the occupied trap."""
    E_v = np.sort(rng.uniform(-6.5, 0, N_VALENCE)); E_c = np.sort(rng.uniform(GAP - gap_shift, GAP - gap_shift + 5, N_CONDUCTION)); E_s = np.sort(rng.uniform(-22, -20, N_SEMICORE))
    tail = lambda n, lo, hi: np.minimum(n_atoms, lo + (hi - lo) * np.linspace(0, 1, n) ** 0.5)
    at_v, at_c, at_s = tail(N_VALENCE, 8, 0.7 * n_atoms)[::-1], tail(N_CONDUCTION, 6, 0.9 * n_atoms), np.full(N_SEMICORE, 0.5 * n_atoms)
    fA_v, fA_c, fA_s = rng.normal(0.15, 0.03, N_VALENCE), rng.normal(0.62, 0.03, N_CONDUCTION), np.full(N_SEMICORE, 0.5)
    pre = dict(E=E_c[0] - PRECURSOR_eV, atoms=rng.uniform(15, 40), fA=0.6); post = dict(E=E_c[0] - depth, atoms=rng.uniform(3, 6), fA=0.6)
    shift = {'neutral': 0.0, 'charged': 0.6 + rng.normal(0, 0.05)}; out = {}
    for tag, st, occ_state in (('neutral', pre, 0.0), ('charged', post, 1.0)):
        E = np.concatenate([E_s, E_v, [st['E']], E_c]) + shift[tag]; o = np.argsort(E, kind='stable'); occ = np.concatenate([np.ones(N_SEMICORE + N_VALENCE), [occ_state], np.zeros(N_CONDUCTION)])[o]
        wA = np.concatenate([fA_s, fA_v, [st['fA']], fA_c])[o]; at = np.concatenate([at_s, at_v, [st['atoms']], at_c])[o]
        out[tag] = dict(energy_eV=E[o], occupation=occ, weight_A=wA, weight_B=1 - wA, atoms=at)
    return out


def corrector_json(name, potential, density):
    """Synthetic schema-4.0 corrector JSON of one charged cell relaxed in its charge, from the two modes' clamped-ion image energies, holding only the fields vfa reads."""
    run = {s: f'relaxed_{s}/{name}' for s in ('charged', 'neutral')}; r = lambda x: round(float(x), 4); p, d = potential * EPS_INF / EPS_0, density * EPS_INF / EPS_0
    picked = lambda mode, scheme, e, **extra: dict(mode=mode, scheme=scheme, E_corr_eV=r(e), reliability_ok=True, reasons=[], impact_eV=r(0.02 * e), screening='static', ionic_geometry='relaxed', delta_epsilon_eV=r(2 * e), level_reliability_ok=True, **extra)
    pot = {'inputs': dict(CORRECTOR, defect_cube=f"{run['charged']}/v_hartree.cube", bulk_cube=f"{run['neutral']}/v_hartree.cube"), 'selected': picked('potential', 'concerted_FNV_planar_gaussian', p),
           'schemes': {'concerted_FNV_planar_gaussian': {'E_corr_eV': r(p)}, 'concerted_LZ_scaled_PC_plus_alignment': {'E_corr_eV': r(1.02 * p)}, 'concerted_PC_IIC': {'E_corr_eV': r(1.5 * p)}, 'concerted_site_gaussian_median': {'E_corr_eV': r(1.1 * p)}}}
    dens = {'inputs': dict(CORRECTOR, rho_charged_cube=f"{run['charged']}/electron_density.cube", rho_neutral_cube=f"{run['neutral']}/electron_density.cube"), 'selected': picked('density', 'bare_carrier_image_plus_DA_model_alignment', d, carrier='localized'),
            'energies': {'selected_correction_status': 'applied_model_referenced_alignment'}}
    return pot, dens


def readme(cells):
    """The example README: construction, the planted event, seeds and regeneration commands."""
    text = ('# Example data\n\n'
            f'Synthetic network glasses, no real material: {len(cells)} cells (`cells.csv`), {PER_DENSITY} at each of {len(DENSITIES)} number densities, each of {N_A} A and {N_B} B atoms. '
            f'They are grown to a target of four B neighbours per A atom (three for {DEFECTS} A atoms per cell) and two A neighbours per B atom, bond {R0} A, with no two A atoms sharing two B atoms, '
            'and relaxed under bond, angle and repulsion springs. `radii.json` and `radii_ionic.json` are the two radii conventions.\n\n'
            '`spectra/<cell>.npz`: a neutral and a charged spectrum (energies, occupations, species weights, participation). '
            f'The trap atom is the A atom with the largest free volume (probe {PROBE} A, grid {DX} A, `radii.json`). '
            f'The added electron occupies an A-character gap state {DEPTH0_eV} eV + {DEPTH_eV_PER_A3} eV/A3 x (free volume at the trap atom) + N(0, {DEPTH_SD_eV} eV) below the lowest conduction state; '
            f'before trapping an empty A-character state sits {PRECURSOR_eV} eV below that edge. The gap narrows with density. '
            f'`events.csv`: the free volume at the trap atom, the three total energies, and the kind (deep above {DEEP_eV} eV, else shallow). '
            '`corrections/`: synthetic schema-4.0 corrector JSON of each charged cell, relaxed in its charge, with only the fields vfa reads.\n\n'
            f'`python example/make_example.py` regenerates everything (seeds {SEED} and {SEED + 1}), bit for bit with {STRUCTURE_VERSIONS}; under other versions the structure minimisation can take another path. '
            '`python example/make_example.py electronic` regenerates the spectra, `events.csv`, `corrections/` and this file from the shipped structures.')
    return '\n\n'.join(textwrap.fill(p, 130, break_long_words=False, break_on_hyphens=False) for p in text.strip().split('\n\n')) + '\n'



def electronic(out):
    """Spectra, total energies, corrector JSON and the README for every structure in out/structures/."""
    rng = np.random.default_rng(SEED + 1); names, cells = read_structure_files(out / 'structures'); meta = pd.read_csv(out / 'cells.csv').set_index('cell'); (out / 'spectra').mkdir(exist_ok=True); (out / 'corrections').mkdir(exist_ok=True); rows = []
    for name, (sym, pos, cell) in zip(names, cells):
        C, n = clearance_field(pos, cell, RADII.of(sym), dx=DX); fv = atom_free_volume(C, n, cell, pos, RADII.of(sym), PROBE); isA = sym == 'A'; trap = int(np.flatnonzero(isA)[np.argmax(fv[isA])])
        depth = DEPTH0_eV + DEPTH_eV_PER_A3 * fv[trap] + rng.normal(0, DEPTH_SD_eV); e_relax = 0.5 + 0.015 * fv[trap] + rng.normal(0, 0.05); gap_shift = 40 * (meta.number_density[name] - 0.062)
        sp = spectrum(rng, len(sym), depth, gap_shift); np.savez_compressed(out / 'spectra' / f'{name}.npz', **{f'{t}_{k}': v for t, d in sp.items() for k, v in d.items()})
        e0 = -500.0 + rng.normal(0, 0.01); e_vert = e0 - (3.0 + 0.2 * gap_shift) / HARTREE_EV; e_rel = e_vert - e_relax / HARTREE_EV
        pot, dens = corrector_json(name, 0.30 + rng.normal(0, 0.02), 0.31 + rng.normal(0, 0.02))
        (out / 'corrections' / f'{name}.relaxed.potential.json').write_text(json.dumps(pot, indent=1) + '\n'); (out / 'corrections' / f'{name}.relaxed.density.json').write_text(json.dumps(dens, indent=1) + '\n')
        rows.append(dict(cell=name, free_volume_trap_A3=fv[trap], e_neutral_Ha=e0, e_charged_vertical_Ha=e_vert, e_charged_relaxed_Ha=e_rel, kind='deep' if depth > DEEP_eV else 'shallow'))
    pd.DataFrame(rows).to_csv(out / 'events.csv', index=False); (out / 'README.md').write_text(readme(cells))


def main():
    """Structures, cells.csv and the two radii files, then the electronic data."""
    rng = np.random.default_rng(SEED); (OUT / 'structures').mkdir(parents=True, exist_ok=True); rows = []
    for density in DENSITIES:
        for k in range(PER_DENSITY):
            name = f'rho{density:.3f}_{k + 1}'; sym, pos, cell = network(rng, density); write_extxyz(OUT / 'structures' / f'{name}.extxyz', sym, pos, cell)
            rows.append(dict(cell=name, number_density=density, volume_A3=float(np.linalg.det(cell)), n_atoms=len(sym)))
    pd.DataFrame(rows).to_csv(OUT / 'cells.csv', index=False); RADII.save(OUT / 'radii.json'); RADII_IONIC.save(OUT / 'radii_ionic.json'); electronic(OUT)


if __name__ == '__main__':
    if sys.argv[1:2] == ['electronic']: electronic(OUT)
    else: main()
