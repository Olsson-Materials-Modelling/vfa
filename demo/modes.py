#!/usr/bin/env python3
"""The modes side by side on every cell: Voronoi against pointwise, three radii conventions, three ring criteria, and the convergence checks (demo/out/modes.csv, convergence.csv)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from vfa.io.extxyz import read_structure_files
from vfa.voids.radii import Radii
from vfa.voids.clearance import clearance_field, sample_clearance, accessible_fraction, clearance_statistics, sampling_convergence, grid_convergence
from vfa.voids.network import voronoi_network, descriptors
from vfa.structure.graph import bond_graph
from vfa.structure.rings import rings, ring_sizes, ring_extents

DATA, OUT = ROOT / 'example/data', ROOT / 'demo/out'; PROBE = 0.8; CUTOFF = 2.06; CAP = 24


def main():
    names, cells = read_structure_files(DATA / 'structures'); meta = pd.read_csv(DATA / 'cells.csv').set_index('cell'); rows, conv = [], []
    for name, (sym, pos, cell) in zip(names, cells):
        row = dict(cell=name, number_density=meta.number_density[name]); e, o, _ = bond_graph(sym, pos, cell, {('A', 'B'): CUTOFF})
        for crit in ('primitive', 'king', 'guttman'):
            found = rings(len(sym), e, o, CAP, crit); keep = ring_extents(found, pos, cell) < np.linalg.norm(cell, axis=1).min(); sz = ring_sizes([r for r, k in zip(found, keep) if k]); row[f'rings_{crit}'] = len(sz); row.update({f'{crit}_size{s}': int((sz == s).sum()) for s in range(3, 13)})
        for radii in (Radii.centre(sym), Radii.load(DATA / 'radii.json'), Radii.load(DATA / 'radii_ionic.json')):
            rad = radii.of(sym); C, n = clearance_field(pos, cell, rad, dx=0.2); s = sample_clearance(pos, cell, rad, 8192); d = descriptors(voronoi_network(pos, cell, rad))
            row.update({f'{radii.name}_grid_fraction': accessible_fraction(C, PROBE), f'{radii.name}_pointwise_fraction': accessible_fraction(s, PROBE), f'{radii.name}_grid_largest_A': 2 * clearance_statistics(C)['largest_A'], f'{radii.name}_pointwise_largest_A': 2 * clearance_statistics(s)['largest_A'],
                        f'{radii.name}_Di_A': d['Di_A'], f'{radii.name}_Df_A': d['Df_A'], f'{radii.name}_Dif_A': d['Dif_A']})
        rows.append(row); print(name, flush=True)
        if name == names[len(names) // 2]:
            rad = Radii.load(DATA / 'radii.json').of(sym); conv += [dict(cell=name, mode='pointwise', **r) for r in sampling_convergence(pos, cell, rad, PROBE)] + [dict(cell=name, mode='grid', **r) for r in grid_convergence(pos, cell, rad, PROBE)]
    pd.DataFrame(rows).to_csv(OUT / 'modes.csv', index=False); pd.DataFrame(conv).to_csv(OUT / 'convergence.csv', index=False); m = pd.DataFrame(rows); pd.set_option('display.width', 220)
    print(m[[c for c in m.columns if c.endswith('_fraction') or c.endswith('_Di_A') or c.startswith('rings_')]].describe().loc[['mean', 'min', 'max']].round(3).T.to_string())


if __name__ == '__main__':
    main()
