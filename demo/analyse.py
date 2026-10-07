#!/usr/bin/env python3
"""Structural descriptors and void geometry of every example cell (demo/out/cells.csv, sweep.csv, atoms.csv, rdf.npz, example_cell.npz)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from vfa.io.extxyz import read_structure_files
from vfa.structure.cell import volume
from vfa.voids.radii import Radii
from vfa.structure.graph import bond_graph, coordination, bond_vectors, angles, tetrahedral_order
from vfa.structure.rings import primitive_rings, ring_sizes, ring_extents
from vfa.structure.rdf import ensemble_rdf, partial_rdf, first_minimum
from vfa.voids.clearance import clearance_field, probe_sweep, clearance_statistics
from vfa.voids.network import voronoi_network, descriptors, pore_network, percolating_edges
from vfa.voids.sites import atom_free_volume, cavity_distance

DATA, OUT = ROOT / 'example/data', ROOT / 'demo/out'
PROBES = (0.3, 0.5, 0.8, 1.0, 1.2); PROBE = 0.8; DX = 0.2; RING_CAP = 24; RING_SIZES = range(3, 13)   # probe radii (A), the main probe, the grid spacing, the ring length cap (atoms)


def main():
    names, cells = read_structure_files(DATA / 'structures'); meta = pd.read_csv(DATA / 'cells.csv').set_index('cell'); radii = Radii.load(DATA / 'radii.json'); OUT.mkdir(exist_ok=True)
    r, g = ensemble_rdf(cells, ('A', 'B'), r_max=6.0, n_bins=600); fm = first_minimum(r, g); cutoff = float(np.mean(fm['basin_A']))   # the bond cutoff is measured, not declared: the middle of the first gap
    print(f"A-B first peak {fm['peak_A']:.2f} A, first minimum {cutoff:.2f} A (basin {fm['basin_A'][0]:.2f}-{fm['basin_A'][1]:.2f} A): bond cutoff")
    rows, sweep, atoms, rdfs, angs = [], [], [], {}, {}
    for name, (sym, pos, cell) in zip(names, cells):
        rho = meta.number_density[name]; V = volume(cell); isA = sym == 'A'
        e, o, lengths = bond_graph(sym, pos, cell, {('A', 'B'): cutoff}); z = coordination(len(sym), e); vec = bond_vectors(pos, cell, e, o)
        bab, aba = angles(vec, np.flatnonzero(isA)), angles(vec, np.flatnonzero(~isA)); q = tetrahedral_order(vec, np.flatnonzero(isA))
        found = primitive_rings(len(sym), e, o, RING_CAP); ext = ring_extents(found, pos, cell); keep = ext < np.linalg.norm(cell, axis=1).min(); rings = [r for r, k in zip(found, keep) if k]; sizes = ring_sizes(rings)   # rings spanning the cell are periodic artefacts
        rad = radii.of(sym); C, n = clearance_field(pos, cell, rad, dx=DX); sw = probe_sweep(C, n, cell, PROBES); st = clearance_statistics(C)
        net = voronoi_network(pos, cell, rad); vd = descriptors(net); pores = pore_network(net, PROBE)
        fv = atom_free_volume(C, n, cell, pos, PROBE); cd = cavity_distance(pos, cell, pores['pore_xyz'])
        row = dict(cell=name, number_density=rho, volume_A3=V, n_atoms=len(sym), z_A=z[isA].mean(), z_B=z[~isA].mean(), A_threefold=np.mean(z[isA] == 3), bond_A=lengths.mean(), bond_sd_A=lengths.std(),
                   angle_BAB=bab.mean(), angle_BAB_sd=bab.std(), angle_ABA=aba.mean(), angle_ABA_sd=aba.std(), q_A=np.nanmean(q), rings_per_1000A3=1000 * len(rings) / V, ring_cap_atoms=RING_CAP, rings_spanning_cell=int((~keep).sum()), ring_extent_max_over_cell=float(ext[keep].max() / np.linalg.norm(cell, axis=1).min()) if keep.any() else np.nan,
                   largest_clearance_A=st['largest_A'], clearance_p99_A=st['p99_A'], open_fraction=st['open_fraction'], n_pores=len(pores['pore_r']), pores_percolating=pores['pore_percolating'].mean(), **vd)
        row.update({f'ring{s}_share': np.mean(sizes == s) if len(sizes) else np.nan for s in RING_SIZES}); row.update({f"fraction_p{p:g}": x['fraction'] for p, x in zip(PROBES, sw)}); row.update({f"surface_p{p:g}": x['specific_surface'] for p, x in zip(PROBES, sw)}); rows.append(row)
        sweep += [dict(cell=name, number_density=rho, **x) for x in sw]
        atoms += [dict(cell=name, number_density=rho, atom=i + 1, species=sym[i], coordination=int(z[i]), free_volume_A3=fv[i], cavity_distance_A=cd[i], q=q[i] if isA[i] else np.nan) for i in range(len(sym))]
        angs[f'{name}:BAB'], angs[f'{name}:ABA'], angs[f'{name}:q'] = bab, aba, q[isA]
        for pair in (('A', 'B'), ('A', 'A'), ('B', 'B')): rdfs.setdefault((rho, pair), []).append(partial_rdf(sym, pos, cell, pair, r_max=6.0, n_bins=600)[1])
        if name == names[len(names) // 2]:
            k = n[2] // 2; mask, thr = percolating_edges(net, PROBE)
            np.savez_compressed(OUT / 'example_cell.npz', cell=cell, symbols=sym, positions=pos, slice_z=(k + 0.5) / n[2] * cell[2, 2], slice=C[:, :, k], field=C.astype(np.float32), shape=n, probe=PROBE, pore_xyz=pores['pore_xyz'], pore_r=pores['pore_r'],
                                pore_percolating=pores['pore_percolating'], throat_pairs=pores['throat_pairs'], throat_r=pores['throat_r'], edge_xyz=net['xyz'], edge_pairs=net['pairs'], edge_neck=net['neck'], edge_percolating=mask, Df_A=vd['Df_A'], Di_A=vd['Di_A'], name=name)
        print(f"{name}: z_A {row['z_A']:.2f}, rings {len(rings)}, Di {vd['Di_A']:.2f} Df {vd['Df_A']:.2f} A, accessible {row['fraction_p0.8']:.3f}, pores {len(pores['pore_r'])}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / 'cells.csv', index=False); pd.DataFrame(sweep).to_csv(OUT / 'sweep.csv', index=False); pd.DataFrame(atoms).to_csv(OUT / 'atoms.csv', index=False)
    np.savez_compressed(OUT / 'angles.npz', **angs); np.savez_compressed(OUT / 'rdf.npz', r=r, g_AB_all=g, cutoff=cutoff, peak=fm['peak_A'], keys=np.array([f'{rho:.3f}:{a}{b}' for (rho, (a, b)) in rdfs]), curves=np.array([np.mean(v, axis=0) for v in rdfs.values()]))
    pd.set_option('display.width', 200); print(pd.DataFrame(rows).groupby('number_density')[['z_A', 'A_threefold', 'rings_per_1000A3', 'Di_A', 'Df_A', 'fraction_p0.8', 'surface_p0.8', 'n_pores']].mean().round(3).to_string())


if __name__ == '__main__':
    main()
