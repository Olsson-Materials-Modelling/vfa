#!/usr/bin/env python3
"""Descriptors and void geometry of every example cell, with provenance sidecars (demo/out)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from vfa.io.extxyz import read_structure_files, structure_paths
from vfa.structure.cell import volume, shortest_vector
from vfa.voids.radii import Radii
from vfa.structure.graph import bond_graph, coordination, bond_vectors, angles, tetrahedral_order
from vfa.structure.rings import primitive_rings, ring_sizes, ring_extents
from vfa.structure.rdf import ensemble_rdf, partial_rdf, first_minimum
from vfa.voids.clearance import clearance_field, probe_sweep, clearance_statistics
from vfa.voids.network import voronoi_network, descriptors, pore_network
from vfa.voids.sites import atom_free_volume, cavity_distance
from vfa.structure.scattering import ensemble_direct, ensemble_transform, total_sq, total_gr
from vfa.provenance import RunContext, write_csv, write_sidecar

DATA, OUT = ROOT / 'example/data', ROOT / 'demo/out'
PROBES = (0.3, 0.5, 0.8, 1.0, 1.2); PROBE = 0.8; DX = 0.2; RING_CAP = 24; RING_SIZES = range(3, 13)


def main():
    """Descriptors and void geometry of every example cell, written with sidecars."""
    names, cells = read_structure_files(DATA / 'structures'); meta = pd.read_csv(DATA / 'cells.csv').set_index('cell'); radii = Radii.load(DATA / 'radii.json'); OUT.mkdir(exist_ok=True)
    r, g = ensemble_rdf(cells, ('A', 'B'), r_max=6.0, n_bins=600); fm = first_minimum(r, g)
    if not fm['resolved']: raise ValueError(f"the A-B first minimum is not resolved inside the search window (basin {fm['basin_A']})")
    cutoff = fm['basin_middle_A']
    print(f"A-B first peak {fm['peak_A']:.2f} A; first-minimum basin {fm['basin_A'][0]:.2f}-{fm['basin_A'][1]:.2f} A; bond cutoff at its middle {cutoff:.2f} A")
    rows, sweep, atoms, rdfs, angs = [], [], [], {}, {}
    for name, (sym, pos, cell) in zip(names, cells):
        rho = meta.number_density[name]; V = volume(cell); isA = sym == 'A'
        e, o, lengths = bond_graph(sym, pos, cell, {('A', 'B'): cutoff}); z = coordination(len(sym), e); vec = bond_vectors(pos, cell, e, o)
        bab, aba = angles(vec, np.flatnonzero(isA)), angles(vec, np.flatnonzero(~isA)); q = tetrahedral_order(vec, np.flatnonzero(isA))
        found = primitive_rings(len(sym), e, o, RING_CAP, images=True); ext = ring_extents(found, pos, cell); keep = ext < np.linalg.norm(shortest_vector(cell)); rings = [r for r, k in zip(found, keep) if k]; sizes = ring_sizes(rings)
        rad = radii.of(sym); C, n = clearance_field(pos, cell, rad, dx=DX); sw = probe_sweep(C, n, cell, PROBES); st = clearance_statistics(C)
        net = voronoi_network(pos, cell, rad); vd = descriptors(net); pores = pore_network(net, PROBE)
        fv = atom_free_volume(C, n, cell, pos, rad, PROBE); cd = cavity_distance(pos, cell, pores['pore_xyz'])
        row = dict(cell=name, number_density=rho, volume_A3=V, n_atoms=len(sym), z_A=z[isA].mean(), z_B=z[~isA].mean(), A_threefold=np.mean(z[isA] == 3), bond_A=lengths.mean(), bond_sd_A=lengths.std(),
                   angle_BAB=bab.mean(), angle_BAB_sd=bab.std(), angle_ABA=aba.mean(), angle_ABA_sd=aba.std(), q_A=np.nanmean(q), rings_per_1000A3=1000 * len(rings) / V, ring_cap_atoms=RING_CAP, rings_spanning_cell=int((~keep).sum()), ring_extent_max_over_cell=float(ext[keep].max() / np.linalg.norm(cell, axis=1).min()) if keep.any() else np.nan,
                   largest_clearance_A=st['largest_A'], clearance_p99_A=st['p99_A'], open_fraction=st['open_fraction'], n_pores=len(pores['pore_r']), pores_percolating=pores['pore_percolating'].mean() if pores['decided'] else np.nan, **vd)
        row.update({f'ring{s}_share': np.mean(sizes == s) if len(sizes) else np.nan for s in RING_SIZES}); row.update({f"fraction_p{p:g}": x['fraction'] for p, x in zip(PROBES, sw)}); row.update({f"surface_p{p:g}": x['surface_m2_cm3'] for p, x in zip(PROBES, sw)}); rows.append(row)
        sweep += [dict(cell=name, number_density=rho, **x) for x in sw]
        atoms += [dict(cell=name, number_density=rho, atom=i + 1, species=sym[i], coordination=int(z[i]), free_volume_A3=fv[i], cavity_distance_A=cd[i], q=q[i] if isA[i] else np.nan) for i in range(len(sym))]
        angs[f'{name}:BAB'], angs[f'{name}:ABA'], angs[f'{name}:q'] = bab, aba, q[isA]
        rdfs.setdefault(rho, []).append(partial_rdf(sym, pos, cell, ('A', 'B'), r_max=6.0, n_bins=600)[1])
        if name == names[len(names) // 2]:
            k = n[2] // 2
            np.savez_compressed(OUT / 'example_cell.npz', cell=cell, symbols=sym, positions=pos, slice_z=(k + 0.5) / n[2] * cell[2, 2], slice=C[:, :, k], field=C.astype(np.float32), shape=n, probe=PROBE, pore_xyz=pores['pore_xyz'], pore_r=pores['pore_r'],
                                pore_percolating=pores['pore_percolating'], throat_pairs=pores['throat_pairs'], Df_A=vd['Df_A'], Di_A=vd['Di_A'], name=name)
        print(f"{name}: z_A {row['z_A']:.2f}, rings {len(rings)}, Di {vd['Di_A']:.2f} Df {vd['Df_A']:.2f} A, accessible {row['fraction_p0.8']:.3f}, pores {len(pores['pore_r'])}, network {vd['n_nodes']} nodes, complete {vd['complete']}", flush=True)
    ctx = RunContext().with_config(dict(probes=PROBES, probe=PROBE, dx=DX, ring_cap=RING_CAP, cutoff_A=cutoff, radii=radii.name, radii_A=radii.table)).with_inputs(structure_paths(DATA / 'structures') + [DATA / 'cells.csv', DATA / 'radii.json'])
    for df, f in ((rows, 'cells.csv'), (sweep, 'sweep.csv'), (atoms, 'atoms.csv')): write_csv(pd.DataFrame(df), OUT / f, ctx, float_format=None)
    np.savez_compressed(OUT / 'angles.npz', **angs); np.savez_compressed(OUT / 'rdf.npz', r=r, cutoff=cutoff, densities=np.array(list(rdfs)), curves_AB=np.array([np.mean(v, axis=0) for v in rdfs.values()]))
    for f in ('angles.npz', 'rdf.npz', 'example_cell.npz'): write_sidecar(OUT / f, ctx)
    same = names[len(names) // 2].split('_')[0]; mid = [x for n, x in zip(names, cells) if n.split('_')[0] == same]; b = {'A': 1.0, 'B': 1.0}; qd, Sd, cd = ensemble_direct(mid, 20.0, dq=0.2); qt = np.linspace(0.5, 20.0, 391); St, ct = ensemble_transform(mid, qt, window='lorch')
    rmax = min(float(np.linalg.norm(shortest_vector(cl))) / 2 for _, _, cl in mid); G = [total_gr(s, p, cl, b, r_max=rmax) for s, p, cl in mid]
    np.savez_compressed(OUT / 'scattering.npz', q_direct=qd, S_direct=total_sq(qd, Sd, cd, b)[1], q_transform=qt, S_transform=total_sq(qt, St, ct, b)[1], r=G[0][0], G=np.mean([g for _, g in G], axis=0))
    write_sidecar(OUT / 'scattering.npz', ctx.with_config(dict(cells=same, weights=b, q_max=20.0, dq=0.2, window='lorch')))
    pd.set_option('display.width', 200); print(pd.DataFrame(rows).groupby('number_density')[['z_A', 'A_threefold', 'rings_per_1000A3', 'Di_A', 'Df_A', 'fraction_p0.8', 'surface_p0.8', 'n_pores']].mean().round(3).to_string())
    print(f"pore networks complete and exact in {sum(r['complete'] and r['exact'] for r in rows)} of {len(rows)} cells; nodes added by the enumeration {sum(r['n_nodes_added'] for r in rows)}, edges without a node {sum(r['n_isolated_edges'] for r in rows)}")


if __name__ == '__main__':
    main()
