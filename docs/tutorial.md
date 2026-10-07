# Tutorial

Each step runs on the example data; `bash demo/run.sh` runs them all. Paths are relative to the repository. The conventions
behind every call are in [conventions.md](conventions.md).

## 1. Structures and radii

```python
from pathlib import Path
from vfa.io.extxyz import read_structure_files
from vfa.voids.radii import Radii
names, cells = read_structure_files(Path('example/data/structures'))      # (symbols, positions, cell) per file; general cells
radii = Radii.load('example/data/radii.json')                              # or Radii.covalent(symbols), Radii.ionic(symbols), Radii.centre(symbols)
```

The radii decide what void means: covalent radii leave the space outside the bonded network, ionic radii the space between
ions (one charge state and coordination per species), zero radii the bare centres. The tables ship in `vfa/data/` with their
sources. State the convention with every result. CP2K restart files: `vfa.io.cp2k.read_restart`; LAMMPS data files:
`vfa.io.lammps.read_lammps_data`.

## 2. The bond cutoff, measured

```python
from vfa.structure.rdf import ensemble_rdf, first_minimum
r, g = ensemble_rdf(cells, ('A', 'B'), r_max=6.0, n_bins=600)
fm = first_minimum(r, g, search=(1.0, 4.5), smoothing_bins=9); cutoff = fm['minimum_A']
```

The first peak is the maximum of the smoothed curve in the window, the cutoff the minimum after it; `fm['basin_A']` is the
range over which g stays within 10 % of that minimum. A wide basin means the cutoff is not sharp: check a result at both ends.
Lattice peaks are spikes and need a window on the first shell.

## 3. Bond graph and local geometry

```python
from vfa.structure.graph import bond_graph, coordination, bond_vectors, angles, tetrahedral_order, planarity
sym, pos, cell = cells[0]
edges, offsets, lengths = bond_graph(sym, pos, cell, {('A', 'B'): cutoff})    # one entry per species pair that bonds
z = coordination(len(sym), edges); vec = bond_vectors(pos, cell, edges, offsets)
theta = angles(vec, [i for i in range(len(sym)) if sym[i] == 'A'])             # B-A-B angles
q = tetrahedral_order(vec, range(len(sym)))                                    # four shortest bonds; NaN below four
p = planarity(vec, range(len(sym)))                                            # angle sum at three-bonded atoms
```

Edges carry the image of the second atom, so a bond across the boundary is one bond with an exact vector.

## 4. Rings

```python
from vfa.structure.rings import rings, rings_converged, ring_sizes, ring_extents
prim = rings(len(sym), edges, offsets, max_length=24, criterion='primitive')      # Franzblau shortest-path rings: no shortcut between any two ring atoms
king = rings(len(sym), edges, offsets, max_length=24, criterion='king')           # shortest paths between two neighbours of an atom
gutt = rings(len(sym), edges, offsets, max_length=24, criterion='guttman')        # shortest paths between the two atoms of a bond
conv, cap = rings_converged(len(sym), edges, offsets, criterion='primitive')      # the cap raised by 4 until the count repeats
keep = ring_extents(prim, pos, cell) < np.linalg.norm(cell, axis=1).min()        # rings spanning the cell are periodic artefacts
sizes = ring_sizes([r for r, k in zip(prim, keep) if k])                          # network-former units around each ring
```

The criteria agree on diamond and differ on a cubic lattice (`tests/test_structure.py`). All three enumerate on the covering
graph, so a walk returning to a periodic image is never a ring; a ring whose unwrapped extent reaches a cell length runs through
the images and is dropped. `rings_converged` needs a cell large compared with its rings; otherwise declare the cap and say so
(the demo: 24 atoms, 204-atom cells).

## 5. Clearance: grid and pointwise

```python
from vfa.voids.clearance import clearance_field, sample_clearance, accessible_fraction, void_surface, probe_sweep, clearance_statistics, sampling_convergence, grid_convergence
C, n = clearance_field(pos, cell, radii.of(sym), dx=0.15)                     # grid mode: the field, also the surface
s = sample_clearance(pos, cell, radii.of(sym), n=8192, sampler='sobol')         # pointwise mode: no grid, no surface
rows = probe_sweep(C, n, cell, probes=(0.3, 0.5, 0.8, 1.0, 1.2))              # fraction, volume, surface, specific surface
stats = clearance_statistics(s)                                               # largest clearance, percentiles, open fraction
grid_convergence(pos, cell, radii.of(sym), probe=0.8); sampling_convergence(pos, cell, radii.of(sym), probe=0.8)
```

Both modes give the accessible fraction, isolated cavities included, and agree when converged (fig08a); only the grid gives the
surface. The largest clearance of either underestimates the largest included sphere, which the Voronoi network gives exactly
(fig08b). Run both convergence checks before quoting a number (fig08d).

## 6. The Voronoi pore network

```python
from vfa.voids.network import voronoi_network, descriptors, pore_network, percolating_edges
net = voronoi_network(pos, cell, radii.of(sym))
d = descriptors(net)                        # Di_A, Df_A, Dif_A: diameters; cavity_to_neck = Di/Df
pores = pore_network(net, probe=0.8)        # pores with radius and centre, throats with radius, which pores percolate
mask, thr = percolating_edges(net)          # the single percolating thread at the free-sphere radius
```

Voronoi vertices are the cavity centres, edges the channels, the neck of a channel its smallest clearance. The largest free
sphere is the percolation threshold: a loop with a net lattice translation, which separates a channel from a large pocket.
`merge_frac` (0.92) is the one free parameter of the pore partition.

## 7. The void space from each atom

```python
from vfa.voids.sites import atom_free_volume, cavity_distance
fv = atom_free_volume(C, n, cell, pos, probe=0.8)       # accessible volume nearest to each atom; sums to the accessible volume
cd = cavity_distance(pos, cell, pores['pore_xyz'])      # distance to the nearest pore centre
```

## 8. Reading a spectrum

```python
from vfa.electronic.spectrum import character, occupied_count, semicore_reference, align, band_ladder, tail_spread, bulk_character, trap_level, delocalisation_onset, internal_depth
f = character({'A': wA, 'B': wB}, 'A'); n = occupied_count(occupation)         # species character per state; occupied count
k, ref = semicore_reference(E, n); E = align(E, ref)                          # one scale for the neutral and the charged cell
ladder = band_ladder(E, n); tail = tail_spread(E, n)                          # gaps to the k-th empty state; the band control
trap = trap_level(E, f, n, bulk_character(f, n)['valence'])                   # the trap by character, not by occupation
onset = delocalisation_onset(E, atoms, n, n_atoms, fractions=(5, 4, 3))       # the threshold is swept
depth = internal_depth(E, atoms, n, n_atoms)                                  # inside one spectrum, no alignment
```

`vfa.io.cp2k.read_spectrum` reads CP2K projected densities of states into these arrays; participation comes from per-atom
weights (`participation_from_weights`) or an export. Energies of the event: `vfa.electronic.energies.trapping_energies`,
checked by `closes_cycle`. Mulliken blocks of a relaxation: `vfa.io.cp2k.mulliken_blocks`, `vfa.electronic.mulliken.track`.

## 9. Corrections

```python
from vfa.io.cp2k import read_output
from vfa.corrections.gates import gate_pair
from vfa.corrections.schemes import summarise, corrected_relaxation
bad = gate_pair(read_output(charged_log), read_output(neutral_log), energy_reference={'charged': e_ref})   # [] when the pair passes
row = summarise('cell.potential.json', 'cell.density.json')                   # the corrector's schemes, labelled
e_corr = corrected_relaxation(e_relax, row_vertical['density_corrected_eV'], row_relaxed['density_corrected_eV'])
```

The corrector is cp2k-charge-corrections; its two modes are alternatives, never summed.

## 10. Statistics

```python
from vfa.stats.trends import positions, held_out_window, smooth_in_order, linear_change
from vfa.stats.contrasts import cliff_delta_interval, scale_ratio
from vfa.stats.intervals import percentile_bootstrap, bounded_prevalence, wilson
from vfa.stats.models import held_out_shares, learning_curve
q, at = positions(density); w = held_out_window(Y, widths=(2, 4, 6, 10, 15))  # Y: cells sorted by density, one column per quantity
ends = smooth_in_order(Y, at, w['chosen'])                                    # the values at the density extremes
ch = linear_change(density, y, q[0], q[1])                                    # Theil-Sen and least squares side by side
```

The cell is the sampling unit everywhere. Where an order enters a result, fix the tie-break (`demo/trends.py`).

## 11. Provenance

```python
from vfa.provenance import RunContext, write_csv
ctx = RunContext().with_config(dict(cutoff=cutoff, radii=radii.name)).with_inputs(paths)
write_csv(table, 'out/cells.csv', ctx)                                        # cells.csv and cells.csv.meta.json
```

## 12. The demo

`demo/analyse.py` runs steps 2 to 7 on every cell (`demo/out/cells.csv`, `sweep.csv`, `atoms.csv`, `rdf.npz`, `angles.npz`,
`example_cell.npz`); `demo/modes.py` the modes side by side; `demo/electronic.py` steps 8 and 9 on the synthetic spectra;
`demo/trends.py` step 10; `demo/schematic.py` draws the workflow (`fig00_workflow`). `demo/plot.py` draws:

- `fig01_clearance_and_pores`: the clearance field of one cell in a slice with the probe isoline; its pore network.
- `fig02_probe_sweep`: accessible fraction and specific surface against the probe radius at the lowest and highest density.
- `fig03_density_trends`: largest included and free spheres, accessible fraction and ring-size shares against density.
- `fig04_sites`: free volume at the atom by species; distance to the nearest pore of three- and four-fold A atoms.
- `fig05_structure`: the A–B pair correlation with the measured cutoff, bond-angle distributions, tetrahedral order.
- `fig06_spectrum_reading`: one cell before and after trapping: precursor, trap by character, onsets.
- `fig07_trends_and_model`: trends with density with the held-out window and both straight-line changes; the held-out model.
- `fig08_modes`: grid against pointwise, three radii conventions, three ring criteria, the convergence checks.
- `fig09_void_cube`: the void surface of one cell in three dimensions, with the atoms and with the pore network.

## 13. Your own material

Point `read_structure_files` at a folder of extended-XYZ files, give `Radii` your species, measure the cutoff with step 2 for
each bonding pair, and keep `demo/analyse.py` as the template: its species names and probe radii are the only things to change.
The example data are synthetic (`example/data/README.md`); no real material is represented.
