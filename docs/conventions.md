# Conventions

Defaults and options of each part; signatures and docstrings give the rest. Lengths are in Å; energies in eV, total energies from
CP2K in hartree.

## Periodic geometry

Positions are folded into the cell; results refer to the positions given, bonds by their image offsets. The minimum image is exact
in any cell shape. `workers=-1` (default) uses every CPU available and does not change results.

## Void space

| Mode | Function | Gives | Defaults and options | Check |
|---|---|---|---|---|
| Pore network | `voids.network.voronoi_network`, `descriptors`, `pore_network` | diameters Di (largest included sphere), Df (largest free sphere), Dif (largest included sphere on the percolating component); pores and throats | `verify=True` enumerates every node as well as tracing the network; `merge_frac=0.92` merges nodes into pores | `exact` |
| Grid clearance | `voids.clearance.clearance_field`, `probe_sweep`, `void_surface` | accessible fraction and volume, void surface, specific surface (m2 cm-3) | `dx=0.15` | `grid_convergence` |
| Pointwise clearance | `voids.clearance.sample_clearance` | accessible fraction and clearance distribution without a grid; no surface | `n=8192`, `sampler='sobol'` (scrambled, n a power of two) or `'random'`, `seed=0` | `sampling_convergence` |
| Per atom | `voids.sites.atom_free_volume`, `cavity_distance` | accessible volume of each atom (grid points go to the atom with the nearest surface); distance to the nearest pore centre | the radii of the clearance field | free volumes sum to the accessible volume |

Percolation is a loop with a net lattice translation; an isolated pocket never percolates, but counts in the accessible fraction.
When `exact` is True, Di, Df and Dif are exact; otherwise Di and Df are lower bounds and percolation can stay undecided:
`percolates` None, `pore_network` `decided` False, and an empty `percolating_edges` mask proves nothing. `verify=False` is never
exact. Df and Dif are NaN unless a loop percolates.

## Radii

| Convention | `Radii` | Measures |
|---|---|---|
| centre | `Radii.centre(symbols)` | distance to atom centres |
| covalent | `Radii.covalent(symbols)` (Cordero) | space outside the bonded network |
| ionic | `Radii.ionic(symbols)` (Shannon) | space between ions, one charge and coordination per species (`vfa/data/README.md`) |
| your own | `Radii(table, name)`, `Radii.load(path)` | as you define |

Rankings between materials can change with the convention: record `name` with the results.

## Bond graph

`structure.graph.bond_graph(symbols, positions, cell, cutoffs)` bonds only the pairs given a cutoff. Edges carry the image offset
of the second atom, so a bond across the boundary is one bond with its exact vector.

`structure.rdf.first_minimum(r, g)` measures a cutoff on an ensemble pair correlation (`ensemble_rdf`): defaults `search=(1.0,
4.5)`, `smoothing_A=0.1` (widen it for one cell), `prominence=0.25`. `basin_middle_A` is the middle of the basin where g lies
within 0.1 |g_min| + 0.01 of its minimum (`basin_tolerance=1.1`, `basin_floor=0.01`). Check `resolved` and a plateau of
`running_coordination` at the cutoff.

## Rings

| Criterion | `structure.rings.rings(criterion=...)` | A ring is |
|---|---|---|
| primitive (default) | `'primitive'` (Franzblau) | a cycle with no shortcut: around-the-ring distance equals the graph distance for every pair |
| King | `'king'` | per atom and pair of its neighbours, the shortest paths between them avoiding the atom |
| Guttman | `'guttman'` | per bond, the shortest paths between its atoms avoiding the bond |

`max_length` caps the ring size in atoms and returns exactly the rings up to it; `rings_converged` raises the cap (16 by 4 to 48)
until the size histogram repeats at a cap of at least twice the largest ring. `bipartite=True` (default) raises on a homopolar
bond or an odd cycle, and `ring_sizes` then counts network-former units (half the atoms): pass the same `bipartite` to both.
`images=True` returns image offsets, with which `ring_extents` is exact; a ring as wide as the shortest lattice vector is a
finite-size artefact.

## Pair correlations and local geometry

`partial_rdf(symbols, positions, cell, pair)` normalises by the density of the second species, up to half the shortest lattice
vector (default and limit), 1000 bins. `ensemble_rdf` is the mean over cells, each weighted equally; `running_coordination` the
mean number of the second species within r. `angles`, `tetrahedral_order` (four shortest bonds; NaN below four) and `planarity`
(three-bonded atoms) take the bond vectors.

## Total scattering

| Quantity | Function | Defaults and options |
|---|---|---|
| Faber-Ziman partials, direct | `structure.scattering.direct_partials(..., q_max)`, `ensemble_direct` | the reciprocal-lattice vectors in shells of `dq=0.05`; exact for the cell, none below 2 pi/L |
| Faber-Ziman partials, transform | `transform_partials(..., q)` | sine transform of g - 1 up to half the shortest lattice vector; `window=None` or `'lorch'` |
| total F(Q), S(Q) | `total_sq(q, partials, conc, weights)` | weights: scattering lengths, or callables f(Q); no tables ship |
| real space | `total_gr`, `real_space` | Keen's G(r), not the reduced G(r) of PDF programs |
| first sharp diffraction peak | `fsdp(q, S, window)` | position, height, height above a straight baseline and FWHM inside the window |

The direct route is the reference; the transform smooths features narrower than about 2 pi/r_max. One cell rarely resolves a first
sharp diffraction peak: average cells with `ensemble_direct`.

## Electronic spectra

| Function | Defaults | Gives |
|---|---|---|
| `occupied_count` | `threshold=0.5` | occupied states of one spin channel |
| `character` | | each state's fraction of weight on one species |
| `semicore_reference`, `align` | `min_gap=1.5` | the centroid of the deep band below the one gap wider than `min_gap` in the lower half of the occupied states; (0, NaN) unless there is exactly one |
| `band_ladder` | `ks=(0, 1, 2, 3, 5, 10)` | gap to the k-th empty state; a defect level is a step between k = 0 and 1 that larger k do not repeat |
| `tail_spread` | `n_tail=5` | spread and mean spacing of the first empty states |
| `bulk_character` | `window=(20, 120)` | median character of states 20 to 119 from the gap on each side |
| `trap_level` | `window=30` | among the 30 highest occupied states, the one with the largest excess character over the valence band; `margin` to the runner-up |
| `delocalisation_onset`, `internal_depth` | `fractions=(5, 4, 3)` | the states nearest the gap spread over at least 1/5, 1/4, 1/3 of the atoms; the depth below that onset |
| `energies.trapping_energies` | `in_hartree=True` | uncorrected addition, vertical addition and relaxation energies (eV) |
| `mulliken.track` | | per step, the atom with the largest moment, its share and the participation |

Energies must be finite and ascending (all but `align`). Ties go to the first index.

## Finite-size corrections

`corrections.gates.gate_pair(charged, neutral)` lists the reasons to reject two `read_output` records, empty when they pass: a run
not ended or not converged, a charged spin outside `unpaired` - 0.5 to `unpaired` + 1.0 (`unpaired=1`, or `spin_window`), a
neutral spin above 0.5, and, when given, an energy missing or beyond 5e-4 Ha of `energy_reference` (`{'charged': Ha, 'neutral':
Ha}`), `geometry_identical=False` or `grids_match=False`. A spin not printed is not checked; geometry and grids are yours to
check.

`corrections.schemes.summarise(potential_json, density_json)` reads the JSON of
[cp2k-charge-corrections](https://github.com/Olsson-Materials-Modelling/cp2k-charge-corrections) 0.8.0 (schema 4.0) into one row:
each mode's selected correction, reliability and level shift, and the schemes FNV, LZ, PC and site (their `concerted_` forms at a
relaxed geometry; None when not written). `schemes={name: column}` replaces these four. Files from different corrector versions,
geometries or screening are refused, as are cubes whose last two directory names differ (`override=True` accepts them).

`corrected_addition(e_add, row)` returns E(q) - E(0) plus the density mode's selected correction, both cells relaxed in their
charge; another declared geometry needs `override=True`, an undeclared one is accepted. It does not check reliability: read
`density_reliable` and `density_status`. `transition_levels({q: formation energy at a Fermi level of zero})` returns the levels
from the highest charge down, also outside the gap. How each correction is defined: [the corrector's
conventions](https://github.com/Olsson-Materials-Modelling/cp2k-charge-corrections/blob/v0.8.0/docs/conventions.md).

## Statistics

The cell is the sampling unit: reduce site, bond, ring and state quantities to one value per cell first. Defaults: `alpha=0.05`
(`wilson`: `z=1.96`), `seed=0`.

| Function | Defaults | Gives |
|---|---|---|
| `stats.intervals.percentile_bootstrap` | `statistic=np.mean`, 10000 replicates | estimate and percentile interval under whole-cell resampling |
| `exact_lattice_bootstrap` | `max_grid=2**24` | the exact interval of a mean of counts or indicators; None off a lattice or above `max_grid` |
| `bounded_prevalence` | `sites=None` | interval of a mean share: Clopper-Pearson over the cells, or with `sites` Korn-Graubard over the sites |
| `wilson`, `clopper_pearson` | | interval of a proportion |
| `influence_mean`, `maxt_simultaneous` | 10000 replicates, `groups=None` | a band covering a family of estimates at once; members estimated on separate groups of cells need `groups` (four cells or more each) |
| `stats.contrasts.cliff_delta_interval`, `scale_ratio`, `location_shift` | 1000, 4000 replicates | Cliff's delta; ratio of standard deviations with the Brown-Forsythe p; shift in reference standard deviations |
| `stats.trends.smooth_in_order`, `held_out_window` | | local-linear smoothing in cell order with Gaussian weights; the width by leave-one-out prediction (`edge`: at the end of the grid) |
| `stats.trends.linear_change` | 2000 replicates | change between two positions by Theil-Sen and least squares |
| `stats.models.held_out_shares` | 400 replicates | class shares predicted for test cells by a model fitted on training cells; the interval holds the model fixed |

Percentile intervals undercover at few cells.

## Provenance

`RunContext().with_config(config).with_inputs(paths)` and the writers put a `<file>.meta.json` sidecar beside every table: the
code fingerprint (modules and `vfa/data`), the inputs hash (file names and contents), the configuration in full and hashed,
versions and platform. vfa adds no absolute path. Parquet needs pyarrow.
