# Conventions

Every quantity below can be defined in more than one way. Each function states the convention it implements; this page lists the
alternatives, what each assumes, where it fails on an amorphous cell, and the check that shows whether a result depends on the choice.

## Void space

| Mode | Function | Gives | Assumes | Fails when | Check |
|---|---|---|---|---|---|
| Voronoi network | `voids.network.voronoi_network`, `descriptors` | largest included sphere, largest free sphere (percolation), largest included on the free path, pores and throats | the Voronoi vertices of the atom centres are the cavity centres (exact for equal radii; with unequal radii the vertex is a stationary point of the clearance, not its maximum, so Di is a lower bound by at most the radius spread) | the pore partition depends on `merge_frac` | report Di/Df at two `merge_frac`; the closed-form lattices in `tests/` |
| Grid clearance | `voids.clearance.clearance_field`, `probe_sweep`, `void_surface` | accessible fraction and volume at a probe radius, the void surface, the clearance distribution | the field is resolved at the spacing `dx` | thin channels narrower than `dx`; surfaces at the grid scale | `grid_convergence` at half the spacing |
| Pointwise clearance | `voids.clearance.sample_clearance` | the same distribution and accessible fraction without a grid, no surface | enough points for the smallest feature | the largest clearance of a sample underestimates Di | `sampling_convergence` at twice the points |
| Per atom | `voids.sites.atom_free_volume`, `cavity_distance` | the accessible volume nearest to each atom; the distance to the nearest pore centre | the pore network at the chosen probe | a probe larger than the local channels empties the partition | sum of the free volumes equals the accessible volume |
| Reference crystal | `descriptors` on a crystal cell | the intrinsic interstice of the composition: voids at or below it are packing, not porosity | the crystal chosen is not itself channelled or a vacancy-ordered phase | a high-pressure or channelled polymorph gives a wrong zero | Df/Di of the reference is small |

Percolation is a loop with a net lattice translation in the periodic network, not a path across the box; a large isolated pocket
never percolates. The accessible fraction counts isolated cavities; connectivity is the network's business. The clearance is
exact for unequal radii: the neighbour search widens until no omitted centre can be closer in surface distance. Both modes use
the cell and its 26 neighbouring images, so a strongly skewed cell is reduced first; Sobol counts are powers of two.

## Radii

| Convention | `Radii` | Measures | Caveat |
|---|---|---|---|
| centre | `Radii.centre` | distance to atom centres | not porosity; the convention of the closed-form checks |
| covalent | `Radii.covalent` (Cordero) | space outside the bonded network | bonding radii, not exclusion radii: an oxide or nitride is mostly "void" |
| ionic | `Radii.ionic` (Shannon) | interstitial space between ions | depends on charge state and coordination; the shipped entries hold for one state each |

Rankings between materials can change with the convention. A table carries the convention in its name or its sidecar.

## Bond graph

| Mode | Function | Caveat |
|---|---|---|
| declared cutoff per pair | `structure.graph.bond_graph` with a cutoff | the cutoff is a claim about the material; state where it came from |
| measured cutoff | `structure.rdf.first_minimum` on the ensemble pair correlation | a wide basin (`basin_A`) means the first minimum is not sharp; check a result at both ends of the basin; delta peaks of a lattice need a window on the first shell |

Only the pairs given a cutoff bond. Edges carry the image offset of the second atom, so bonds across the boundary are single bonds
with exact vectors.

## Rings

| Criterion | Function | Definition | Caveat |
|---|---|---|---|
| primitive (Franzblau shortest-path) | `structure.rings.primitive_rings` | no shortcut: around-the-ring distance equals the graph distance for every pair | includes cycles no shortest path visits (skew hexagons of a cubic lattice) |
| King | `structure.rings.king_rings` | per atom and pair of its neighbours, the shortest paths between the neighbours avoiding the atom | counts non-primitive cycles (2x1 rectangles of a cubic lattice) |
| Guttman | `structure.rings.guttman_rings` | per bond, the shortest paths between its atoms avoiding the bond | the smallest set |

All three enumerate on the covering graph, so a closed walk returning to a periodic image (a helix) is never a ring. Every criterion
needs a length cap: `rings_converged` raises it until the count stops changing, which works when the cell is large compared with
the rings; in a small cell, rings as long as the box keep appearing and a declared cap is the honest choice. Ring size counts
network-former units on a bipartite graph (half the atoms) and atoms otherwise; say which. `ring_extents` gives each ring's
unwrapped extent: a ring whose extent reaches a cell length runs through the periodic images and is a finite-size artefact, so
the demo drops those and reports how many.

## Pair correlations and local geometry

`partial_rdf` normalises ordered pairs by the density of the second species with explicit image expansion; `ensemble_rdf` weights
cells equally. `tetrahedral_order` uses the four shortest bonds and is NaN below four; `planarity` is the angle sum at three-bonded
atoms. Descriptor distributions on declared grids are right-continuous empirical CDFs.

## Electronic spectra

| Reading | Function | Caveat |
|---|---|---|
| band edge | `band_ladder` | the lowest empty state of an amorphous cell is usually a tail state; a defect level is a step between k = 0 and 1 that is gone by k = 5 |
| tail control | `tail_spread` | a gap of the order of the empty-state spacing is a band, not a state in the gap |
| trap | `trap_level` | by species character, not by occupation: the highest occupied state of a charged cell need not be the trap |
| alignment | `semicore_reference`, `align` | eigenvalues of different calculations compare only on one scale; a deep isolated band of the same cell is the reference; a cell without exactly one such band is dropped, not misaligned |
| mobility edge | `delocalisation_onset` | a threshold on the participation is a choice: sweep 1/5, 1/4, 1/3 of the cell and report all |
| internal depth | `internal_depth` | inside one spectrum, no alignment; the conduction onset must be inside the window of states available |
| energies | `energies.trapping_energies` | the relaxation energy is alignment-free; the addition energies carry the reservoir zero and the periodic background and need a finite-size correction |

## Finite-size corrections

The corrector is `cp2k-charge-corrections` (external, public). `corrections.gates.gate_pair` rejects a pair before anything is read:
run ended, SCF converged by the outer verdict, spin inside a window, identical geometries, matching grids, energy against a reference.
`corrections.schemes.summarise` labels the schemes the corrector returns: potential mode (FNV, LZ, eFNV, PC) and density mode
(image interaction, with the charging alignment added). The two modes are alternative estimates and are never summed; a Kohn-Sham
shift attaches only to the matching orbital. The corrected relaxation energy is E_relax + C_vertical - C_relaxed.

## Statistics

| Quantity | Function | Convention |
|---|---|---|
| interval of a cell mean | `stats.intervals.percentile_bootstrap` | whole-cell resampling, PCG64 seed, left-inverse quantile |
| interval of a prevalence | `bounded_prevalence` | exact n-fold convolution when the cell statistic is lattice valued, percentile otherwise |
| a family of estimates | `maxt_simultaneous` | one multiplier per cell shared across the family; zero-variance members excluded and listed |
| a proportion | `wilson` | never the normal approximation |
| two populations | `contrasts.cliff_delta_interval`, `scale_ratio`, `location_shift` | distribution-free effect size; SD ratio with Brown-Forsythe; shift in units of the reference spread |
| trend with an ordering variable | `trends.smooth_in_order`, `held_out_window` | local-linear with Gaussian weights in cells; the width chosen by leave-one-out prediction |
| change between two positions | `trends.linear_change` | Theil-Sen and least squares side by side, intervals from the same resamples; not interchangeable |
| convergence with cells | `prefix_halfwidth` | uncorrected t half-width per prefix |
| held-out classification | `models.held_out_shares`, `learning_curve` | standardisation, fit and prediction inside the training cells |

The cell is the sampling unit everywhere: site, bond, ring and state quantities are reduced to one value per cell first.
