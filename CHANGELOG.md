# Changelog

## 1.1.1

- Corrections: `summarise` reads the scheme names of the vertical geometry.
- Files: extended-XYZ keys read whole, with escapes; `.xyz` folders (`structure_paths`); CP2K 2026 restarts; `read_output` reports
  unconverged empty states and unfinished optimisations, which the gate rejects; restricted PDOS curve files skipped; LAMMPS units
  and type labels from `write_data`; `write_csv` and `write_parquet` keep named index levels.
- Statistics: `held_out_shares` and `learning_curve` refuse pandas labels that disagree.

## 1.1.0

- Corrections: reads cp2k-charge-corrections 0.8.0 output (schema 4.0) only; `summarise` options are keyword-only;
  `corrected_formation_energy` and `transition_levels` added.
- Voids: the additively weighted Voronoi pore network, exact for unequal radii, with a completeness check (`verify`, `complete`);
  Dif on the percolating component.
- Structure: total scattering (`structure.scattering`); exact minimum image in any cell; rings with image offsets
  (`images=True`), placed exactly by `ring_extents`; species compared as strings in pair correlations and scattering.
- Files: `read_output` judges the last SCF run of the last program run (OT outer loop, CDFT, linear scaling); CP2K units as CP2K
  defines them; `read_lammps_data` needs the atom style.
- Statistics: `bounded_prevalence` by Clopper-Pearson, with `sites` by Korn and Graubard; `maxt_simultaneous` by a studentized
  bootstrap; `exact_lattice_bootstrap` on the values' common step.
- Removed: `corrected_relaxation`, `participation_from_weights`, `influence_ratio`, `planning_n` and the reader of corrector schema
  3.x.

## 1.0.0

First version.
