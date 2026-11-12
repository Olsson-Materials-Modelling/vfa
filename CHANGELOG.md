# Changelog

## 1.1.0

- Corrections: reads cp2k-charge-corrections 0.8.0 output (schema 4.0) only; `summarise` options are keyword-only; the comparison
  schemes are the `concerted_` ones in a relaxed cell; `transition_levels` added.
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
