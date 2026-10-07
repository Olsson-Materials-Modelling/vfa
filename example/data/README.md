# Example data

Synthetic network glasses: 68 four-fold A atoms bridged by 136 two-fold B atoms, bond 1.62 A, no two A atoms sharing two B atoms, relaxed under bond, angle and repulsion springs at six number densities, five cells each (`cells.csv`); two A atoms per cell are three-fold by valence and the growth leaves further under-bonded atoms. `radii.json` and `radii_ionic.json` are the example's two radii conventions. No real material is represented.

`spectra/<cell>.npz`: a neutral and a charged spectrum per cell (energies, occupations, species weights, participation) with a localised state, empty in the gap before trapping and occupied below it after. `events.csv`: the trap atom and the three total energies. `corrections/`: corrector-style summaries of each charged endpoint. By construction the trap deepens with the free volume at its atom and the gap narrows with density.
