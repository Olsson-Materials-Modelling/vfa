# Example data

Synthetic network glasses, no real material: 30 cells (`cells.csv`), 5 at each of 6 number densities, each of 68 A and 136 B
atoms. They are grown to a target of four B neighbours per A atom (three for 2 A atoms per cell) and two A neighbours per B atom,
bond 1.62 A, with no two A atoms sharing two B atoms, and relaxed under bond, angle and repulsion springs. `radii.json` and
`radii_ionic.json` are the two radii conventions.

`spectra/<cell>.npz`: a neutral and a charged spectrum (energies, occupations, species weights, participation). The trap atom is
the A atom with the largest free volume (probe 0.8 A, grid 0.3 A, `radii.json`). The added electron occupies an A-character gap
state 1.0 eV + 0.04 eV/A3 x (free volume at the trap atom) + N(0, 0.1 eV) below the lowest conduction state; before trapping an
empty A-character state sits 0.3 eV below that edge. The gap narrows with density. `events.csv`: the free volume at the trap atom,
the three total energies, and the kind (deep above 2.0 eV, else shallow). `corrections/`: synthetic schema-4.0 corrector JSON of
each charged cell, relaxed in its charge, with only the fields vfa reads.

`python example/make_example.py` regenerates everything (seeds 11 and 12), bit for bit with numpy 2.4 and scipy 1.17; under other
versions the structure minimisation can take another path. `python example/make_example.py electronic` regenerates the spectra,
`events.csv`, `corrections/` and this file from the shipped structures.
