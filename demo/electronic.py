#!/usr/bin/env python3
"""Spectrum readings, trapping energies and corrections of every example cell, with provenance sidecars (demo/out)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from vfa.electronic.spectrum import character, occupied_count, semicore_reference, align, band_ladder, tail_spread, bulk_character, trap_level, delocalisation_onset, internal_depth
from vfa.electronic.energies import trapping_energies
from vfa.corrections.schemes import summarise, corrected_addition
from vfa.provenance import RunContext, write_csv, write_sidecar

DATA, OUT = ROOT / 'example/data', ROOT / 'demo/out'; FRACTIONS = (5, 4, 3)


def read(name, tag):
    """Energies, occupations, A character and participation of one spectrum of one cell."""
    z = np.load(DATA / 'spectra' / f'{name}.npz'); E, occ, at = z[f'{tag}_energy_eV'], z[f'{tag}_occupation'], z[f'{tag}_atoms']; f = character({'A': z[f'{tag}_weight_A'], 'B': z[f'{tag}_weight_B']}, 'A'); return E, occ, f, at


def main():
    """Spectrum readings, energies and corrections of every cell, written with sidecars."""
    ev = pd.read_csv(DATA / 'events.csv').set_index('cell'); meta = pd.read_csv(DATA / 'cells.csv').set_index('cell'); OUT.mkdir(exist_ok=True); rows = []
    for name in ev.index:
        E0, o0, f0, a0 = read(name, 'neutral'); E1, o1, f1, a1 = read(name, 'charged'); n0, n1 = occupied_count(o0), occupied_count(o1); n_atoms = int(meta.n_atoms[name])
        k0, r0 = semicore_reference(E0, n0); _, r1 = semicore_reference(E1, n1); A0, A1 = align(E0, r0), align(E1, r1)
        lad = band_ladder(A0, n0); bc = bulk_character(f0, n0); trap = trap_level(A1, f1, n1, bc['valence']); on = delocalisation_onset(A0, a0, n0, n_atoms, FRACTIONS); dep = internal_depth(A1, a1, n1, n_atoms, FRACTIONS)
        en = trapping_energies(ev.e_neutral_Ha[name], ev.e_charged_relaxed_Ha[name], ev.e_charged_vertical_Ha[name])
        cr = summarise(DATA / 'corrections' / f'{name}.relaxed.potential.json', DATA / 'corrections' / f'{name}.relaxed.density.json')
        row = dict(cell=name, number_density=meta.number_density[name], kind=ev.kind[name], free_volume_trap_A3=ev.free_volume_trap_A3[name], semicore_states=k0, semicore_shift_eV=r1 - r0, gap_k0_eV=lad[0]['gap_eV'], gap_k5_eV=lad[5]['gap_eV'], tail_spacing_eV=tail_spread(A0, n0)['spacing_eV'],
                   precursor_eV=A0[n0], precursor_atoms=a0[n0], trap_eV=trap['energy_eV'], trap_character=trap['character'], trap_margin=trap['margin'], trap_is_homo=trap['is_homo'], trap_below_lumo_eV=lad[0]['empty_eV'] - trap['energy_eV'],
                   relaxation_eV=en['relaxation_eV'], addition_eV=en['addition_eV'], correction_eV=cr['density_selected_eV'], correction_reliable=cr['density_reliable'], addition_corrected_eV=corrected_addition(en['addition_eV'], cr), level_shift_eV=cr['density_level_shift_eV'])
        row.update({f'onset_{fr}_above_lumo_eV': on[f'conduction_onset_{fr}']['above_lumo_eV'] for fr in FRACTIONS}); row.update({f'depth_{fr}_eV': dep[f'depth_{fr}_eV'] for fr in FRACTIONS}); rows.append(row)
        if name == ev.index[len(ev) // 2]: np.savez_compressed(OUT / 'spectrum_example.npz', E0=A0, f0=f0, a0=a0, n0=n0, E1=A1, f1=f1, a1=a1, n1=n1, trap=trap['index'], onset5=on['conduction_onset_5']['index'], n_atoms=n_atoms)
    ctx = RunContext().with_config(dict(fractions=FRACTIONS)).with_inputs(sorted((DATA / 'spectra').glob('*.npz')) + sorted((DATA / 'corrections').glob('*.json')) + [DATA / 'events.csv', DATA / 'cells.csv'])
    d = pd.DataFrame(rows); write_csv(d, OUT / 'events.csv', ctx, float_format=None); write_sidecar(OUT / 'spectrum_example.npz', ctx); pd.set_option('display.width', 220)
    print(d.groupby('number_density')[['gap_k0_eV', 'gap_k5_eV', 'trap_below_lumo_eV', 'depth_5_eV', 'relaxation_eV', 'addition_corrected_eV']].mean().round(3).to_string()); print('trap is the highest occupied state in', int(d.trap_is_homo.sum()), 'of', len(d), 'cells; semicore alignment shifts', d.semicore_shift_eV.abs().mean().round(3), 'eV on average')


if __name__ == '__main__':
    main()
