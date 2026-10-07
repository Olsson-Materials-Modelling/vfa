"""A synthetic spectrum with known structure, the energy cycle, the gates and the scheme summary."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vfa.electronic.spectrum import character, occupied_count, semicore_reference, align, band_ladder, tail_spread, bulk_character, trap_level, delocalisation_onset, internal_depth, participation_from_weights
from vfa.electronic.energies import trapping_energies, closes_cycle
from vfa.electronic.mulliken import participation, carriers, track, moved
from vfa.corrections.gates import gate_pair
from vfa.corrections.schemes import summarise, corrected_relaxation


def spectrum(n_atoms=100, trap_depth=1.0):
    """Semicore band, valence band of species B, a gap, an occupied localised state of species A 1 eV above the valence band, conduction band of species A."""
    E = np.concatenate([np.linspace(-20, -18, 30), np.linspace(-6, 0, 170), [trap_depth], np.linspace(5, 9, 150)]); n_occ = 201
    f = np.concatenate([np.full(30, 0.5), np.full(170, 0.15), [0.6], np.full(150, 0.62)]); atoms = np.concatenate([np.full(30, 20.0), np.linspace(10, 60, 170), [4.0], np.linspace(8, 90, 150)])
    occ = np.r_[np.ones(n_occ), np.zeros(E.size - n_occ)]; return E, occ, f, atoms, n_occ


def test_reference_ladder_and_trap_by_character():
    E, occ, f, atoms, n_occ = spectrum(); assert occupied_count(occ) == n_occ; k, ref = semicore_reference(E, n_occ); assert k == 30 and abs(ref + 19) < 1e-9 and abs(align(E, ref)[0] + 1) < 1e-9
    lad = band_ladder(E, n_occ); assert abs(lad[0]['gap_eV'] - 4.0) < 1e-9 and lad[10]['gap_eV'] > lad[0]['gap_eV']; assert abs(tail_spread(E, n_occ)['spacing_eV'] - 4 / 149) < 1e-9
    bc = bulk_character(f, n_occ); assert abs(bc['valence'] - 0.15) < 1e-9 and abs(bc['conduction'] - 0.62) < 1e-9
    t = trap_level(E, f, n_occ, bc['valence']); assert t['index'] == n_occ - 1 and t['is_homo'] and abs(t['excess'] - 0.45) < 1e-9 and t['margin'] > 0.4
    E2 = E.copy(); E2[198], E2[200] = E[200], E[198]; f2 = f.copy(); f2[198], f2[200] = f[200], f[198]      # the trap two levels below the HOMO
    t2 = trap_level(E2, f2, n_occ, bc['valence']); assert t2['index'] == 198 and not t2['is_homo'] and t2['levels_above'] == 2


def test_onsets_and_depth_follow_the_threshold():
    E, occ, f, atoms, n_occ = spectrum(); o = delocalisation_onset(E, atoms, n_occ, 100)
    i5, i3 = o['conduction_onset_5']['index'], o['conduction_onset_3']['index']; assert i5 < i3 and atoms[i5] >= 20 and atoms[i5 - 1] < 20 and o['conduction_onset_5']['above_lumo_eV'] > 0
    d = internal_depth(E, atoms, n_occ, 100); assert abs(d['depth_5_eV'] - (E[i5] - 1.0)) < 1e-9 and d['depth_3_eV'] > d['depth_5_eV']
    assert abs(participation_from_weights([0.25] * 4 + [0] * 6) - 4) < 1e-12


def test_energy_cycle_and_mulliken():
    r = trapping_energies(-100.0, -100.3, -100.25); assert closes_cycle(r) and abs(r['relaxation_eV'] - 0.05 * 27.211386245988) < 1e-9
    m = np.zeros(10); m[3] = 0.8; m[4] = 0.2; assert abs(participation(m) - 1 / 0.68) < 1e-9 and list(carriers(m, 0.5)) == [3]
    blocks = [(np.array(['A'] * 10), np.zeros(10), m), (np.array(['A'] * 10), np.zeros(10), m[::-1])]; t = track(blocks); assert t[0]['top_atom'] == 3 and t[1]['top_atom'] == 6 and moved(blocks)


def test_gates_and_schemes(tmp_path):
    ok = dict(ended=True, converged=True, energy_Ha=-1.0, spin=1.0); q0 = dict(ended=True, converged=True, energy_Ha=-0.9, spin=0.0)
    assert gate_pair(ok, q0) == [] and gate_pair(dict(ok, spin=2.5), q0) == ['charged spin 2.500 outside (0.5, 2.0)'] and len(gate_pair(dict(ok, ended=False, converged=False), q0, energy_reference={'charged': -1.1})) == 3
    pot = {'schemes': {'FNV_planar_gaussian': {'E_corr_eV': 0.30, 'recommended': True, 'status': 'ok'}, 'PC_IIC': {'E_corr_eV': 0.25}}, 'ks': {'delta_epsilon_eV': -0.12, 'corrected_eigenvalue_eV': 1.0}}
    dens = {'energies': {'E_corr_eV': 0.25}, 'neutral_pipeline': {'components': {'CD': {'selected_minus_q_median_eV': 0.10, 'selected_quality': {'ok': True}}}}, 'state_corrections': [{'energy_expectation_eV': 0.2, 'eigenvalue_shift_status': 'ok', 'numerical_reliability': {'ok': True}}]}
    (tmp_path / 'p.json').write_text(json.dumps(pot)); (tmp_path / 'd.json').write_text(json.dumps(dens)); s = summarise(tmp_path / 'p.json', tmp_path / 'd.json')
    assert s['potential_scheme'] == 'FNV_planar_gaussian' and s['fnv_eV'] == 0.30 and s['pc_eV'] == 0.25 and s['ks_shift_eV'] == -0.12 and abs(s['density_corrected_eV'] - 0.35) < 1e-12 and s['state_reliable']
    assert abs(corrected_relaxation(1.0, 0.30, 0.35) - 0.95) < 1e-12
