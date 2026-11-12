"""A synthetic spectrum with known structure, the energy cycle and the gates."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vfa.units import HARTREE_EV
from vfa.electronic.spectrum import occupied_count, semicore_reference, align, band_ladder, tail_spread, bulk_character, trap_level, delocalisation_onset, internal_depth
from vfa.electronic.energies import trapping_energies, closes_cycle
from vfa.electronic.mulliken import participation, carriers, track, moved
from vfa.corrections.gates import gate_pair


def spectrum(trap_depth=1.0):
    """Energies, occupations, A character, participation and occupied count of a synthetic spectrum with a trap above the valence band."""
    E = np.concatenate([np.linspace(-20, -18, 30), np.linspace(-6, 0, 170), [trap_depth], np.linspace(5, 9, 150)]); n_occ = 201
    f = np.concatenate([np.full(30, 0.5), np.full(170, 0.15), [0.6], np.full(150, 0.62)]); atoms = np.concatenate([np.full(30, 20.0), np.linspace(10, 60, 170), [4.0], np.linspace(8, 90, 150)])
    occ = np.r_[np.ones(n_occ), np.zeros(E.size - n_occ)]; return E, occ, f, atoms, n_occ


def test_reference_ladder_and_trap_by_character():
    E, occ, f, _, n_occ = spectrum(); assert occupied_count(occ) == n_occ; k, ref = semicore_reference(E, n_occ); assert k == 30 and abs(ref + 19) < 1e-9 and abs(align(E, ref)[0] + 1) < 1e-9
    lad = band_ladder(E, n_occ); assert abs(lad[0]['gap_eV'] - 4.0) < 1e-9 and lad[10]['gap_eV'] > lad[0]['gap_eV']; assert abs(tail_spread(E, n_occ)['spacing_eV'] - 4 / 149) < 1e-9
    bc = bulk_character(f, n_occ); assert abs(bc['valence'] - 0.15) < 1e-9 and abs(bc['conduction'] - 0.62) < 1e-9
    t = trap_level(E, f, n_occ, bc['valence']); assert t['index'] == n_occ - 1 and t['is_homo'] and abs(t['excess'] - 0.45) < 1e-9 and t['margin'] > 0.4
    f2 = f.copy(); f2[198], f2[200] = f[200], f[198]
    t2 = trap_level(E, f2, n_occ, bc['valence']); assert t2['index'] == 198 and not t2['is_homo'] and t2['levels_above'] == 2


def test_onsets_and_depth_follow_the_threshold():
    E, _, _, atoms, n_occ = spectrum(); o = delocalisation_onset(E, atoms, n_occ, 100)
    i5, i3 = o['conduction_onset_5']['index'], o['conduction_onset_3']['index']; assert i5 < i3 and atoms[i5] >= 20 and atoms[i5 - 1] < 20 and o['conduction_onset_5']['above_lumo_eV'] > 0
    d = internal_depth(E, atoms, n_occ, 100); assert abs(d['depth_5_eV'] - (E[i5] - 1.0)) < 1e-9 and d['depth_3_eV'] > d['depth_5_eV']
    assert abs(participation([0.25] * 4 + [0] * 6) - 4) < 1e-12


def test_energy_cycle_and_mulliken():
    r = trapping_energies(-100.0, -100.3, -100.25); assert closes_cycle(r) and abs(r['relaxation_eV'] - 0.05 * HARTREE_EV) < 1e-9
    m = np.zeros(10); m[3] = 0.8; m[4] = 0.2; assert abs(participation(m) - 1 / 0.68) < 1e-9 and list(carriers(m, 0.5)) == [3]
    blocks = [(np.array(['A'] * 10), np.zeros(10), m), (np.array(['A'] * 10), np.zeros(10), m[::-1])]; t = track(blocks); assert t[0]['top_atom'] == 3 and t[1]['top_atom'] == 6 and moved(blocks)


def test_gates():
    ok = dict(ended=True, converged=True, energy_Ha=-1.0, spin=1.0); q0 = dict(ended=True, converged=True, energy_Ha=-0.9, spin=0.0)
    assert gate_pair(ok, q0) == [] and gate_pair(dict(ok, spin=2.5), q0) == ['charged spin 2.500 outside (0.5, 2.0)'] and len(gate_pair(dict(ok, ended=False, converged=False), q0, energy_reference={'charged': -1.1})) == 3


def test_index_guards_raise():
    E, _, f, _, n_occ = spectrum()
    for n in (0, E.size):
        with pytest.raises(ValueError, match='band_ladder'): band_ladder(E, n)
    with pytest.raises(ValueError, match='tail_spread'): tail_spread(E, 0)
    assert np.isnan(tail_spread(E, E.size - 2)['spread_eV'])
    f15 = np.r_[np.full(15, 0.1), np.full(185, 0.9)]; f2 = np.r_[np.full(130, 0.1), np.full(15, 0.9)]
    for ff, n, w in ((f15, 15, (20, 120)), (f2, 130, (20, 120)), (f, n_occ, (30, 10)), (f, n_occ, (-1, 10))):
        with pytest.raises(ValueError, match='bulk_character'): bulk_character(ff, n, w)
    assert bulk_character(f15, 15, (0, 15)) == dict(valence=0.1, conduction=0.9) and bulk_character(f2, 130, (5, 15)) == dict(valence=0.1, conduction=0.9)
    with pytest.raises(ValueError, match='no state of the window'): bulk_character(np.full(300, np.nan), 150)
    with pytest.raises(ValueError, match='exceeds'): trap_level(E, f, E.size + 1, 0.15)


def test_ties_go_to_the_first_index():
    m = np.zeros(300); m[7] = 0.9; r = track([(np.array(['A'] * 300), np.zeros(300), m)])[0]; assert r['top_atom'] == 7 and r['second_atom'] == 0
    a = np.r_[np.zeros(40), np.full(30, 0.5), np.zeros(230)]; assert carriers(a, 0.5).tolist() == list(range(40, 55))
    E = np.linspace(-5, 0, 300); f = np.full(300, 0.2); f[[272, 273]] = 0.6; t = trap_level(E, f, 300, 0.2); assert t['index'] == 272 and t['margin'] == 0


def test_participation_counts_signed_weights_by_size():
    w = np.array([0.6, 0.6, -0.2]); want = 1.4 ** 2 / 0.76
    assert abs(participation(w) - want) < 1e-12 and abs(participation(-w) - want) < 1e-12 and np.isnan(participation(np.zeros(4)))


def test_moved_takes_any_iterable_and_gate_spin_window_follows_unpaired():
    m = np.zeros(10); m[3] = 0.8; blocks = [(np.array(['A'] * 10), np.zeros(10), m), (np.array(['A'] * 10), np.zeros(10), m[::-1])]
    assert moved(b for b in blocks) and moved(iter(blocks)) and not moved(blocks[:1])
    with pytest.raises(ValueError, match='moved'): moved(iter([]))
    q0 = dict(ended=True, converged=True, energy_Ha=-0.9, spin=0.0); ok = lambda s: dict(ended=True, converged=True, energy_Ha=-1.0, spin=s)
    assert gate_pair(ok(1.9), q0) == gate_pair(ok(1.9), q0, spin_window=(0.5, 2.0)) == [] and gate_pair(ok(0.4), q0) == ['charged spin 0.400 outside (0.5, 2.0)']
    assert gate_pair(ok(2.1), q0, unpaired=2) == [] and gate_pair(ok(1.1), q0, unpaired=2) == ['charged spin 1.100 outside (1.5, 3.0)'] and gate_pair(ok(2.1), q0, unpaired=2, spin_window=(0.5, 2.0)) == ['charged spin 2.100 outside (0.5, 2.0)']
    with pytest.raises(ValueError, match='spin window'): gate_pair(ok(1.0), q0, spin_window=(2.0, 0.5))


def test_track_a_single_atom():
    r = track([(np.array(['A']), np.array([0.2]), np.array([0.9]))])[0]; assert r['top_atom'] == 0 and r['second_atom'] == -1 and np.isnan(r['second_moment'])


def test_energies_out_of_order_are_refused():
    E, _, f, atoms, n_occ = spectrum(); E2 = E.copy(); E2[[198, 200]] = E2[[200, 198]]
    for call in (lambda: band_ladder(E2, n_occ), lambda: tail_spread(E2, n_occ), lambda: trap_level(E2, f, n_occ, 0.15), lambda: delocalisation_onset(E2, atoms, n_occ, 100), lambda: internal_depth(E2, atoms, n_occ, 100), lambda: semicore_reference(E2, n_occ)):
        with pytest.raises(ValueError, match='ascending'): call()


def test_empty_moments_raise():
    with pytest.raises(ValueError, match='carriers'): carriers(np.zeros(3))
    with pytest.raises(ValueError, match='track'): track([(np.array([]), np.array([]), np.array([]))])


def test_degenerate_energies_pass_and_nan_energies_raise():
    E, _, _, _, n_occ = spectrum(); D = E.copy(); D[n_occ - 3] = D[n_occ - 4]; assert band_ladder(D, n_occ)[0]['gap_eV'] == band_ladder(E, n_occ)[0]['gap_eV']
    N = E.copy(); N[5] = np.nan
    with pytest.raises(ValueError, match='finite'): band_ladder(N, n_occ)
    for bad in (dict(moments=[np.nan, 1.0]), dict(moments=[1.0, 2.0], share=0.0), dict(moments=[1.0, 2.0], share=1.5)):
        with pytest.raises(ValueError, match='carriers'): carriers(**bad)
