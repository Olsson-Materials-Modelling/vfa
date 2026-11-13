"""The reader of the corrector's JSON, corrected energies and transition levels."""
import copy, json, sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vfa.corrections.schemes import summarise, corrected_formation_energy, transition_levels

DATA = Path(__file__).resolve().parent / 'data'
INPUTS = {'tool_name': 'cp2k_charge_corrections', 'code_version': '0.8.0', 'schema_version': '4.0'}
PICK = {'scheme': 'concerted_FNV_planar_gaussian', 'E_corr_eV': 0.18, 'reliability_ok': False, 'reasons': ['structured_far_residual'], 'impact_eV': 0.03, 'screening': 'static', 'ionic_geometry': 'relaxed', 'delta_epsilon_eV': 0.4, 'level_reliability_ok': False}
POT = {'inputs': INPUTS, 'selected': dict(PICK, mode='potential'), 'schemes': {'FNV_planar_gaussian': {'E_corr_eV': 0.3}, 'LZ_scaled_PC_plus_alignment': {'E_corr_eV': 0.25}, 'PC_IIC': {'E_corr_eV': 0.32},
       'concerted_FNV_planar_gaussian': {'E_corr_eV': 0.18}, 'concerted_LZ_scaled_PC_plus_alignment': {'E_corr_eV': 0.17}, 'concerted_PC_IIC': {'E_corr_eV': 0.21}, 'concerted_site_gaussian_median': {'E_corr_eV': 0.19}}}
DENS = {'inputs': INPUTS, 'selected': dict(PICK, mode='density', scheme='bare_carrier_image_plus_DA_model_alignment', E_corr_eV=0.19, reasons=['PA_axis_or_window_assessment_failed'], delta_epsilon_eV=None, carrier='localized'),
        'energies': {'selected_correction_status': 'applied_with_failed_checks'}}


def write(tmp_path, pot=POT, dens=DENS):
    """Paths of a potential and a density JSON written from the given dictionaries."""
    (tmp_path / 'p.json').write_text(json.dumps(pot)); (tmp_path / 'd.json').write_text(json.dumps(dens)); return tmp_path / 'p.json', tmp_path / 'd.json'


def real(leg):
    """The potential and density JSON written by cp2k-charge-corrections 0.8.0."""
    return DATA / f'corrector_{leg}.potential.json', DATA / f'corrector_{leg}.density.json'


def test_real_corrector_output_is_read_by_its_selected_block():
    for leg, geometry, screening in (('vertical', 'vertical', 'clamped'), ('relaxed', 'relaxed', 'static'), ('plain', None, 'as_given')):
        s = summarise(*real(leg)); p, d = (json.loads(f.read_text())['selected'] for f in real(leg))
        assert (s['schema_version'], s['code_version']) == ('4.0', '0.8.0') and s['density_selected_eV'] == d['E_corr_eV'] and s['potential_selected_eV'] == p['E_corr_eV'] and s['potential_scheme'] == p['scheme'] and s['density_scheme'] == d['scheme']
        assert (s['density_ionic_geometry'], s['density_screening'], s['potential_ionic_geometry'], s['potential_screening']) == (geometry, screening, geometry, screening) and s['density_carrier'] == d['carrier']
        sc = json.loads(real(leg)[0].read_text())['schemes']; pre = {'relaxed': 'concerted_', 'vertical': 'clamped_'}.get(geometry, '')
        assert s['fnv_eV'] == sc[pre + 'FNV_planar_gaussian']['E_corr_eV'] and s['pc_eV'] == sc[pre + 'PC_IIC']['E_corr_eV'] and s['lz_eV'] == sc[pre + 'LZ_scaled_PC_plus_alignment']['E_corr_eV'] and s['site_eV'] == sc.get(pre + 'site_gaussian_median', {}).get('E_corr_eV')
    cr = summarise(*real('relaxed')); assert corrected_formation_energy(-1.0, cr) == -1.0 + cr['density_selected_eV']


def test_selected_values_read_exactly(tmp_path):
    s = summarise(*write(tmp_path))
    want = dict(schema_version='4.0', code_version='0.8.0', density_scheme='bare_carrier_image_plus_DA_model_alignment', density_selected_eV=0.19, density_reliable=False, density_reasons=['PA_axis_or_window_assessment_failed'], density_impact_eV=0.03, density_screening='static', density_ionic_geometry='relaxed',
                density_level_shift_eV=None, density_level_reliable=None, density_carrier='localized', density_status='applied_with_failed_checks', potential_scheme='concerted_FNV_planar_gaussian', potential_selected_eV=0.18, potential_reliable=False, potential_reasons=['structured_far_residual'],
                potential_impact_eV=0.03, potential_screening='static', potential_ionic_geometry='relaxed', potential_level_shift_eV=0.4, potential_level_reliable=False, fnv_eV=0.18, lz_eV=0.17, pc_eV=0.21, site_eV=0.19)
    assert s == want and all(type(s[k]) is type(v) for k, v in want.items())
    with pytest.raises(TypeError): summarise(*write(tmp_path), None, True)
    d = summarise(None, write(tmp_path)[1]); assert d['density_selected_eV'] == 0.19 and 'fnv_eV' not in d and 'potential_scheme' not in d
    p = summarise(write(tmp_path)[0], None, schemes={'PC_IIC': 'pc'}); assert p['pc_eV'] == 0.32 and 'fnv_eV' not in p and 'density_selected_eV' not in p


def test_other_schemas_foreign_and_mixed_files_raise(tmp_path):
    for version in ('3.2', '3.3', '4.1', 'beta'):
        dens = copy.deepcopy(DENS); dens['inputs'] = dict(INPUTS, schema_version=version)
        with pytest.raises(ValueError, match=f"schema '{version}'; vfa reads schema 4.0"): summarise(None, write(tmp_path, POT, dens)[1])
    dens = copy.deepcopy(DENS); del dens['inputs']['schema_version']
    with pytest.raises(ValueError, match='inputs/schema_version'): summarise(None, write(tmp_path, POT, dens)[1])
    dens = copy.deepcopy(DENS); dens['inputs'] = dict(INPUTS, tool_name='other_corrector')
    with pytest.raises(ValueError, match='other_corrector'): summarise(None, write(tmp_path, POT, dens)[1])
    pot = copy.deepcopy(POT); pot['inputs'] = dict(INPUTS, code_version='0.8.1')
    with pytest.raises(ValueError, match='corrector 0.8.1'): summarise(*write(tmp_path, pot, DENS))
    with pytest.raises(ValueError, match='holds the density mode, given as the potential JSON'): summarise(real('relaxed')[1], real('relaxed')[1])
    with pytest.raises(ValueError, match='holds the potential mode, given as the density JSON'): summarise(None, real('relaxed')[0])


def test_missing_file_and_fields_raise(tmp_path):
    with pytest.raises(ValueError, match='no file'): summarise(tmp_path / 'none.potential.json', tmp_path / 'none.density.json')
    with pytest.raises(ValueError, match='neither'): summarise(None, None)
    for key in ('E_corr_eV', 'reliability_ok', 'reasons', 'screening', 'ionic_geometry', 'delta_epsilon_eV', 'carrier', 'mode'):
        dens = copy.deepcopy(DENS); del dens['selected'][key]
        with pytest.raises(ValueError, match=f'selected/{key}'): summarise(None, write(tmp_path, POT, dens)[1])
    dens = copy.deepcopy(DENS); del dens['energies']['selected_correction_status']
    with pytest.raises(ValueError, match='energies/selected_correction_status'): summarise(None, write(tmp_path, POT, dens)[1])
    pot = copy.deepcopy(POT); del pot['selected']['level_reliability_ok']
    with pytest.raises(ValueError, match='selected/level_reliability_ok'): summarise(*write(tmp_path, pot, DENS))
    pot['selected']['delta_epsilon_eV'] = None; assert summarise(*write(tmp_path, pot, DENS))['potential_level_reliable'] is None
    pot = copy.deepcopy(POT); del pot['schemes']['concerted_PC_IIC']['E_corr_eV']
    with pytest.raises(ValueError, match='schemes/concerted_PC_IIC/E_corr_eV'): summarise(*write(tmp_path, pot, DENS))


def row(selected, geometry, screening):
    """A summarise row of one leg."""
    return dict(density_selected_eV=selected, density_ionic_geometry=geometry, density_screening=screening, density_status='applied', density_reasons=[])


def test_corrected_formation_energy_takes_the_relaxed_cell_and_refuses_a_missing_density_correction():
    assert corrected_formation_energy(-1.5, row(0.19, 'relaxed', 'static')) == -1.5 + 0.19 and corrected_formation_energy(-1.5, row(0.25, None, 'as_given')) == -1.5 + 0.25
    with pytest.raises(ValueError, match='vertical geometry'): corrected_formation_energy(-1.5, row(0.3, 'vertical', 'clamped'))
    assert corrected_formation_energy(-1.5, row(0.3, 'vertical', 'clamped'), override=True) == -1.5 + 0.3
    nan = float('nan'); potential_only = dict(summarise(real('relaxed')[0], None), **{k: nan for k in row(0, 0, 0)})
    for r, why in ((row(None, 'relaxed', 'static'), 'no selected density correction'), (row(nan, 'relaxed', 'static'), 'no selected density correction'), (summarise(real('relaxed')[0], None), 'read without a density JSON'), (potential_only, 'read without a density JSON')):
        for override in (False, True):
            with pytest.raises(ValueError, match=why): corrected_formation_energy(-1.0, r, override=override)


def test_transition_levels_follow_the_lowest_formation_energy():
    for f, want in (({0: 0.0, 1: -2.0, 2: -3.5}, [((2, 1), 1.5), ((1, 0), 2.0)]), ({0: 0.0, 1: -1.0, 2: -3.0}, [((2, 0), 1.5)]), ({-1: 2.0, 0: 0.0, 1: -1.0}, [((1, 0), 1.0), ((0, -1), 2.0)])):
        assert [(t['charges'], t['level_eV']) for t in transition_levels(f)] == want
    for f in ({0: 0.0, 1: -1.0, 2: -2.0}, {0: 0.0, 1: -1.0 + 1e-13, 2: -2.0}): assert [(t['charges'], t['level_eV']) for t in transition_levels(f)] == [((2, 0), 1.0)]
    for bad in ({0: 0.0}, {0: 0.0, 1: float('nan')}, {0: 0.0, 1: None}):
        with pytest.raises(ValueError, match='transition_levels'): transition_levels(bad)


def test_potential_and_density_files_of_different_legs_are_refused_even_with_override(tmp_path):
    pot = copy.deepcopy(POT); pot['selected'].update(ionic_geometry='vertical', screening='clamped')
    with pytest.raises(ValueError, match='vertical geometry with clamped screening, the density JSON for the relaxed one with static'): summarise(*write(tmp_path, pot, DENS), override=True)
    pot['selected'].update(ionic_geometry=None, screening='as_given')
    with pytest.raises(ValueError, match='undeclared geometry with as_given screening'): summarise(*write(tmp_path, pot, DENS), override=True)
    with pytest.raises(ValueError, match='different runs'): summarise(real('vertical')[0], real('relaxed')[1])


def test_files_of_different_runs_are_refused_unless_overridden(tmp_path):
    def cubes(base, kind, cell):
        js = copy.deepcopy(base); names = ('defect_cube', 'bulk_cube') if kind == 'v' else ('rho_charged_cube', 'rho_neutral_cube'); root = 'runs' if kind == 'v' else 'copy/runs'; js['inputs'] = dict(INPUTS, **{n: f'{root}/relaxed_{state}/{cell}/{kind}.cube' for n, state in zip(names, ('charged', 'neutral'))}); return js
    p, same = write(tmp_path, cubes(POT, 'v', 'rho0.060_2'), cubes(DENS, 'rho', 'rho0.060_2')); assert summarise(p, same)['density_selected_eV'] == 0.19
    p, other = write(tmp_path, cubes(POT, 'v', 'rho0.060_2'), cubes(DENS, 'rho', 'rho0.060_1'))
    with pytest.raises(ValueError, match='different runs'): summarise(p, other)
    assert summarise(p, other, override=True)['density_selected_eV'] == 0.19


def test_each_geometry_reads_the_corrector_names_of_that_geometry(tmp_path):
    names, prefixes = ('FNV_planar_gaussian', 'LZ_scaled_PC_plus_alignment', 'PC_IIC', 'site_gaussian_median'), ('', 'clamped_', 'concerted_')
    for geometry, screening, prefix in ((None, 'as_given', ''), ('vertical', 'clamped', 'clamped_'), ('relaxed', 'static', 'concerted_')):
        pot, dens = copy.deepcopy(POT), copy.deepcopy(DENS); pot['schemes'] = {p + n: {'E_corr_eV': i + j / 10} for i, p in enumerate(prefixes) for j, n in enumerate(names)}
        for js in (pot, dens): js['selected'].update(ionic_geometry=geometry, screening=screening)
        s = summarise(*write(tmp_path, pot, dens)); i = prefixes.index(prefix)
        assert [s[c] for c in ('fnv_eV', 'lz_eV', 'pc_eV', 'site_eV')] == [i + j / 10 for j in range(4)]


def test_schemes_not_run_read_none_unless_asked_for(tmp_path):
    pot = copy.deepcopy(POT); del pot['schemes']['concerted_LZ_scaled_PC_plus_alignment']; p, d = write(tmp_path, pot)
    assert summarise(p, d)['lz_eV'] is None and summarise(p, d)['fnv_eV'] == 0.18
    with pytest.raises(ValueError, match='LZ_scaled_PC_only'): summarise(p, d, schemes={'LZ_scaled_PC_only': 'lz'})


def test_run_directories_compare_on_their_shared_tail(tmp_path):
    def cubes(base, kind, charged, neutral):
        js = copy.deepcopy(base); names = ('defect_cube', 'bulk_cube') if kind == 'v' else ('rho_charged_cube', 'rho_neutral_cube'); js['inputs'] = dict(INPUTS, **dict(zip(names, (charged, neutral)))); return js
    pot = cubes(POT, 'v', 'runs/relaxed_charged/rho0.060_2/v.cube', 'runs/relaxed_neutral/rho0.060_2/v.cube')
    for c, n in (('C:\\copy\\relaxed_charged\\rho0.060_2\\rho.cube', 'C:\\copy\\relaxed_neutral\\rho0.060_2\\rho.cube'), ('rho0.060_2/./rho.cube', 'rho0.060_2/rho.cube'), ('rho.cube', 'rho.cube')):
        assert summarise(*write(tmp_path, pot, cubes(DENS, 'rho', c, n)))['density_selected_eV'] == 0.19
    with pytest.raises(ValueError, match='different runs'): summarise(*write(tmp_path, pot, cubes(DENS, 'rho', 'rho0.060_1/rho.cube', 'rho0.060_2/rho.cube')))


def test_run_guard_normalises_paths_and_compares_both_names(tmp_path):
    def cubes(base, kind, charged):
        js = copy.deepcopy(base); names = ('defect_cube', 'bulk_cube') if kind == 'v' else ('rho_charged_cube', 'rho_neutral_cube'); js['inputs'] = dict(INPUTS, **dict(zip(names, (charged, 'n.cube')))); return js
    pot = cubes(POT, 'v', 'runs/relaxed_charged/rho0.060_2/v.cube')
    for same in ('runs/relaxed_charged/x/../rho0.060_2/rho.cube', '../rho.cube', 'C:\\runs\\relaxed_charged\\rho0.060_2\\rho.cube'):
        assert summarise(*write(tmp_path, pot, cubes(DENS, 'rho', same)))['density_selected_eV'] == 0.19
    for other in ('runs/vertical_charged/rho0.060_2/rho.cube', 'runs/vertical_charged/x/../rho0.060_2/rho.cube', 'C:\\runs\\relaxed_charged\\rho0.060_1\\rho.cube'):
        with pytest.raises(ValueError, match='different runs'): summarise(*write(tmp_path, pot, cubes(DENS, 'rho', other)))


def test_unreadable_json_and_scheme_lists_raise(tmp_path):
    bad = tmp_path / 'b.json'; bad.write_text('{"inputs": ')
    with pytest.raises(ValueError, match='is not valid JSON'): summarise(None, bad)
    with pytest.raises(ValueError, match='schemes must map'): summarise(*write(tmp_path), schemes=['PC_IIC'])


def test_two_charges_give_one_transition_level():
    assert transition_levels({0: 0.0, -1: 1.3}) == [dict(charges=(0.0, -1.0), level_eV=1.3)] and transition_levels({1: -0.4, 0: 0.0}) == [dict(charges=(1.0, 0.0), level_eV=0.4)]


def test_a_charge_lowest_within_1e_13_of_a_crossing_is_a_tie_and_within_1e_9_is_not():
    t = transition_levels({0: 0.0, 1: -1.0 - 1e-13, 2: -2.0}); assert [x['charges'] for x in t] == [(2, 0)] and abs(t[0]['level_eV'] - 1.0) < 1e-12
    t = transition_levels({0: 0.0, 1: -1.0 - 1e-9, 2: -2.0}); assert [x['charges'] for x in t] == [(2, 1), (1, 0)] and abs(t[0]['level_eV'] - (1 - 1e-9)) < 1e-14 and abs(t[1]['level_eV'] - (1 + 1e-9)) < 1e-14


def test_corrected_formation_energy_takes_override_by_keyword_only():
    r = dict(density_selected_eV=0.3, density_ionic_geometry='vertical', density_screening='clamped', density_status='applied', density_reasons=[])
    with pytest.raises(TypeError): corrected_formation_energy(-1.5, r, True)
    assert corrected_formation_energy(-1.5, r, override=True) == -1.5 + 0.3
