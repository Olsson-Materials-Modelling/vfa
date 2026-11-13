"""The readers of vfa.io on small synthetic files in tests/data, and the pair gate on read_output records."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vfa import units
from vfa.io.cp2k import read_restart, read_output, read_pdos_kind, read_spectrum, mulliken_blocks, trajectory_energies, cube_header
from vfa.io.extxyz import read_extxyz, write_extxyz, read_structure_files, structure_paths
from vfa.io.lammps import read_lammps_data
from vfa.corrections.gates import gate_pair

D = Path(__file__).resolve().parent / 'data'; XYZ = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], float)
EMPTY = dict(ended=False, converged=None, energy_Ha=None, spin=None, unoccupied_converged=None, optimised=None)


def test_read_output_judges_the_last_scf_by_its_own_criterion(tmp_path):
    cases = {'cp2k_sp_ot.out': True, 'cp2k_sp_ot_inner_short.out': True, 'cp2k_sp_ot_failed.out': False, 'cp2k_sp_diag.out': True, 'cp2k_sp_diag_abort.out': False,
             'cp2k_geo_last_failed.out': False, 'cp2k_geo_last_failed_outer.out': False, 'cp2k_geo_recovered.out': True, 'cp2k_killed.out': None}
    for name, want in cases.items(): assert read_output(str(D / name))['converged'] is want, name
    con = tmp_path / 'constrained.out'; con.write_text((D / 'cp2k_sp_ot_inner_short.out').read_text().replace('No variables optimised in outer loop', 'DDAPC constraint enforced'))
    assert read_output(con)['converged'] is False and set(read_output(D / 'cp2k_sp_ot.out')) == set(EMPTY)
    r = read_output(D / 'cp2k_sp_ot.out'); assert r['ended'] and r['energy_Ha'] == -100.01 and r['spin'] == 1.05
    assert not read_output(D / 'cp2k_sp_diag_abort.out')['ended'] and read_output(D / 'cp2k_sp_diag.out')['spin'] is None and read_output(D / 'cp2k_geo_last_failed.out')['energy_Ha'] == -100.02


def test_read_output_takes_the_cdft_verdict_and_the_last_program_run(tmp_path):
    start, end = '  **** **** ******  **  PROGRAM STARTED AT               2026-01-01 00:00:00.000\n', ' ENERGY| Total FORCE_EVAL ( QS ) energy [a.u.]:             -100.030000000000\n  **** **** ******  **  PROGRAM ENDED AT\n'
    scf = ' SCF WAVEFUNCTION OPTIMIZATION\n\n  *** SCF run converged in     1 steps ***\n'; loop = ' CDFT EXTERNAL SCF WAVEFUNCTION OPTIMIZATION\n' + scf + '  CDFT SCF iter =     1 RMS gradient =   1.00E-02 energy =      -100.0300000000\n' + scf
    ok, failed = '  CDFT SCF loop converged in   2 iterations or    2 steps\n', '  CDFT SCF loop FAILED to converge after   2 iterations or    2 steps\n'
    for name, text, want in (('ok', loop + ok + end, True), ('failed', loop + failed + end, False), ('killed', loop, None), ('then_plain', loop + failed + scf + end, True), ('ok_then_killed', loop + ok + ' CDFT EXTERNAL SCF WAVEFUNCTION OPTIMIZATION\n' + scf, None)):
        p = tmp_path / f'cdft_{name}.out'; p.write_text(start + text); assert read_output(p)['converged'] is want, name
    ot, con = (D / 'cp2k_sp_ot.out').read_text(), (D / 'cp2k_sp_ot_inner_short.out').read_text().replace('No variables optimised in outer loop', 'DDAPC constraint enforced')
    for name, text, want in (('killed', (D / 'cp2k_killed.out').read_text(), EMPTY), ('constrained', con, dict(read_output(D / 'cp2k_sp_ot_inner_short.out'), converged=False))):
        p = tmp_path / f'appended_{name}.out'; p.write_text(start + ot + start + text); assert read_output(p) == want, name
    p = tmp_path / 'appended_killed.out'; assert gate_pair(read_output(p), read_output(D / 'cp2k_sp_diag.out')) == ['charged run did not end', 'charged last SCF printed no verdict']
    short = ' SCF WAVEFUNCTION OPTIMIZATION\n\n  Leaving inner SCF loop after reaching     1 steps.\n'; spin = '  Integrated absolute spin density  :                               1.0000000000\n'
    for name, text, want in (('seed_then_killed', scf + spin + end.split('\n')[0] + '\n' + ' SCF WAVEFUNCTION OPTIMIZATION\n', EMPTY),
                             ('cdft_ok_last_inner_short', loop + short + ok + end, dict(EMPTY, ended=True, converged=False, energy_Ha=-100.03)),
                             ('cdft_killed_after_header', loop + ok + spin + end.split('\n')[0] + '\n' + ' CDFT EXTERNAL SCF WAVEFUNCTION OPTIMIZATION\n', EMPTY)):
        p = tmp_path / f'{name}.out'; p.write_text(start + text); assert read_output(p) == want, name


def test_read_output_reads_the_linear_scaling_scf_verdict(tmp_path):
    step = lambda ok, e: ('\n SCF run converged in    3 steps.\n\n' if ok else ' SCF not converged! \n') + f' ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]               {e}\n'
    end = '  **** **** ******  **  PROGRAM ENDED AT\n'
    for name, text, want in (('geo_last_failed', step(True, -100.0) + step(False, -100.1) + end, False), ('geo_recovered', step(False, -100.0) + step(True, -100.1) + end, True), ('single_failed', step(False, -100.0) + end, False)):
        p = tmp_path / f'ls_{name}.out'; p.write_text(text); assert read_output(p)['converged'] is want, name


def test_read_output_takes_the_energy_line_in_hartree_only(tmp_path):
    ot = (D / 'cp2k_sp_ot.out').read_text(); line = ' ENERGY| Total FORCE_EVAL ( QS ) energy [a.u.]:             -100.010000000000'
    for name, new, want in (('hartree', ' ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]               -100.010000000000', -100.01), ('cut', ' ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]', None),
                            ('gap_after', line + '\n\n ENERGY| Total FORCE_EVAL ( PWDFT ) gap [hartree]                   0.120000000000', -100.01), ('au_e', line.replace('[a.u.]:', '[au_e]'), -100.01),
                            ('cut_sign', ' ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]               -', None)):
        p = tmp_path / f'{name}.out'; p.write_text(ot.replace(line, new)); assert read_output(p)['energy_Ha'] == want, name
    p = tmp_path / 'ev.out'; p.write_text(ot.replace(line, ' ENERGY| Total FORCE_EVAL ( QS ) energy [eV]                    -2721.380000000000'))
    with pytest.raises(ValueError, match='energies printed in eV'): read_output(p)
    p = tmp_path / 'cut_spin.out'; p.write_text(ot.split('Integrated absolute spin density')[0] + 'Integrated absolute spin density:'); assert read_output(p)['spin'] is None


def test_gate_judges_the_end_point_only_and_names_why():
    good, q0 = read_output(D / 'cp2k_sp_ot.out'), read_output(D / 'cp2k_sp_diag.out')
    assert gate_pair(good, q0) == [] and gate_pair(read_output(D / 'cp2k_sp_ot_inner_short.out'), q0) == [] and gate_pair(read_output(D / 'cp2k_geo_recovered.out'), q0) == []
    assert gate_pair(read_output(D / 'cp2k_sp_ot_failed.out'), q0) == gate_pair(read_output(D / 'cp2k_geo_last_failed_outer.out'), q0) == ['charged last SCF not converged']
    assert gate_pair(good, read_output(D / 'cp2k_killed.out')) == ['neutral run did not end', 'neutral last SCF printed no verdict']
    assert gate_pair(read_output(D / 'cp2k_sp_diag_abort.out'), q0) == ['charged run did not end', 'charged last SCF not converged']


def test_read_restart_subsys_cell_scaled_abc_and_units(tmp_path):
    s, p, c = read_restart(str(D / 'cp2k_cell_ref.restart')); assert list(s) == ['A', 'B', 'B'] and np.allclose(c, [[10, 0, 0], [1, 10, 0], [0, 0, 10]]) and np.allclose(p, [[1, 2, 3], [4, 5, 6], [12, 8, 9]])
    s, p, c = read_restart(D / 'cp2k_scaled.restart'); assert np.allclose(c, [[10, 0, 0], [2, 10, 0], [0, 0, 10]]) and np.allclose(p, [[6, 5, 5], [1, 0, 2]])
    g = np.radians(120.0); s, p, c = read_restart(D / 'cp2k_abc.inp'); assert np.allclose(c, [[10, 0, 0], [11 * np.cos(g), 11 * np.sin(g), 0], [0, 0, 12]]) and np.allclose(p, [[1, 2, 3], [4, 5, 6]])
    b = units.BOHR_A; s, p, c = read_restart(D / 'cp2k_abc_bohr.inp'); assert np.allclose(c, np.eye(3) * 10 * b) and np.allclose(p, [[b, 0, 0], [0, 2 * b, 0]])
    s, p, c = read_restart(D / 'cp2k_units.inp'); assert np.allclose(c, np.diag([10, 10 * b, 10])) and np.allclose(p, [[1, 2, 3]])
    for name, msg in (('cp2k_coord_file.inp', 'no atoms'), ('cp2k_cell_file.inp', 'CELL_FILE_NAME')):
        with pytest.raises(ValueError, match=msg): read_restart(D / name)
    f = tmp_path / 'x.inp'
    for new, want in (('SCALED', [6, 5, 5]), ('SCALED .TRUE.', [6, 5, 5]), ('SCALED F', [0.5, 0.5, 0.5])):
        f.write_text((D / 'cp2k_scaled.restart').read_text().replace('SCALED  T', new)); assert np.allclose(read_restart(f)[1][0], want), new
    f.write_text((D / 'cp2k_abc.inp').read_text().replace('ALPHA_BETA_GAMMA', 'ANGLES')); assert np.allclose(read_restart(f)[2][1], [11 * np.cos(g), 11 * np.sin(g), 0])
    for name, old, new, msg in (('cp2k_units.inp', '[nm]', '[furlong]', 'unknown unit'), ('cp2k_abc.inp', 'ABC 10.0 11.0 12.0', 'ABC 10.0 11.0', 'three numbers'), ('cp2k_abc.inp', '90.0 90.0 120.0', '30.0 30.0 120.0', 'admits no cell'), ('cp2k_abc.inp', 'B 4.0 5.0 6.0', 'B 4.0 5.0 SQRT(2)', 'line not read'), ('cp2k_scaled.restart', 'SCALED  T', 'SCALED  maybe', 'not a logical')):
        f.write_text((D / name).read_text().replace(old, new))
        with pytest.raises(ValueError, match=msg): read_restart(f)


def test_read_lammps_data_atom_styles(tmp_path):
    for name in ('lammps_atomic.data', 'lammps_charge.data', 'lammps_molecular.data', 'lammps_full.data'):
        s, p, c = read_lammps_data(str(D / name), ['A', 'B']); assert list(s) == ['A', 'B', 'B'] and np.allclose(p, XYZ), name
    s, p, c = read_lammps_data(D / 'lammps_full.data', {1: 'A', 2: 'B'}); assert list(s) == ['A', 'B', 'B'] and np.allclose(c, [[10, 0, 0], [1.5, 10, 0], [0.5, -0.25, 10]])
    f = tmp_path / 'x.data'
    for style in ('bond', 'angle'): f.write_text((D / 'lammps_molecular.data').read_text().replace('# molecular', f'# {style}')); assert np.allclose(read_lammps_data(f, ['A', 'B'])[1], XYZ)
    f.write_text((D / 'lammps_atomic.data').read_text().replace(' # atomic', ''))
    with pytest.raises(ValueError, match='pass atom_style'): read_lammps_data(f, ['A', 'B'])
    assert np.allclose(read_lammps_data(f, ['A', 'B'], atom_style='atomic')[1], XYZ)
    f.write_text((D / 'lammps_charge.data').read_text().replace(' # charge', ''))
    with pytest.raises(ValueError, match='pass atom_style'): read_lammps_data(f, ['A', 'B'])
    assert np.allclose(read_lammps_data(f, ['A', 'B'], atom_style='charge')[1], XYZ)
    with pytest.raises(ValueError, match='the file says charge'): read_lammps_data(D / 'lammps_charge.data', ['A', 'B'], atom_style='full')
    f.write_text((D / 'lammps_charge.data').read_text().replace('# charge', '# sphere'))
    with pytest.raises(ValueError, match='not supported'): read_lammps_data(f, ['A', 'B'])
    for types in ({1: 'A'}, ['A']):
        with pytest.raises(ValueError, match='atom type 2 has no species'): read_lammps_data(D / 'lammps_atomic.data', types)


def test_read_extxyz_frames_pbc_and_properties(tmp_path):
    s, p, c = read_extxyz(str(D / 'xyz_frames.extxyz')); assert list(s) == ['A', 'B', 'B'] and np.allclose(p[:, 1:], XYZ[:, 1:]) and np.allclose(p[:, 0], [1.5, 4.5, 7.5]) and np.allclose(c, [[11, 0, 0], [1, 11, 0], [0, 0, 11]])
    for name in ('xyz_aliases.extxyz', 'xyz_no_properties.extxyz'):
        s, p, c = read_extxyz(D / name); assert list(s) == ['A', 'B'] and np.allclose(p, XYZ[:2]) and np.allclose(c, np.eye(3) * 10), name
    for name, msg in (('xyz_truncated.extxyz', 'announces 3 atoms'), ('xyz_slab.extxyz', 'periodic in all three'), ('xyz_no_lattice.extxyz', 'no Lattice'), ('xyz_count_comment.extxyz', 'no Lattice'), ('xyz_short_row.extxyz', 'columns of Properties')):
        with pytest.raises(ValueError, match=msg): read_extxyz(D / name)
    (tmp_path / 'set').mkdir(); write_extxyz(str(tmp_path / 'set' / 'b.extxyz'), ['A', 'B'], XYZ[:2], np.eye(3) * 10); write_extxyz(tmp_path / 'set' / 'a.extxyz', ['B'], XYZ[2:], np.eye(3) * 12)
    names, frames = read_structure_files(str(tmp_path / 'set')); assert names == ['a', 'b'] and list(frames[1][0]) == ['A', 'B'] and np.allclose(frames[1][1], XYZ[:2]) and np.allclose(frames[0][2], np.eye(3) * 12)
    with pytest.raises(FileNotFoundError, match='no .extxyz or .xyz file'): read_structure_files(tmp_path / 'none')


def test_read_spectrum_spin_restricted_appended_and_total_column(tmp_path):
    E, occ, w = read_spectrum(str(D / 'pdos_uks')); assert np.allclose(E, np.array([-0.5, -0.2, 0.1]) * units.HARTREE_EV) and list(occ) == [1, 1, 0] and sorted(w) == ['A', 'B'] and np.allclose(w['A'] + w['B'], 1) and np.allclose(w['A'], [0.6, 0.4, 0.65])
    Eb, ob, wb = read_spectrum(D / 'pdos_uks', 'BETA'); assert np.allclose(Eb / units.HARTREE_EV, [-0.45, -0.1, 0.2]) and list(ob) == [1, 0, 0] and np.allclose(wb['B'], [0.3, 0.7, 0.5])
    Er, orr, wr = read_spectrum(D / 'pdos_rks', spin=None); assert np.allclose(Er, E) and list(orr) == [2, 2, 0] and np.allclose(wr['A'], w['A'])
    Ea, _, wa = read_spectrum(D / 'pdos_append'); assert np.allclose(Ea / units.HARTREE_EV, [-0.6, -0.3, 0.05]) and np.allclose(wa['B'], w['B'])
    Et, _, wt = read_spectrum(D / 'pdos_total'); assert np.allclose(Et, E) and all(np.allclose(wt[k], w[k]) for k in w)
    kind, e, _, ww = read_pdos_kind(str(D / 'pdos_total' / 'run-ALPHA_k1-1.pdos')); assert kind == 'A' and np.allclose(e, [-0.5, -0.2, 0.1]) and np.allclose(ww, [0.6, 0.4, 0.65])
    with pytest.raises(FileNotFoundError, match='no file matches'): read_spectrum(D / 'pdos_rks')
    (tmp_path / 'run-ALPHA_k1-1.pdos').write_text((D / 'pdos_append' / 'run-ALPHA_k1-1.pdos').read_text()); (tmp_path / 'run-ALPHA_k2-1.pdos').write_text((D / 'pdos_uks' / 'run-ALPHA_k2-1.pdos').read_text())
    with pytest.raises(ValueError, match='different steps'): read_spectrum(tmp_path)


def test_mulliken_blocks_unrestricted_and_restricted():
    b = mulliken_blocks(str(D / 'mulliken_uks.out')); assert isinstance(b, list) and len(b) == 2 and list(b[0][0]) == ['A', 'B', 'B'] and np.allclose(b[0][2], [0, 0.1, 0.6]) and np.allclose(b[1][1], [0.1, -0.7, -0.6]) and np.allclose(b[1][2], [0, 0.7, 0])
    r = mulliken_blocks(D / 'mulliken_rks.out'); assert len(r) == 1 and list(r[0][0]) == ['A', 'B', 'B'] and np.allclose(r[0][1], [0.1, -0.7, 0.6]) and not r[0][2].any()


def test_trajectory_energies_and_cube_header(tmp_path):
    assert trajectory_energies(str(D / 'traj_geo-pos-1.xyz')) == [-100.0, -100.01] and trajectory_energies(D / 'traj_md-pos-1.xyz') == [-100.0, -99.99]
    with pytest.raises(ValueError, match='no comment line'): trajectory_energies(D / 'traj_plain.xyz')
    h = cube_header(str(D / 'cube_angstrom.cube')); assert h['n_atoms'] == 2 and h['shape'] == (40, 40, 20) and np.allclose(h['voxel_A'], np.diag([0.25, 0.25, 0.5])) and np.allclose(h['origin_A'], [1, 2, 3])
    b = units.BOHR_A; h = cube_header(D / 'cube_bohr.cube'); assert h['n_atoms'] == 2 and h['shape'] == (30, 30, 40) and np.allclose(h['voxel_A'], np.array([[0.5, 0, 0], [0.1, 0.5, 0], [0, 0, 0.5]]) * b) and np.allclose(h['origin_A'], [2 * b, 0, 0])
    f = tmp_path / 'x.cube'; f.write_text((D / 'cube_angstrom.cube').read_text().replace('  -20 ', '   20 '))
    with pytest.raises(ValueError, match='mix signs'): cube_header(f)


def test_read_restart_refuses_a_slab(tmp_path):
    f = tmp_path / 'slab.inp'; f.write_text('&FORCE_EVAL\n&SUBSYS\n&CELL\nABC 10 10 30\nPERIODIC XY\n&END CELL\n&COORD\nA 1 1 1\n&END COORD\n&END SUBSYS\n&END FORCE_EVAL\n')
    with pytest.raises(ValueError, match='PERIODIC XY'): read_restart(f)
    f.write_text(f.read_text().replace('PERIODIC XY', 'PERIODIC XYZ')); assert read_restart(f)[2][2, 2] == 30.0


def test_read_restart_refuses_a_replicated_cell(tmp_path):
    f = tmp_path / 'muc.inp'; f.write_text('&FORCE_EVAL\n&SUBSYS\n&CELL\nABC 5 5 5\nMULTIPLE_UNIT_CELL 2 2 2\n&END CELL\n&COORD\nA 1 1 1\n&END COORD\n&TOPOLOGY\nMULTIPLE_UNIT_CELL 2 2 2\n&END TOPOLOGY\n&END SUBSYS\n&END FORCE_EVAL\n')
    with pytest.raises(ValueError, match='MULTIPLE_UNIT_CELL 2 2 2'): read_restart(f)
    f.write_text(f.read_text().replace('2 2 2', '1 1 1')); assert read_restart(f)[2][0, 0] == 5.0


def test_read_restart_maps_kind_labels_to_elements(tmp_path):
    f = tmp_path / 'kinds.inp'; f.write_text('&FORCE_EVAL\n&SUBSYS\n&CELL\nABC 5 5 5\n&END CELL\n&COORD\nA_d 0 0 0\nB 1 1 1\nA 2 2 2\n&END COORD\n&KIND A_d\nELEMENT A\n&END KIND\n&KIND B\n&END KIND\n&END SUBSYS\n&END FORCE_EVAL\n')
    assert read_restart(f)[0].tolist() == ['A', 'B', 'A']


def test_write_extxyz_checks_shapes(tmp_path):
    with pytest.raises(ValueError, match='write_extxyz'): write_extxyz(tmp_path / 'x.extxyz', ['A', 'B'], np.zeros((3, 3)), np.eye(3))


def test_read_restart_matches_kinds_without_case_and_skips_the_default_kind(tmp_path):
    f = tmp_path / 'kinds.inp'; f.write_text('&FORCE_EVAL\n&SUBSYS\n&CELL\nABC 5 5 5\n&END CELL\n&COORD\nA_d 0 0 0\nB 1 1 1\n&END COORD\n&KIND a_D\nELEMENT a\n&BASIS\nELEMENT B\n&END BASIS\n&END KIND\n&KIND\nELEMENT B\n&END KIND\n&END SUBSYS\n&END FORCE_EVAL\n')
    assert read_restart(f)[0].tolist() == ['A', 'B']


def test_write_extxyz_refuses_what_it_cannot_write_back(tmp_path):
    with pytest.raises(ValueError, match='finite'): write_extxyz(tmp_path / 'x.extxyz', ['A'], np.array([[np.nan, 0.0, 0.0]]), np.eye(3))
    with pytest.raises(ValueError, match='spaces'): write_extxyz(tmp_path / 'x.extxyz', ['A B'], np.zeros((1, 3)), np.eye(3))


def test_write_extxyz_refuses_empty_or_quoted_symbols_and_a_singular_cell(tmp_path):
    for sym, cell in ((['A', ''], np.eye(3)), (['A', 'B"'], np.eye(3)), (['A', 'B'], np.diag([1.0, 1.0, 0.0]))):
        with pytest.raises(ValueError, match='write_extxyz'): write_extxyz(tmp_path / 'x.extxyz', sym, np.zeros((2, 3)), cell)


def test_read_restart_follows_the_cp2k_cell_keywords_and_units(tmp_path):
    base = (D / 'cp2k_units.inp').read_text(); f = tmp_path / 'x.inp'
    for old, new, msg in (('      A [angstrom] 10.0 0.0 0.0\n', '      UNIT bohr\n      A [angstrom] 10.0 0.0 0.0\n', 'no UNIT keyword'), ('      C [nm] 0.0 0.0 1.0\n', '      C [nm] 0.0 0.0 1.0\n      A 9.0 0.0 0.0\n', 'A is given twice'),
                          ('    &COORD\n', '    &COORD\n@INCLUDE coords.inc\n', 'not expanded'), ('[nm]', '[ang]', 'unknown unit')):
        f.write_text(base.replace(old, new))
        with pytest.raises(ValueError, match=msg): read_restart(f)
    f.write_text(base.replace('[bohr]', '[au_l]')); assert np.allclose(read_restart(f)[2], read_restart(D / 'cp2k_units.inp')[2])


def test_pdos_curve_files_are_skipped_and_refused(tmp_path):
    for p in (D / 'pdos_uks').glob('*.pdos'): (tmp_path / p.name).write_text(p.read_text())
    curve = tmp_path / 'run-ALPHA_k1_curve-1.pdos'; curve.write_text('# Projected DOS for atomic kind A at iteration step i = 0\n#  Energy [a.u.]   s   p\n -0.5  0.1  0.2\n')
    assert np.allclose(read_spectrum(tmp_path)[0], read_spectrum(D / 'pdos_uks')[0])
    with pytest.raises(ValueError, match='no Occupation column'): read_pdos_kind(curve)


def test_trajectory_energies_read_extended_xyz_and_name_an_unreadable_value(tmp_path):
    f = tmp_path / 't.extxyz'; f.write_text(''.join(f'1\nLattice="5 0 0 0 5 0 0 0 5" Properties=species:S:1:pos:R:3 pbc="T T T" Step={i} Time=0.5 Energy={e}\nA 0 0 0\n' for i, e in ((1, -100.5), (2, -100.75))))
    assert trajectory_energies(f) == [-100.5, -100.75]
    f.write_text(' i =        1, time =        0.500, E =   ****************\n')
    with pytest.raises(ValueError, match='unreadable energy'): trajectory_energies(f)


def test_gate_reports_an_energy_it_cannot_compare():
    ok = read_output(D / 'cp2k_sp_ot.out'); q0 = read_output(D / 'cp2k_sp_diag.out')
    for e in (None, float('nan')): assert 'charged energy not read' in gate_pair(dict(ok, energy_Ha=e), q0, energy_reference={'charged': -100.01})
    assert gate_pair(ok, q0, energy_reference={'charged': -100.01}) == []


def test_truncated_cube_headers_and_bad_extxyz_cells_raise(tmp_path):
    f = tmp_path / 'c.cube'; f.write_text('t\nt\n 1 0.0 0.0 0.0\n 10 0.1 0.0\n')
    with pytest.raises(ValueError, match='truncated'): cube_header(f)
    g = tmp_path / 'x.extxyz'
    for lat, pos, msg in (('5 0 0 0 5 0 0 0 0', '0 0 0', 'non-singular'), ('5 0 0 0 5 0 0 0 5', 'nan 0 0', 'finite')):
        g.write_text(f'1\nLattice="{lat}" Properties=species:S:1:pos:R:3 pbc="T T T"\nA {pos}\n')
        with pytest.raises(ValueError, match=msg): read_extxyz(g)
    g.write_bytes('1\nLattice="5 0 0 0 5 0 0 0 5" Properties=species:S:1:pos:R:3 pbc="T T T" note=\xe9\nA 0 0 0\n'.encode('latin-1')); assert read_extxyz(g)[1].shape == (1, 3)


def _restart(path, cell, coord):
    path.write_text('&FORCE_EVAL\n&SUBSYS\n&CELL\n' + cell + '\n&END CELL\n&COORD\n' + coord + '\n&END COORD\n&END SUBSYS\n&END FORCE_EVAL\n'); return read_restart(path)


def test_read_restart_builds_an_oblique_cell_from_lengths_and_angles(tmp_path):
    abc, (al, be, ga) = np.array([10.0, 11.0, 12.0]), np.radians([70.0, 80.0, 100.0]); cy = (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
    ref = np.array([[abc[0], 0, 0], [abc[1] * np.cos(ga), abc[1] * np.sin(ga), 0], [abc[2] * np.cos(be), abc[2] * cy, abc[2] * np.sqrt(1 - np.cos(be) ** 2 - cy ** 2)]])
    _, p, c = _restart(tmp_path / 'oblique.inp', 'ABC 10.0 11.0 12.0\nALPHA_BETA_GAMMA 70.0 80.0 100.0', 'A 0.5 0.25 0.125\nSCALED T'); angle = lambda u, v: np.degrees(np.arccos(u @ v / np.linalg.norm(u) / np.linalg.norm(v)))
    assert abs(c - ref).max() < 1e-12 and abs(np.linalg.norm(c, axis=1) - abc).max() < 1e-12 and abs(np.array([angle(c[1], c[2]), angle(c[0], c[2]), angle(c[0], c[1])]) - [70.0, 80.0, 100.0]).max() < 1e-10
    assert np.linalg.det(c) > 0 and c[0, 1] == c[0, 2] == c[1, 2] == 0 and abs(p[0] - np.array([0.5, 0.25, 0.125]) @ ref).max() < 1e-12


def test_read_restart_skips_blank_and_comment_lines_inside_coord(tmp_path):
    s, p, c = _restart(tmp_path / 'gaps.inp', 'ABC 5 5 5', 'A 1 1 1\n\n# a comment line\n   ! another\nB 2 2 2  # after the numbers\n\n   \nA 3 3 3 ! trailing')
    assert list(s) == ['A', 'B', 'A'] and np.array_equal(p, [[1, 1, 1], [2, 2, 2], [3, 3, 3]]) and abs(c - np.eye(3) * 5).max() < 1e-12


def test_read_restart_converts_length_and_angle_units(tmp_path):
    ref = _restart(tmp_path / 'ref.inp', 'ABC 10 11 12\nALPHA_BETA_GAMMA 70 80 100', 'A 1 2 3')[2]
    for k, (cell, unit, f) in enumerate((('ABC [pm] 1000 1100 1200\nALPHA_BETA_GAMMA [deg] 70 80 100', 'pm', 0.01), ('ABC [angstrom] 10 11 12\nALPHA_BETA_GAMMA [rad] ' + ' '.join(repr(float(x)) for x in np.radians([70, 80, 100])), 'angstrom', 1.0), ('ABC [nm] 1.0 1.1 1.2\nANGLES [DEG] 70 80 100', 'nm', 10.0))):
        _, p, c = _restart(tmp_path / f'u{k}.inp', cell, f'UNIT {unit}\nA 100 200 300'); assert abs(c - ref).max() < 1e-12 and abs(p - np.array([[100.0, 200.0, 300.0]]) * f).max() < 1e-12, unit
    for cell, msg in (('ABC [furlong] 10 11 12', r'unknown unit \[furlong\] for ABC'), ('ABC [ang] 10 11 12', r'unknown unit \[ang\] for ABC'), ('ABC 10 10 10\nALPHA_BETA_GAMMA 120 90 30', 'admits no cell'), ('ABC', 'ABC needs three numbers'), ('ABC 10 11 x', 'ABC needs three numbers')):
        with pytest.raises(ValueError, match=msg): _restart(tmp_path / 'bad.inp', cell, 'A 1 2 3')


def test_read_spectrum_keeps_projects_named_like_curves_and_read_restart_scopes_preprocessor_lines(tmp_path):
    for p in (D / 'pdos_uks').glob('*.pdos'): (tmp_path / p.name.replace('run-', 'iv_curve-')).write_text(p.read_text())
    assert np.allclose(read_spectrum(tmp_path)[0], read_spectrum(D / 'pdos_uks')[0])
    base = (D / 'cp2k_units.inp').read_text(); f = tmp_path / 'k.inp'
    f.write_text(base.replace('    &CELL\n', '    &KIND A\n@INCLUDE kinds.inc\n    &END KIND\n    &CELL\n')); assert np.allclose(read_restart(f)[2], read_restart(D / 'cp2k_units.inp')[2])
    f.write_text(base.replace('  &END SUBSYS\n', '  &END SUBSYS\n  &SUBSYS\n    &CELL\n      ABC 5 5 5\n    &END CELL\n  &END SUBSYS\n'))
    with pytest.raises(ValueError, match='more than one &SUBSYS &CELL'): read_restart(f)
    f.write_text(base.replace('[nm] 0.0 0.0 1.0', '[m] 0.0 0.0 1e-9')); assert np.allclose(read_restart(f)[2], read_restart(D / 'cp2k_units.inp')[2])


def test_read_spectrum_refuses_kinds_that_list_different_states(tmp_path):
    for p in (D / 'pdos_uks').glob('*ALPHA*.pdos'): (tmp_path / p.name).write_text(p.read_text())
    f = sorted(tmp_path.glob('*.pdos'))[-1]; lines = f.read_text().splitlines(); f.write_text('\n'.join(lines[:-1]) + '\n')
    with pytest.raises(ValueError, match='list different states'): read_spectrum(tmp_path)


def test_read_output_flags_unconverged_empty_states_and_an_optimisation_that_did_not_complete(tmp_path):
    ot, end = (D / 'cp2k_sp_ot.out').read_text(), '  **** **** ******  **  PROGRAM ENDED AT'
    lumo, warn = '  Using OT eigensolver for additional unoccupied orbitals spin            {}\n', ' WARNING : did not converge in ot_eigensolver\n'
    geo, cell = ' ***                     STARTING GEOMETRY OPTIMIZATION                      ***\n', ' ***                     STARTING   CELL   OPTIMIZATION                      ***\n'
    done = ' ***                    GEOMETRY OPTIMIZATION COMPLETED                      ***\n'
    for name, extra, key, want in (('lumo', lumo.format(1) + lumo.format(2), 'unoccupied_converged', True), ('lumo_failed', lumo.format(1) + warn + lumo.format(2), 'unoccupied_converged', False),
                                   ('no_lumo', '', 'unoccupied_converged', None), ('geo_done', geo + done, 'optimised', True), ('cell_done', cell + done, 'optimised', True),
                                   ('geo_limit', geo + ' *** MAXIMUM NUMBER OF OPTIMIZATION STEPS REACHED ***\n', 'optimised', False), ('cell_limit', cell, 'optimised', False),
                                   ('geo_walltime', geo + ' *** GEO run terminated - exceeded requested execution time:       9.000 seconds\n', 'optimised', False),
                                   ('dimer_rotation', ' ***                    ROTATION OPTIMIZATION COMPLETED                      ***\n', 'optimised', None), ('no_optimiser', '', 'optimised', None),
                                   ('md_shells', done + ' *** MAXIMUM NUMBER OF OPTIMIZATION STEPS REACHED ***\n' + done, 'optimised', None)):
        p = tmp_path / f'{name}.out'; p.write_text(ot.replace(end, extra + end)); assert read_output(p)[key] is want, name
    assert gate_pair(read_output(tmp_path / 'geo_limit.out'), read_output(D / 'cp2k_sp_diag.out')) == ['charged optimisation did not complete']


def test_read_output_starts_afresh_when_a_rerun_aborts_on_its_input(tmp_path):
    box = ' *******************************************************************************\n * [ABORT]                                                                     *\n *  \\___/           found an unknown keyword MULTIPLICTY in section DFT        *\n'
    p = tmp_path / 'rerun.out'; p.write_text((D / 'cp2k_sp_ot.out').read_text() + box); assert read_output(p) == EMPTY
    assert gate_pair(read_output(p), read_output(D / 'cp2k_sp_diag.out')) == ['charged run did not end', 'charged last SCF printed no verdict'] and read_output(D / 'cp2k_sp_diag_abort.out')['converged'] is False


def test_read_restart_reads_quoted_values_and_takes_the_element_as_cp2k_does(tmp_path):
    f = tmp_path / 'w-1.restart'; f.write_text(' &FORCE_EVAL\n   &SUBSYS\n     &CELL\n       A 10.0 0.0 0.0\n       B 0.0 10.0 0.0\n       C 0.0 0.0 10.0\n     &END CELL\n     &COORD\n       UNIT "bohr"\n'
                                              '       O1 1.0 2.0 3.0\n       H_a 1.0 2.0 4.0\n       Hb 1.0 3.0 3.0\n       Si2_b 2.0 2.0 3.0\n     &END COORD\n     &KIND "O1"\n       ELEMENT "O"\n     &END KIND\n'
                                              '     &KIND "H_a"\n     &END KIND\n     &KIND Hb\n       ELEMENT_SYMBOL H\n     &END KIND\n   &END SUBSYS\n &END FORCE_EVAL\n')
    s, p, c = read_restart(f); assert s.tolist() == ['O', 'H', 'H', 'Si'] and np.allclose(p[0], np.array([1, 2, 3]) * units.BOHR_A) and np.allclose(c, np.eye(3) * 10)
    f.write_text(f.read_text().replace('Si2_b', '_H'))
    with pytest.raises(ValueError, match='kind _H has no ELEMENT and its label names none'): read_restart(f)


def test_read_spectrum_skips_restricted_curve_files_and_other_projections(tmp_path):
    for p in (D / 'pdos_rks').glob('*.pdos'): (tmp_path / p.name.replace('run-', 'si-kpts-')).write_text(p.read_text())
    (tmp_path / 'si-kpts-k1_curve-1.pdos').write_text('# Projected DOS for atomic kind A at iteration step i = 0\n#  Energy [a.u.]   s   p\n -0.5  0.1  0.2\n')
    (tmp_path / 'si-kpts-list1-1.pdos').write_text('# Projected DOS for list 1 of atoms at iteration step i = 0\n')
    assert np.allclose(read_spectrum(tmp_path, spin=None)[0], read_spectrum(D / 'pdos_rks', spin=None)[0])


def test_read_lammps_data_takes_the_units_and_type_labels_write_data_records(tmp_path):
    head = 'LAMMPS data file via write_data, version 29 Aug 2024, timestep = 0, units = {}\n\n2 atoms\n2 atom types\n\n0.0 10.0 xlo xhi\n0.0 10.0 ylo yhi\n0.0 10.0 zlo zhi\n\nAtom Type Labels\n\n1 {}\n2 O\n\nAtoms # atomic\n\n'
    f = tmp_path / 'l.data'; f.write_text(head.format('electron', 'Si') + '1 Si 1.0 2.0 3.0\n2 O 4.0 5.0 6.0\n')
    s, p, c = read_lammps_data(f, ['Si', 'O']); assert s.tolist() == ['Si', 'O'] and np.allclose(p, XYZ[:2] * units.BOHR_A) and np.allclose(c, np.eye(3) * 10 * units.BOHR_A)
    with pytest.raises(ValueError, match='atom type 1 is Si in the file, O in type_to_species'): read_lammps_data(f, ['O', 'Si'])
    f.write_text(head.format('metal', 'Ob') + '1 1 1.0 2.0 3.0\n2 2 4.0 5.0 6.0\n'); s, p, c = read_lammps_data(f, ['O', 'O']); assert s.tolist() == ['O', 'O'] and np.allclose(p, XYZ[:2])
    f.write_text(head.format('lj', 'Pd') + '1 1 1.0 2.0 3.0\n2 2 4.0 5.0 6.0\n'); assert np.allclose(read_lammps_data(f, ['Pd', 'O'])[1], XYZ[:2])
    with pytest.raises(ValueError, match='atom type 1 is Pd in the file, Ru in type_to_species'): read_lammps_data(f, ['Ru', 'O'])
    f.write_text(head.format('nonsense', 'Si') + '1 1 1.0 2.0 3.0\n2 2 4.0 5.0 6.0\n')
    with pytest.raises(ValueError, match='units nonsense have no length in A'): read_lammps_data(f, ['Si', 'O'])


def test_read_extxyz_takes_whole_keys_with_escapes_and_the_exact_names(tmp_path):
    lat, other = 'Lattice="8.2 0.0 0.0 0.4 8.6 0.0 0.3 -0.2 9.1"', '"8.0 0.0 0.0 0.0 8.0 0.0 0.0 0.0 8.0"'
    for name, head, row in (('info_lattice', f'{lat} Properties=species:S:1:pos:R:3 lattice={other} pbc="T T T"', 'Si 1.0 2.0 3.0'),
                            ('dotted_key', f'{lat} Properties=species:S:1:pos:R:3 initial.Lattice={other} pbc="T T T"', 'Si 1.0 2.0 3.0'),
                            ('escaped_quote', f'{lat} Properties=species:S:1:pos:R:3:forces:R:3 note="from \\"run1\\" Properties=species:S:1:forces:R:3:pos:R:3" pbc="T T T"', 'Si 1.0 2.0 3.0 9.0 9.0 9.0'),
                            ('species_column', f'{lat} Properties=species:S:1:pos:R:3:Species:S:1 pbc="T T T"', 'Si 1.0 2.0 3.0 Si0')):
        p = tmp_path / f'{name}.extxyz'; p.write_text(f'1\n{head}\n{row}\n'); s, x, c = read_extxyz(p)
        assert s.tolist() == ['Si'] and np.allclose(x, [[1, 2, 3]]) and np.allclose(c, [[8.2, 0, 0], [0.4, 8.6, 0], [0.3, -0.2, 9.1]]), name
    p = tmp_path / 'case.extxyz'; p.write_text(f'1\nlattice={other} LATTICE={other} pbc="T T T"\nSi 1.0 2.0 3.0\n')
    with pytest.raises(ValueError, match='keys lattice and LATTICE differ only by case'): read_extxyz(p)


def test_read_structure_files_reads_xyz_in_any_case_and_skips_hidden_files(tmp_path):
    for name in ('a.extxyz', 'b.xyz', 'c.EXTXYZ'): write_extxyz(tmp_path / name, ['A'], XYZ[:1], np.eye(3) * 10)
    (tmp_path / '._a.extxyz').write_bytes(b'\x00\x05\x16\x07'); assert read_structure_files(tmp_path)[0] == ['a', 'b', 'c'] and [p.name for p in structure_paths(tmp_path)] == ['a.extxyz', 'b.xyz', 'c.EXTXYZ']
    write_extxyz(tmp_path / 'a.xyz', ['A'], XYZ[:1], np.eye(3) * 10)
    with pytest.raises(ValueError, match=r"read_structure_files: .*two files share a name: \['a.extxyz', 'a.xyz'\]"): read_structure_files(tmp_path)
    with pytest.raises(FileNotFoundError, match='structure_paths: no .extxyz or .xyz file'): structure_paths(tmp_path / 'none')
