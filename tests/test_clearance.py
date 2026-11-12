"""The clearance: exactness in any cell, blocks, workers, lattice translations, the surface, the sweep, the free-volume partition, input checks."""
import os, sys, tracemalloc
from pathlib import Path
import numpy as np, pytest
from skimage.measure import marching_cubes
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vfa.io.extxyz import read_extxyz
from vfa.structure.cell import images, volume
from scipy.spatial import cKDTree
from vfa.voids.clearance import (threads, exact_clearance, clearance_field, sample_clearance, covered_clearance, void_surface, probe_sweep, accessible_fraction, void_volume, clearance_cdf, clearance_statistics,
                                 sampling_convergence, grid_convergence)
from vfa.voids.radii import Radii
from vfa.voids.sites import atom_free_volume, cavity_distance

ROOT = Path(__file__).resolve().parents[1]


def brute(pos, cell, rad, pts, m=6):
    """Clearance at points by direct minimisation over a box of lattice images."""
    shifts = np.array([[i, j, k] for i in range(-m, m + 1) for j in range(-m, m + 1) for k in range(-m, m + 1)], float); rep = (pos[None] + (shifts @ cell)[:, None]).reshape(-1, 3); r = np.tile(rad, len(shifts))
    return np.array([np.min(np.linalg.norm(rep - p, axis=1) - r) for p in pts])


def test_skewed_cell_grows_the_image_shell():
    cell = np.array([[3.0, 0, 0], [18.0, 3.0, 0], [0, 0, 3.0]]); pos = np.array([[0.2, 0.3, 0.1], [1.7, 1.2, 0.4]]); rad = np.array([0.4, 0.1])
    s = sample_clearance(pos, cell, rad, n=128, sampler='random', seed=1); f = np.random.default_rng(1).random((128, 3)) @ cell; assert abs(s - brute(pos, cell, rad, f, m=14)).max() < 1e-12


def test_block_size_does_not_change_the_field():
    rng = np.random.default_rng(4); cell = np.diag([8.0, 9.0, 10.0]); pos = rng.random((30, 3)) @ cell; rad = rng.choice([0.5, 1.2], 30)
    a, _ = clearance_field(pos, cell, rad, dx=0.4); b, _ = clearance_field(pos, cell, rad, dx=0.4, chunk=777); assert np.array_equal(a, b)


def test_field_and_free_volume_do_not_depend_on_lattice_translations_of_atoms():
    rng = np.random.default_rng(5); cell = np.array([[9, 0, 0], [3, 9, 0], [1, -2, 9.0]]); pos = rng.random((40, 3)) @ cell; rad = rng.choice([0.6, 1.1], 40); moved = pos + rng.integers(-2, 3, (40, 3)) @ cell
    a, n = clearance_field(pos, cell, rad, dx=0.4); b, _ = clearance_field(moved, cell, rad, dx=0.4); assert abs(a - b).max() < 1e-12
    assert np.array_equal(atom_free_volume(a, n, cell, pos, rad, 0.5), atom_free_volume(b, n, cell, moved, rad, 0.5))


def test_surface_range_and_shape():
    C = np.random.default_rng(6).random((8, 8, 8)); cell = np.eye(3) * 4.0
    assert void_surface(C, (8, 8, 8), cell, 2.0) == 0.0 and void_surface(C, (8, 8, 8), cell, -1.0) == 0.0 and void_surface(C, (8, 8, 8), cell, 0.5) > 0
    with pytest.raises(ValueError): void_surface(C[:, :, 0], (8, 8), cell, 0.5)


def test_sweep_reports_the_specific_surface_in_m2_per_cm3():
    cell = np.eye(3) * 10.0; pos = np.array([[5.0, 5.0, 5.0]]); C, n = clearance_field(pos, cell, np.array([1.0]), dx=0.25); row = probe_sweep(C, n, cell, (0.5,))[0]
    assert set(row) == {'probe_A', 'fraction', 'volume_A3', 'surface_A2', 'surface_m2_cm3'} and abs(row['surface_m2_cm3'] - 1e4 * row['surface_A2'] / volume(cell)) < 1e-12 and abs(row['surface_A2'] / (4 * np.pi * 1.5 ** 2) - 1) < 0.03


def test_void_volume_is_the_accessible_fraction_of_the_cell():
    cell = np.eye(3) * 6.0; C, _ = clearance_field(np.array([[3.0, 3.0, 3.0]]), cell, np.array([1.0]), dx=0.3); assert abs(void_volume(C, cell, 0.5) - accessible_fraction(C, 0.5) * 216.0) < 1e-12


def tiled_area(C, n, cell, probe):
    """Void surface area by marching cubes on the twice-tiled field, the reference for the padded one."""
    v, f, _, _ = marching_cubes(np.tile(C, (2, 2, 2)), level=probe); frac = v / np.asarray(n, float); keep = np.all((frac[f].mean(1) >= 0) & (frac[f].mean(1) < 1), axis=1); f = f[keep]
    x = frac @ cell; a, b = x[f[:, 1]] - x[f[:, 0]], x[f[:, 2]] - x[f[:, 0]]; return float(0.5 * np.linalg.norm(np.cross(a, b), axis=1).sum())


def test_free_volume_belongs_to_the_atom_with_the_nearest_surface():
    rng = np.random.default_rng(12); cell = np.array([[9, 0, 0], [3, 8, 0], [-1, 2, 9.0]]); pos = rng.random((30, 3)) @ cell; rad = rng.choice([0.4, 1.4], 30); C, n = clearance_field(pos, cell, rad, dx=0.3)
    fv = atom_free_volume(C, n, cell, pos, rad, 0.5); free = np.flatnonzero(C.ravel() >= 0.5); x = ((np.stack(np.unravel_index(free, tuple(n)), -1) + 0.5) / n) @ cell; rep, r = images(pos, cell, rad, reach=7.0); owner, centre = [], []
    for blk in np.array_split(x, 40): d = np.linalg.norm(blk[:, None] - rep[None], axis=2); owner.append(np.argmin(d - r, axis=1) % 30); centre.append(np.argmin(d, axis=1) % 30)
    owner, centre = np.concatenate(owner), np.concatenate(centre); ref = np.bincount(owner, minlength=30) * volume(cell) / C.size
    assert C.max() + rad.max() < 7.0 and np.array_equal(fv, ref) and np.mean(owner != centre) > 0.1 and abs(fv.sum() - void_volume(C, cell, 0.5)) < 1e-9
    with pytest.raises(ValueError, match='not the clearance of these positions and radii'): atom_free_volume(C, n, cell, pos, rad[::-1], 0.5)


def test_per_atom_values_must_match_the_atoms():
    cell = np.eye(3) * 8.0; pos = np.array([[1.0, 1, 1], [5.0, 5, 5]]); long_ = np.array([0.0, 1.0, 3.0]); C, n = clearance_field(pos, cell, np.array([0.0, 1.0]), dx=0.5)
    for call in (lambda: images(pos, cell, long_), lambda: covered_clearance(pos, cell, long_, np.zeros((4, 3))), lambda: clearance_field(pos, cell, long_, dx=0.5), lambda: clearance_field(pos, cell, 0.5, dx=0.5),
                 lambda: sample_clearance(pos, cell, long_, n=64), lambda: atom_free_volume(C, n, cell, pos, long_, 0.5)):
        with pytest.raises(ValueError, match='one value per atom|one per atom|value per atom'): call()


def test_non_finite_inputs_raise():
    cell = np.eye(3) * 8.0; pos = np.array([[1.0, 1, 1], [5.0, 5, 5]]); rad = np.array([0.0, 1.0]); C, n = clearance_field(pos, cell, rad, dx=0.5); bad = C.copy(); bad[1, 2, 3] = np.nan; nanpos = pos.copy(); nanpos[1, 0] = np.nan
    for call in (lambda: clearance_field(nanpos, cell, rad, dx=0.5), lambda: sample_clearance(nanpos, cell, rad, n=64), lambda: cavity_distance(nanpos, cell, np.zeros((1, 3))), lambda: void_surface(bad, n, cell, 0.5), lambda: accessible_fraction(bad, 0.5),
                 lambda: clearance_statistics(bad), lambda: void_volume(bad, cell, 0.5), lambda: clearance_cdf(bad, [0.5]), lambda: probe_sweep(bad, n, cell, (0.5,)), lambda: atom_free_volume(bad, n, cell, pos, rad, 0.5)):
        with pytest.raises(ValueError, match='finite'): call()


def test_invalid_clearance_inputs_raise():
    cell = np.eye(3) * 8.0; pos = np.array([[1.0, 1, 1]]); rad = np.array([0.5])
    with pytest.raises(ValueError, match='clearance_field: dx'): clearance_field(pos, cell, rad, dx=0)
    with pytest.raises(ValueError, match='clearance_field: no atoms'): clearance_field(np.zeros((0, 3)), cell, np.zeros(0), dx=1.0)
    with pytest.raises(ValueError, match='sample_clearance: n'): sample_clearance(pos, cell, rad, n=0)
    with pytest.raises(ValueError, match='powers of two'): sample_clearance(pos, cell, rad, n=1000)
    with pytest.raises(ValueError, match='clearance_statistics: the clearance is empty'): clearance_statistics(np.zeros(0))
    with pytest.raises(ValueError, match='accessible_fraction: the clearance is empty'): accessible_fraction([], 0.5)
    with pytest.raises(ValueError, match='finite and non-negative'): clearance_field(pos, cell, np.array([-0.5]), dx=1.0)
    with pytest.raises(ValueError, match=r"Radii.of \(covalent\): no value for species \['C'\]"): Radii({'A': 1.1, 'B': 0.7}, 'covalent').of(['A', 'C'])


def test_void_surface_matches_a_tiled_reference_in_little_memory():
    s, p, c = read_extxyz(ROOT / 'example/data/structures/rho0.060_1.extxyz'); C, n = clearance_field(p, c, Radii.load(ROOT / 'example/data/radii.json').of(s), dx=0.3)
    for probe in (0.3, 0.8, 1.2): assert abs(void_surface(C, n, c, probe) - tiled_area(C, n, c, probe)) < 1e-9 * tiled_area(C, n, c, probe)
    g = (np.arange(64) + 0.5) / 64; F = 16.0 * np.sqrt(sum((x - 0.5) ** 2 for x in np.meshgrid(g, g, g, indexing='ij'))); tracemalloc.start(); area = void_surface(F, F.shape, np.eye(3) * 16.0, 5.0); peak = tracemalloc.get_traced_memory()[1]; tracemalloc.stop()
    assert abs(area / (4 * np.pi * 25) - 1) < 2e-3 and peak < 4 * F.nbytes


def test_workers_do_not_change_the_clearance():
    rng = np.random.default_rng(14); cell = np.diag([8.0, 9.0, 10.0]); pos = rng.random((30, 3)) @ cell; rad = rng.choice([0.5, 1.2], 30)
    a, n = clearance_field(pos, cell, rad, dx=0.4, workers=1); b, _ = clearance_field(pos, cell, rad, dx=0.4, workers=-1); assert np.array_equal(a, b)
    assert np.array_equal(sample_clearance(pos, cell, rad, n=256, workers=1), sample_clearance(pos, cell, rad, n=256, workers=2)) and np.array_equal(atom_free_volume(a, n, cell, pos, rad, 0.6, workers=1), atom_free_volume(a, n, cell, pos, rad, 0.6))
    assert sampling_convergence(pos, cell, rad, 0.6, counts=(64,), workers=1) == sampling_convergence(pos, cell, rad, 0.6, counts=(64,)) and grid_convergence(pos, cell, rad, 0.6, spacings=(0.5,), workers=1) == grid_convergence(pos, cell, rad, 0.6, spacings=(0.5,))


def test_threads_follow_the_cpus_this_process_may_use():
    assert threads(-1) == (len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()) and threads(3) == 3
    with pytest.raises(ValueError, match='threads'): threads(0)


def test_a_probe_radius_must_be_finite():
    cell = np.eye(3) * 6.0; pos = np.array([[3.0, 3.0, 3.0]]); C, n = clearance_field(pos, cell, np.array([1.0]), dx=0.5)
    for f in (lambda: accessible_fraction(C, np.nan), lambda: void_surface(C, n, cell, np.nan), lambda: atom_free_volume(C, n, cell, pos, np.array([1.0]), np.nan)):
        with pytest.raises(ValueError, match='probe radius must be finite'): f()


def test_ties_in_ownership_go_to_the_lower_index_and_inputs_are_checked():
    cell = np.eye(3) * 8.0; pos = np.array([[1.75, 4.0, 4.0], [5.75, 4.0, 4.0]]); rad = np.array([0.5, 0.5]); C, n = clearance_field(pos, cell, rad, dx=0.5)
    fv = atom_free_volume(C, n, cell, pos, rad, 0.0); assert abs(fv[0] - fv[1] - 2 * 16 * 16 * 0.125) < 1e-9 and abs(fv.sum() - accessible_fraction(C, 0.0) * 512.0) < 1e-9
    assert exact_clearance(cKDTree(pos), rad, np.array([4.0, 4.0, 4.0])).shape == (1,)
    with pytest.raises(ValueError, match='exact_clearance'): exact_clearance(cKDTree(pos), rad, np.array([np.nan, 4.0, 4.0]))
    with pytest.raises(ValueError, match='void_volume'): void_volume(C, cell, np.nan)
    for bad in (None, 2.7, 0, True, -1.0, 2.0, -2):
        with pytest.raises(ValueError, match='threads'): threads(bad)


def test_covered_clearance_folds_points_outside_the_cell():
    pos, cell = np.array([[4.0, 4.0, 4.0]]), np.eye(3) * 8.0; p = np.array([[1.0, 2.0, 3.0]]); far = p + np.array([3, -2, 5]) @ cell
    assert np.allclose(covered_clearance(pos, cell, np.ones(1), far), covered_clearance(pos, cell, np.ones(1), p))
