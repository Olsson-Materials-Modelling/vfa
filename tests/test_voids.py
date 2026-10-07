"""Closed-form lattices, periodic invariance and the probe sweep."""
import sys
from pathlib import Path
import numpy as np, pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vfa.voids.clearance import clearance_field, accessible_fraction, void_surface, probe_sweep
from vfa.voids.network import voronoi_network, descriptors, free_sphere, pore_network, percolating_edges
from vfa.voids.sites import atom_free_volume, cavity_distance
from vfa.voids.radii import Radii


def lattice(kind, a=3.0, n=3):
    basis = [(0, 0, 0)] if kind == 'sc' else [(0, 0, 0), (.5, .5, 0), (.5, 0, .5), (0, .5, .5)]
    pos = np.array([((i + x) * a, (j + y) * a, (k + z) * a) for i in range(n) for j in range(n) for k in range(n) for x, y, z in basis]); return pos, np.eye(3) * a * n


def test_simple_cubic_closed_form():
    pos, cell = lattice('sc', n=4); a = 3.0; net = voronoi_network(pos, cell, np.zeros(len(pos))); d = descriptors(net)
    assert abs(d['Di_A'] - a * np.sqrt(3)) < 1e-6 and abs(d['Df_A'] - a * np.sqrt(2)) < 2e-3        # body-centre sphere; the face-centre window


def test_fcc_closed_form():
    pos, cell = lattice('fcc'); a = 3.0; d = descriptors(voronoi_network(pos, cell, np.zeros(len(pos))))
    assert abs(d['Di_A'] - a) < 1e-6 and abs(d['Df_A'] - 2 * a / np.sqrt(6)) < 2e-3                  # octahedral hole; the tetrahedral window


def test_radii_shrink_the_spheres():
    pos, cell = lattice('sc', n=4); r = 0.4; d0 = descriptors(voronoi_network(pos, cell, np.zeros(len(pos)))); d1 = descriptors(voronoi_network(pos, cell, np.full(len(pos), r)))
    assert abs((d0['Di_A'] - d1['Di_A']) - 2 * r) < 1e-6 and abs((d0['Df_A'] - d1['Df_A']) - 2 * r) < 4e-3


def test_single_sphere_fraction_and_surface():
    cell = np.eye(3) * 10.0; pos = np.array([[5.0, 5.0, 5.0]]); R = 1.2; C, n = clearance_field(pos, cell, np.array([R]), dx=0.1); p = 0.8
    assert abs((1 - accessible_fraction(C, p)) - 4 / 3 * np.pi * (R + p) ** 3 / 1000) < 2e-3            # the excluded ball
    assert abs(void_surface(C, n, cell, p) / (4 * np.pi * (R + p) ** 2) - 1) < 0.02                      # its surface


def test_periodic_invariance():
    rng = np.random.default_rng(0); cell = np.eye(3) * 9.0; pos = rng.uniform(0, 9, (40, 3)); rad = rng.choice([0.7, 1.1], 40)
    shift = (pos + np.array([3.1, -2.2, 4.7])) % 9.0; a, b = descriptors(voronoi_network(pos, cell, rad)), descriptors(voronoi_network(shift, cell, rad))
    assert abs(a['Di_A'] - b['Di_A']) < 1e-6 and abs(a['Df_A'] - b['Df_A']) < 1e-6
    Ca, _ = clearance_field(pos, cell, rad, dx=0.3); Cb, _ = clearance_field(shift, cell, rad, dx=0.3); assert abs(Ca.max() - Cb.max()) < 0.3 and abs(accessible_fraction(Ca, 0.5) - accessible_fraction(Cb, 0.5)) < 0.02


def test_sweep_is_monotone_and_sites_partition():
    rng = np.random.default_rng(1); cell = np.eye(3) * 9.0; pos = rng.uniform(0, 9, (40, 3)); rad = np.full(40, 0.8); C, n = clearance_field(pos, cell, rad, dx=0.25); rows = probe_sweep(C, n, cell, (0.3, 0.6, 0.9))
    assert all(x['fraction'] >= y['fraction'] for x, y in zip(rows, rows[1:]))
    fv = atom_free_volume(C, n, cell, pos, 0.3); assert abs(fv.sum() - rows[0]['volume_A3']) < 1e-9 and (fv >= 0).all()
    net = voronoi_network(pos, cell, rad); pores = pore_network(net, 0.3); assert pores is not None and len(pores['pore_r']) >= 1 and (pores['pore_r'] >= 0.3).all()
    assert np.isfinite(cavity_distance(pos, cell, pores['pore_xyz'])).all(); mask, probe = percolating_edges(net); assert mask.sum() > 0 and abs(probe - free_sphere(net)) < 1e-12


def test_radii_round_trip(tmp_path):
    r = Radii({'A': 1.1, 'B': 0.7}, 'covalent'); r.save(tmp_path / 'radii.json'); assert Radii.load(tmp_path / 'radii.json') == r and list(r.of(['B', 'A'])) == [0.7, 1.1]


def test_pointwise_and_grid_modes_agree():
    from vfa.voids.clearance import sample_clearance, clearance_cdf
    pos, cell = lattice('sc', n=3); a = 3.0; p = 1.0; exact = 1 - 4 / 3 * np.pi * p ** 3 / a ** 3                   # one excluded ball per atom
    C, n = clearance_field(pos, cell, np.zeros(len(pos)), dx=0.1); s = sample_clearance(pos, cell, np.zeros(len(pos)), n=16384)
    assert abs(accessible_fraction(C, p) - exact) < 3e-3 and abs(accessible_fraction(s, p) - exact) < 5e-3 and abs(clearance_cdf(s, [p])[0] - (1 - exact)) < 5e-3


def test_radii_presets():
    r = Radii.covalent(['Si', 'O']); i = Radii.ionic(['Si', 'O']); assert r.name == 'covalent' and i.name == 'ionic' and r.table['Si'] > i.table['Si'] and i.table['O'] > r.table['O']
    from vfa.voids.radii import mass_density; assert abs(mass_density(['Si'] * 8, np.eye(3) * 5.43) - 2.33) < 0.01                 # crystalline silicon


def test_clearance_is_exact_with_unequal_radii():
    from vfa.voids.clearance import exact_clearance
    from vfa.structure.cell import images
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(5); cell = np.eye(3) * 12.0; pos = rng.uniform(0, 12, (60, 3)); rad = np.where(rng.random(60) < 0.3, 2.0, 0.2); rep, r = images(pos, cell, rad); pts = rng.uniform(0, 12, (500, 3))
    brute = np.min(np.linalg.norm(pts[:, None] - rep[None], axis=2) - r[None], axis=1); assert abs(exact_clearance(cKDTree(rep), r, pts, k=2) - brute).max() < 1e-12
    C, n = clearance_field(pos, cell, rad, dx=0.5, k=2); C2, _ = clearance_field(pos, cell, rad, dx=0.5, k=64); assert abs(C - C2).max() < 1e-12
    net = voronoi_network(pos, cell, rad); assert pore_network(net, 50.0)['pore_r'].size == 0
