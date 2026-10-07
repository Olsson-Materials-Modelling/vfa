"""Acceptance gates on a pair of charged and neutral single points at one geometry, before any finite-size correction is read: a failed gate names its reason."""
import numpy as np


def gate_pair(charged, neutral, spin_window=(0.5, 2.0), neutral_spin_max=0.5, energy_reference=None, energy_tol_Ha=5e-4, geometry_identical=None, grids_match=None):
    """Reasons a pair of vfa.io.cp2k.read_output records is rejected, empty when it passes: run ended, SCF converged, spin windows, identical geometry, matching grids, energy against a reference (Ha)."""
    bad = []
    for tag, r in (('charged', charged), ('neutral', neutral)):
        if not r['ended']: bad.append(f'{tag} run did not end')
        if not r['converged']: bad.append(f'{tag} SCF not converged')
    if neutral['spin'] is not None and neutral['spin'] > neutral_spin_max: bad.append(f'neutral spin {neutral["spin"]:.3f} > {neutral_spin_max}')
    if charged['spin'] is not None and not spin_window[0] <= charged['spin'] <= spin_window[1]: bad.append(f'charged spin {charged["spin"]:.3f} outside {spin_window}')
    if geometry_identical is False: bad.append('the two geometries differ')
    if grids_match is False: bad.append('cube grids differ between the pair')
    for tag, r in (('charged', charged), ('neutral', neutral)):
        ref = (energy_reference or {}).get(tag)
        if ref is not None and r['energy_Ha'] is not None and abs(r['energy_Ha'] - ref) > energy_tol_Ha: bad.append(f'{tag} energy off its reference by {abs(r["energy_Ha"] - ref):.2e} Ha')
    return bad
