"""Acceptance gates on a pair of charged and neutral single points at one geometry."""
import math


def gate_pair(charged, neutral, spin_window=None, neutral_spin_max=0.5, energy_reference=None, energy_tol_Ha=5e-4, geometry_identical=None, grids_match=None, unpaired=1):
    """Reasons a pair of read_output records is rejected (run, SCF, spin; the caller's geometry, grid and energy checks); empty when it passes."""
    win = tuple(spin_window) if spin_window is not None else (unpaired - 0.5, unpaired + 1.0); bad = []
    if len(win) != 2 or not win[0] <= win[1]: raise ValueError(f'gate_pair: spin window {win} is not (low, high)')
    for tag, r in (('charged', charged), ('neutral', neutral)):
        if not r['ended']: bad.append(f'{tag} run did not end')
        if not r['converged']: bad.append(f"{tag} last SCF {'printed no verdict' if r['converged'] is None else 'not converged'}")
    if neutral['spin'] is not None and neutral['spin'] > neutral_spin_max: bad.append(f'neutral spin {neutral["spin"]:.3f} > {neutral_spin_max}')
    if charged['spin'] is not None and not win[0] <= charged['spin'] <= win[1]: bad.append(f'charged spin {charged["spin"]:.3f} outside {win}')
    if geometry_identical is False: bad.append('the two geometries differ')
    if grids_match is False: bad.append('cube grids differ between the pair')
    for tag, r in (('charged', charged), ('neutral', neutral)):
        ref = (energy_reference or {}).get(tag)
        if ref is None: continue
        e = r['energy_Ha']
        if e is None or not math.isfinite(e): bad.append(f'{tag} energy not read')
        elif abs(e - ref) > energy_tol_Ha: bad.append(f'{tag} energy off its reference by {abs(e - ref):.2e} Ha')
    return bad
