"""The finite-size correction schemes of cp2k-charge-corrections (github.com/Olsson-Materials-Modelling/cp2k-charge-corrections), read from its JSON output and labelled.  The potential-based and density-based modes are alternative estimates of one correction and are never summed; a Kohn-Sham shift attaches only to the matching orbital."""
import json
from pathlib import Path

POTENTIAL_SCHEMES = {'FNV_planar_gaussian': 'fnv', 'LZ_scaled_PC_only': 'lz', 'eFNV_interstitial_grid_median': 'efnv', 'PC_IIC': 'pc'}     # the corrector's names and the short labels


def _find(obj, key):
    if isinstance(obj, dict):
        if key in obj: return obj
        for v in obj.values():
            r = _find(v, key)
            if r is not None: return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find(v, key)
            if r is not None: return r
    return None


def summarise(potential_json=None, density_json=None):
    """One row from the corrector's potential and density JSON files: the recommended and each named potential scheme (eV), the Kohn-Sham shift, the density image interaction and its sum with the charging alignment, the state's image expectation."""
    out = {}
    if potential_json is not None and Path(potential_json).is_file():
        js = json.loads(Path(potential_json).read_text()); holder = _find(js, 'schemes'); schemes = holder['schemes'] if holder else {}
        rec = [(k, v) for k, v in schemes.items() if isinstance(v, dict) and v.get('recommended')]
        if rec: out.update(potential_scheme=rec[0][0], potential_correction_eV=rec[0][1].get('E_corr_eV'), potential_status=rec[0][1].get('status'))
        for name, col in POTENTIAL_SCHEMES.items():
            v = schemes.get(name)
            if isinstance(v, dict) and 'E_corr_eV' in v: out[f'{col}_eV'] = v['E_corr_eV']
        ks = _find(js, 'delta_epsilon_eV')
        if ks is not None: out.update(ks_shift_eV=ks.get('delta_epsilon_eV'), ks_corrected_eV=ks.get('corrected_eigenvalue_eV'))
    if density_json is not None and Path(density_json).is_file():
        js = json.loads(Path(density_json).read_text()); en = js.get('energies', {}); out.update(density_image_eV=en.get('E_corr_eV'), density_point_charge_eV=en.get('E_point_charge_corr_eV'))
        cd = ((js.get('neutral_pipeline') or {}).get('components') or {}).get('CD') or {}; pa = cd.get('selected_minus_q_median_eV')
        if out.get('density_image_eV') is not None and pa is not None: out.update(charging_alignment_eV=pa, density_corrected_eV=out['density_image_eV'] + pa, charging_alignment_ok=(cd.get('selected_quality') or {}).get('ok'))
        st = js.get('state_corrections') or []
        if st: out.update(state_image_eV=st[0].get('energy_expectation_eV'), state_shift_status=st[0].get('eigenvalue_shift_status'), state_reliable=(st[0].get('numerical_reliability') or {}).get('ok'))
    return out


def corrected_relaxation(e_relax_eV, correction_vertical_eV, correction_relaxed_eV):
    """Relaxation energy with the finite-size correction of each charged endpoint: E_relax + C_vertical - C_relaxed."""
    return float(e_relax_eV + correction_vertical_eV - correction_relaxed_eV)
