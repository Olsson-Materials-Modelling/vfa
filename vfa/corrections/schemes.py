"""Reader of the JSON written by cp2k-charge-corrections 0.8.0 (schema 4.0), corrected 0 -> q energies and transition levels."""
import json, math, posixpath
from pathlib import Path

TOOL = 'cp2k_charge_corrections'; SCHEMA = '4.0'
POTENTIAL_SCHEMES = {'FNV_planar_gaussian': 'fnv', 'LZ_scaled_PC_plus_alignment': 'lz', 'PC_IIC': 'pc', 'site_gaussian_median': 'site'}
SELECTED = (('scheme', 'scheme'), ('selected_eV', 'E_corr_eV'), ('reliable', 'reliability_ok'), ('reasons', 'reasons'), ('impact_eV', 'impact_eV'), ('screening', 'screening'), ('ionic_geometry', 'ionic_geometry'), ('level_shift_eV', 'delta_epsilon_eV'))


def _at(js, path, where):
    v = js
    for k in path:
        if not isinstance(v, dict) or k not in v: raise ValueError(f"summarise: {where} has no field {'/'.join(path)}")
        v = v[k]
    return v


def _load(path, mode):
    p = Path(path)
    if not p.is_file(): raise ValueError(f'summarise: no file {p}')
    try: js = json.loads(p.read_text())
    except json.JSONDecodeError as e: raise ValueError(f'summarise: {p} is not valid JSON ({e})') from None
    tool = _at(js, ('inputs', 'tool_name'), p); ver = _at(js, ('inputs', 'schema_version'), p)
    if tool != TOOL: raise ValueError(f'summarise: {p} was written by {tool!r}, not {TOOL!r}')
    if ver != SCHEMA: raise ValueError(f'summarise: {p} has schema {ver!r}; vfa reads schema {SCHEMA} (cp2k-charge-corrections 0.8.0)')
    if _at(js, ('selected', 'mode'), p) != mode: raise ValueError(f"summarise: {p} holds the {js['selected']['mode']} mode, given as the {mode} JSON")
    return js, _at(js, ('inputs', 'code_version'), p), p


def _selected(js, mode, where):
    out = {f'{mode}_{col}': _at(js, ('selected', key), where) for col, key in SELECTED}
    out[f'{mode}_level_reliable'] = _at(js, ('selected', 'level_reliability_ok'), where) if out[f'{mode}_level_shift_eV'] is not None else None; return out


def _run(js, key):
    p = (js.get('inputs') or {}).get(key)
    if not isinstance(p, str): return None
    tail = tuple(s for s in posixpath.normpath(p.replace('\\', '/')).split('/')[:-1] if s not in ('', '.'))[-2:]
    return tail if tail and '..' not in tail else None


def summarise(potential_json, density_json, *, schemes=None, override=False):
    """One row from the corrector's potential and density JSON of one charged cell: each mode's selected correction, by path."""
    if potential_json is None and density_json is None: raise ValueError('summarise: neither a potential nor a density JSON given')
    if schemes is not None and not isinstance(schemes, dict): raise ValueError(f'summarise: schemes must map scheme names to column names, not {type(schemes).__name__}')
    out = {}
    if density_json is not None:
        dj, code, w = _load(density_json, 'density'); out.update(schema_version=SCHEMA, code_version=code, **_selected(dj, 'density', w), density_carrier=_at(dj, ('selected', 'carrier'), w), density_status=_at(dj, ('energies', 'selected_correction_status'), w))
    if potential_json is not None:
        js, code, w = _load(potential_json, 'potential')
        if out and code != out['code_version']: raise ValueError(f"summarise: the potential JSON was written by corrector {code}, the density JSON by {out['code_version']}")
        for a, b, state in (('defect_cube', 'rho_charged_cube', 'charged'), ('bulk_cube', 'rho_neutral_cube', 'neutral')):
            pa, pb = _run(js, a), _run(dj, b) if density_json is not None else None
            k = min(len(pa), len(pb)) if None not in (pa, pb) else 0
            if not override and k and pa[-k:] != pb[-k:]: raise ValueError(f"summarise: the {state} cubes were read from {'/'.join(pa)} for the potential and {'/'.join(pb)} for the density: different runs (override=True accepts them)")
        out.update(schema_version=SCHEMA, code_version=code, **_selected(js, 'potential', w))
        if density_json is not None and (out['potential_ionic_geometry'], out['potential_screening']) != (out['density_ionic_geometry'], out['density_screening']): raise ValueError(f"summarise: the potential JSON is for the {out['potential_ionic_geometry'] or 'undeclared'} geometry with {out['potential_screening']} screening, the density JSON for the {out['density_ionic_geometry'] or 'undeclared'} one with {out['density_screening']}")
        pick = schemes if schemes is not None else {('concerted_' if out['potential_ionic_geometry'] == 'relaxed' else '') + k: v for k, v in POTENTIAL_SCHEMES.items()}
        for name, col in pick.items(): out[f'{col}_eV'] = _at(js, ('schemes', name, 'E_corr_eV'), w) if schemes is not None or name in _at(js, ('schemes',), w) else None
    return out


def _density(row, name):
    status = row.get('density_status')
    if not isinstance(status, str): raise ValueError(f'{name}: the row was read without a density JSON; corrected energies take the density mode')
    v = row.get('density_selected_eV')
    if v is None or not math.isfinite(v): raise ValueError(f"{name}: no selected density correction (status {status}, reasons {row.get('density_reasons')})")
    return float(v)


def corrected_addition(e_addition_eV, relaxed, *, override=False):
    """E(q) - E(0) + C, both cells relaxed in their charge, with C the density mode's selected correction from the charged cell's summarise row."""
    v = _density(relaxed, 'corrected_addition'); g = relaxed.get('density_ionic_geometry')
    if isinstance(g, str) and g != 'relaxed' and not override: raise ValueError(f"corrected_addition: the correction was computed at the {g} geometry; pass override=True to accept it")
    return float(e_addition_eV + v)


def transition_levels(formation_eV):
    """Charge transition levels along the lowest formation energy, from {charge: formation energy (eV) at a Fermi level of zero}."""
    try: f = {float(q): float(e) for q, e in dict(formation_eV).items()}
    except (TypeError, ValueError): f = {}
    if len(f) < 2 or not all(math.isfinite(x) for x in list(f) + list(f.values())): raise ValueError('transition_levels: two or more charges with finite formation energies are needed')
    q, out = max(f), []
    while q > min(f):
        cross = {p: (f[p] - f[q]) / (q - p) for p in f if p < q}; level = min(cross.values()); p = min(k for k, v in cross.items() if v <= level + 1e-12)
        out.append(dict(charges=(q, p), level_eV=level)); q = p
    return out
