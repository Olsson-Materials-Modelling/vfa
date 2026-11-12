"""LAMMPS data files: atomic, charge, molecular, bond, angle or full atom style, orthogonal or triclinic box."""
from pathlib import Path
import numpy as np

_STYLE = {'atomic': (1, 2), 'charge': (1, 3), 'molecular': (2, 3), 'bond': (2, 3), 'angle': (2, 3), 'full': (2, 4)}


def _int(t):
    return t.lstrip('+-').isdigit()


def _species(type_to_species, t, path):
    try:
        if hasattr(type_to_species, 'keys'): return type_to_species[t]
        if t < 1: raise IndexError
        return type_to_species[t - 1]
    except (KeyError, IndexError): raise ValueError(f'read_lammps_data: {path}: atom type {t} has no species') from None


def read_lammps_data(path, type_to_species, atom_style=None):
    """Symbols, positions (the file's length unit, from the box corner) and cell of a LAMMPS data file."""
    path = Path(path); raw = path.read_text().splitlines(); lines = [ln.split('#')[0].strip() for ln in raw]; n_atoms, bounds, tilt, body, start = None, {}, (0.0, 0.0, 0.0), len(lines), None
    for i in range(1, len(lines)):
        t = lines[i].split()
        if not t: continue
        if t[0][:1].isalpha(): body = i; break
        if len(t) == 2 and t[1] == 'atoms': n_atoms = int(t[0])
        elif len(t) == 4 and t[3] in ('xhi', 'yhi', 'zhi') and t[2] == t[3][0] + 'lo': bounds[t[2][0]] = (float(t[0]), float(t[1]))
        elif len(t) == 6 and t[3:] == ['xy', 'xz', 'yz']: tilt = (float(t[0]), float(t[1]), float(t[2]))
    for i in range(body, len(lines)):
        if lines[i].split()[:1] == ['Atoms']: start = i; break
    if start is None: raise ValueError(f'read_lammps_data: {path}: no Atoms section')
    if not n_atoms or set(bounds) != {'x', 'y', 'z'}: raise ValueError(f'read_lammps_data: {path}: the header lacks the atom count or the xlo xhi, ylo yhi, zlo zhi bounds')
    noted = raw[start].split('#', 1)[1].split()[:1] if '#' in raw[start] else []; noted = noted[0].lower() if noted else None
    if atom_style is not None and noted is not None and atom_style != noted: raise ValueError(f'read_lammps_data: {path}: atom_style {atom_style} but the file says {noted}')
    style = atom_style or noted; rows = []
    for ln in lines[start + 1:]:
        t = ln.split()
        if not t: continue
        if not _int(t[0]): break
        rows.append(t)
        if len(rows) == n_atoms: break
    if len(rows) != n_atoms: raise ValueError(f'read_lammps_data: {path}: expected {n_atoms} atoms, read {len(rows)}')
    width = {len(t) for t in rows}
    if len(width) != 1: raise ValueError(f'read_lammps_data: {path}: Atoms rows have {sorted(width)} columns')
    width = width.pop()
    if style is None: raise ValueError(f'read_lammps_data: {path}: the Atoms section names no atom style (# style); pass atom_style, since column counts alone are ambiguous')
    if style not in _STYLE: raise ValueError(f'read_lammps_data: {path}: atom style {style} not supported ({", ".join(sorted(_STYLE))})')
    else:
        kt, kx = _STYLE[style]
        if width not in (kx + 3, kx + 6): raise ValueError(f'read_lammps_data: {path}: atom style {style} has {kx + 3} columns (+3 image flags), the rows {width}')
    if not all(_int(t[kt]) for t in rows): raise ValueError(f'read_lammps_data: {path}: atom types must be integers in column {kt + 1}')
    ids = [int(t[0]) for t in rows]; types = [int(t[kt]) for t in rows]; pos = np.array([[float(v) for v in t[kx:kx + 3]] for t in rows])
    (x0, x1), (y0, y1), (z0, z1) = bounds['x'], bounds['y'], bounds['z']; xy, xz, yz = tilt
    cell = np.array([[x1 - x0, 0, 0], [xy, y1 - y0, 0], [xz, yz, z1 - z0]], float); order = np.argsort(ids, kind='stable')
    return np.array([_species(type_to_species, types[i], path) for i in order]), (pos - [x0, y0, z0])[order], cell
