"""LAMMPS data files written by write_data: atomic or charge style, orthogonal or triclinic box."""
import numpy as np


def read_lammps_data(path, type_to_species):
    """Symbols, positions (A, from the box corner) and cell of a data file; types map to species through the given sequence."""
    lines = path.read_text().splitlines(); n_atoms, bounds, tilt, start, style = 0, {}, (0.0, 0.0, 0.0), None, 'atomic'
    for i, raw in enumerate(lines):
        line = raw.split('#')[0].strip(); t = line.split()
        if not line: continue
        if len(t) >= 2 and t[1] == 'atoms': n_atoms = int(t[0])
        elif len(t) >= 4 and t[2] in ('xlo', 'ylo', 'zlo'): bounds[t[2][0]] = (float(t[0]), float(t[1]))
        elif len(t) >= 6 and t[3] == 'xy': tilt = (float(t[0]), float(t[1]), float(t[2]))
        elif raw.strip().startswith('Atoms'): start = i; style = raw.split('#', 1)[1].strip() if '#' in raw else 'atomic'
    if start is None: raise ValueError(f'{path}: no Atoms section')
    k = 3 if style.startswith('charge') else 2; ids, types, pos = [], [], []
    for raw in lines[start + 1:]:
        t = raw.split('#')[0].split()
        if not t: continue
        if not t[0].lstrip('+-').isdigit(): break
        ids.append(int(t[0])); types.append(int(t[1])); pos.append([float(v) for v in t[k:k + 3]])
        if len(ids) == n_atoms: break
    if len(ids) != n_atoms: raise ValueError(f'{path}: expected {n_atoms} atoms, read {len(ids)}')
    (x0, x1), (y0, y1), (z0, z1) = bounds['x'], bounds['y'], bounds['z']; xy, xz, yz = tilt
    cell = np.array([[x1 - x0, 0, 0], [xy, y1 - y0, 0], [xz, yz, z1 - z0]], float); order = np.argsort(ids, kind='stable')
    return np.array([type_to_species[types[i] - 1] for i in order]), (np.array(pos, float) - [x0, y0, z0])[order], cell
