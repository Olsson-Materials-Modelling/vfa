"""Extended-XYZ files: symbols, positions (A) and the 3x3 cell (rows are lattice vectors) of cells periodic in all three directions."""
import re
from pathlib import Path
import numpy as np
from ..structure.cell import _valid

_PAIR = re.compile(r'''([A-Za-z_][\w-]*)\s*=\s*("[^"]*"|'[^']*'|\S+)'''); _SPECIES, _POS = ('species', 'element', 'symbols'), ('pos', 'positions')


def _column(cols, names, kind, width, path):
    hit = [n for n in names if n in cols]
    if not hit: raise ValueError(f'read_extxyz: {path}: Properties has no {" or ".join(names)} column')
    k, typ, a, b = hit[0], *cols[hit[0]]
    if typ != kind or b - a != width: raise ValueError(f'read_extxyz: {path}: Properties column {k} is {typ}:{b - a}, expected {kind}:{width}')
    return a, b


def read_extxyz(path):
    """Symbols, positions (A) and cell (A, rows) of the last frame of a periodic extended-XYZ file."""
    path = Path(path); lines = path.read_text(errors='replace').splitlines(); i, start = 0, None
    while i < len(lines):
        t = lines[i].split()
        if not t: i += 1; continue
        if len(t) != 1 or not t[0].isdigit(): raise ValueError(f'read_extxyz: {path}: line {i + 1} should hold an atom count')
        n = int(t[0])
        if i + 2 + n > len(lines): raise ValueError(f'read_extxyz: {path}: the frame at line {i + 1} announces {n} atoms, the file ends after {len(lines) - i - 2}')
        start, i = i, i + 2 + n
    if start is None: raise ValueError(f'read_extxyz: {path}: no frame')
    n = int(lines[start]); info = {k.lower(): v.strip('"\'') for k, v in _PAIR.findall(lines[start + 1])}
    if 'lattice' not in info: raise ValueError(f'read_extxyz: {path}: the comment line of the last frame has no Lattice')
    lat = info['lattice'].split()
    if len(lat) != 9: raise ValueError(f'read_extxyz: {path}: Lattice needs nine numbers, got {len(lat)}')
    pbc = info.get('pbc', 'T T T').split()
    if len(pbc) != 3 or not all(p.upper() in ('T', 'TRUE', '1') for p in pbc): raise ValueError(f'read_extxyz: {path}: pbc="{info["pbc"]}"; only cells periodic in all three directions are read')
    props = info.get('properties', 'species:S:1:pos:R:3').split(':')
    if len(props) % 3 or not all(c.isdigit() for c in props[2::3]): raise ValueError(f'read_extxyz: {path}: Properties={info["properties"]} is not name:type:count triples')
    col, cols = 0, {}
    for k in range(0, len(props), 3): cols[props[k].lower()] = (props[k + 1].upper(), col, col + int(props[k + 2])); col += int(props[k + 2])
    a, _ = _column(cols, _SPECIES, 'S', 1, path); c, d = _column(cols, _POS, 'R', 3, path); rows = [l.split() for l in lines[start + 2:start + 2 + n]]
    if any(len(r) != col for r in rows): raise ValueError(f'read_extxyz: {path}: an atom line of the last frame does not have the {col} columns of Properties')
    try: pos = np.array([[float(v) for v in r[c:d]] for r in rows], float).reshape(n, 3)
    except ValueError: raise ValueError(f'read_extxyz: {path}: a position of the last frame is not a number') from None
    try: cell = np.array(lat, float).reshape(3, 3)
    except ValueError: raise ValueError(f'read_extxyz: {path}: the Lattice holds a value that is not a number') from None
    pos, cell = _valid(pos, cell, f'read_extxyz: {path}'); return np.array([r[a] for r in rows]), pos, cell


def write_extxyz(path, symbols, positions, cell):
    """One frame of extended XYZ."""
    sym, pos, c = np.asarray(symbols), np.asarray(positions, float), np.asarray(cell, float)
    if c.shape != (3, 3) or pos.ndim != 2 or pos.shape[1] != 3 or sym.shape != (len(pos),): raise ValueError(f'write_extxyz: one symbol and three coordinates per atom and a 3 x 3 cell are needed, not {sym.shape}, {pos.shape}, {c.shape}')
    if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(c)): raise ValueError('write_extxyz: positions and cell must be finite')
    if abs(np.linalg.det(c)) <= 1e-12 * np.prod(np.linalg.norm(c, axis=1)): raise ValueError('write_extxyz: the cell is singular')
    if any(not s or any(ch.isspace() or ch in '"\'' for ch in s) for s in map(str, sym)): raise ValueError('write_extxyz: symbols must be non-empty, without spaces or quotes')
    lat = ' '.join(f'{v:.10f}' for v in c.ravel())
    with open(Path(path), 'w') as f:
        f.write(f'{len(symbols)}\nLattice="{lat}" Properties=species:S:1:pos:R:3 pbc="T T T"\n')
        for s, p in zip(symbols, positions): f.write(f'{s} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}\n')


def read_structure_files(folder):
    """Names and (symbols, positions, cell) of every extended-XYZ file in a folder, sorted by name."""
    folder = Path(folder); paths = sorted(folder.glob('*.extxyz'))
    if not paths: raise FileNotFoundError(f'read_structure_files: no .extxyz file in {folder}')
    return [p.stem for p in paths], [read_extxyz(p) for p in paths]
