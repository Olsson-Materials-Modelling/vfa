"""Extended-XYZ files: symbols, positions (A) and the 3x3 cell (rows are lattice vectors)."""
import re
import numpy as np


def read_extxyz(path):
    """The last frame of an extended-XYZ file."""
    lines = path.read_text().splitlines(); starts = [i for i, l in enumerate(lines) if l.strip().isdigit()]
    start = starts[-1]; n = int(lines[start]); head = lines[start + 1]
    cell = np.array(re.search(r'Lattice="([^"]*)"', head).group(1).split(), float).reshape(3, 3)
    m = re.search(r'Properties=(\S+)', head); props = m.group(1).split(':') if m else ['species', 'S', '1', 'pos', 'R', '3']
    col, cols = 0, {}
    for k in range(0, len(props), 3): cols[props[k]] = (col, col + int(props[k + 2])); col += int(props[k + 2])
    rows = [l.split() for l in lines[start + 2:start + 2 + n]]; a, b = cols['species']; c, d = cols['pos']
    return np.array([r[a] for r in rows]), np.array([[float(v) for v in r[c:d]] for r in rows]), cell


def write_extxyz(path, symbols, positions, cell):
    """One frame of extended XYZ."""
    lat = ' '.join(f'{v:.10f}' for v in np.asarray(cell).ravel())
    with open(path, 'w') as f:
        f.write(f'{len(symbols)}\nLattice="{lat}" Properties=species:S:1:pos:R:3 pbc="T T T"\n')
        for s, p in zip(symbols, positions): f.write(f'{s} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}\n')


def read_structure_files(folder):
    """Names and (symbols, positions, cell) of every extended-XYZ file in a folder, sorted by name."""
    paths = sorted(folder.glob('*.extxyz')); return [p.stem for p in paths], [read_extxyz(p) for p in paths]
