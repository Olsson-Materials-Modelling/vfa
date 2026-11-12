"""CP2K files: restart geometry, output log, projected densities of states, Mulliken blocks, trajectory energies, cube headers."""
import re
from pathlib import Path
import numpy as np
from ..units import BOHR_A, HARTREE_EV

_LENGTH = {'angstrom': 1.0, 'bohr': BOHR_A, 'au_l': BOHR_A, 'nm': 10.0, 'pm': 0.01, 'm': 1e10}
_ANGLE = {'deg': 1.0, 'rad': 180.0 / np.pi}
_TRUE, _FALSE = {'T', '.T.', 'TRUE', '.TRUE.', 'Y', 'YES', 'ON'}, {'F', '.F.', 'FALSE', '.FALSE.', 'N', 'NO', 'OFF'}
_CURVE = re.compile(r'_k\d+_curve-'); _KIND = re.compile(r'atomic kind\s+(\S+)'); _STEP = re.compile(r'step i =\s*(\d+)'); _F = r'(-?\d+\.\d+)'
_MULLIKEN_UKS = re.compile(rf'^\s*(\d+)\s+(\S+)\s+\d+\s+{_F}\s+{_F}\s+{_F}\s+{_F}\s*$'); _MULLIKEN_RKS = re.compile(rf'^\s*(\d+)\s+(\S+)\s+\d+\s+{_F}\s+{_F}\s*$')


def _num(t):
    try: float(t); return True
    except ValueError: return False


def _unit(token, table, path, what):
    u = table.get(token.strip('[]').lower())
    if u is None: raise ValueError(f'read_restart: {path}: unknown unit {token} for {what}')
    return u


def _values(t, table, default, path):
    unit = _unit(t[1], table, path, t[0]) if len(t) > 1 and t[1].startswith('[') else None; v = t[2:] if unit is not None else t[1:]
    if len(v) != 3 or not all(_num(x) for x in v): raise ValueError(f'read_restart: {path}: &CELL {t[0]} needs three numbers, got {" ".join(v)}')
    return np.array([float(x) for x in v]) * (unit if unit is not None else default)


def _abc_cell(abc, angles_deg, path):
    a, b, c = abc; ca, cb, cg = np.cos(np.radians(angles_deg)); sg = np.sin(np.radians(angles_deg[2]))
    cy = (ca - cb * cg) / sg if abs(sg) > 1e-12 else np.inf; cz2 = 1.0 - cb ** 2 - cy ** 2
    if not cz2 > 1e-12: raise ValueError(f'read_restart: {path}: ALPHA_BETA_GAMMA {angles_deg.tolist()} admits no cell')
    return np.array([[a, 0.0, 0.0], [b * cg, b * sg, 0.0], [c * cb, c * cy, c * np.sqrt(cz2)]])


def read_restart(path):
    """Elements (the ELEMENT of each &KIND, else the kind label), positions (A) and cell (A, rows) of a periodic CP2K restart or input file."""
    path = Path(path); stack, kw, symbols, coords, unit_p, scaled, n_cell, n_coord, kind, elements = [], {}, [], [], 1.0, False, 0, 0, None, {}
    for raw in path.read_text(errors='replace').splitlines():
        line = raw.split('#')[0].split('!')[0].strip()
        if not line: continue
        if line.startswith('@'):
            if stack[-2:] in (['SUBSYS', 'CELL'], ['SUBSYS', 'COORD']): raise ValueError(f'read_restart: {path}: the preprocessor line {line.split()[0]} inside &{stack[-1]} is not expanded')
            continue
        t = line.split(); key = t[0].upper()
        if key.startswith('&'):
            if key == '&END': stack = stack[:-1]; continue
            stack.append(key[1:]); n_cell += stack[-2:] == ['SUBSYS', 'CELL']; n_coord += stack[-2:] == ['SUBSYS', 'COORD']
            if n_cell > 1 or n_coord > 1: raise ValueError(f'read_restart: {path}: more than one &SUBSYS &CELL or &COORD section')
            kind = (t[1].upper() if len(t) > 1 else None) if stack[-2:] == ['SUBSYS', 'KIND'] else kind; continue
        if stack[-2:] == ['SUBSYS', 'KIND'] and key == 'ELEMENT' and len(t) >= 2 and kind is not None: elements[kind] = t[1].capitalize(); continue
        if key == 'MULTIPLE_UNIT_CELL' and 'SUBSYS' in stack and [int(x) for x in t[1:4]] != [1, 1, 1]: raise ValueError(f'read_restart: {path}: MULTIPLE_UNIT_CELL {" ".join(t[1:4])} in &{stack[-1]}: replicated cells are not expanded')
        if stack[-2:] == ['SUBSYS', 'CELL']:
            if key == 'CELL_FILE_NAME': raise ValueError(f'read_restart: {path}: the cell is in an external file (CELL_FILE_NAME)')
            if key == 'UNIT': raise ValueError(f'read_restart: {path}: &CELL has no UNIT keyword in CP2K; give the unit on each line, as A [bohr] ...')
            if key in ('A', 'B', 'C', 'ABC', 'ALPHA_BETA_GAMMA', 'ANGLES'):
                name = 'ALPHA_BETA_GAMMA' if key == 'ANGLES' else key
                if name in kw: raise ValueError(f'read_restart: {path}: &CELL {name} is given twice')
                kw[name] = t
            elif key == 'PERIODIC' and len(t) >= 2 and t[1].upper() != 'XYZ': raise ValueError(f'read_restart: {path}: PERIODIC {t[1]}: only cells periodic in all three directions are read')
        elif stack[-2:] == ['SUBSYS', 'COORD']:
            if key == 'UNIT' and len(t) >= 2: unit_p = _unit(t[1], _LENGTH, path, 'UNIT')
            elif key == 'SCALED':
                v = t[1].upper() if len(t) > 1 else 'T'
                if v not in _TRUE | _FALSE: raise ValueError(f'read_restart: {path}: SCALED {t[1]} is not a logical')
                scaled = v in _TRUE
            elif len(t) >= 4 and all(_num(x) for x in t[1:4]): symbols.append(t[0]); coords.append([float(x) for x in t[1:4]])
            else: raise ValueError(f'read_restart: {path}: &COORD line not read: {line}')
    if not symbols: raise ValueError(f'read_restart: {path}: no atoms in &SUBSYS &COORD (coordinates in an external file are not read)')
    if 'ABC' in kw: cell = _abc_cell(_values(kw['ABC'], _LENGTH, 1.0, path), _values(kw['ALPHA_BETA_GAMMA'], _ANGLE, 1.0, path) if 'ALPHA_BETA_GAMMA' in kw else np.full(3, 90.0), path)
    elif {'A', 'B', 'C'} <= kw.keys(): cell = np.vstack([_values(kw[k], _LENGTH, 1.0, path) for k in 'ABC'])
    else: raise ValueError(f'read_restart: {path}: &SUBSYS &CELL holds neither ABC nor all of A, B, C')
    xyz = np.array(coords); return np.array([elements.get(s.upper(), s) for s in symbols]), xyz @ cell if scaled else xyz * unit_p, cell


def _number(text):
    try: return float(text.split()[-1])
    except (ValueError, IndexError): return None


def _energy(line, path):
    unit, closed, value = line.partition(') energy [')[2].partition(']')
    if not closed: return None
    if unit.strip().lower() not in ('hartree', 'au_e', 'a.u.', 'internal_cp2k'): raise ValueError(f'read_output: {path}: energies printed in {unit.strip()}; vfa reads hartree (ENERGY_UNIT)')
    return _number(value.replace(':', ' '))


def read_output(path):
    """Ended, whether the last SCF run met its own criterion, total energy (Ha) and integrated absolute spin density, from the last program run."""
    r = dict(ended=False, converged=None, energy_Ha=None, spin=None); inner = outer = None; free = False; cdft = off = object()
    for ln in Path(path).read_text(errors='replace').splitlines():
        if 'PROGRAM STARTED AT' in ln: r = dict(ended=False, converged=None, energy_Ha=None, spin=None); inner = outer = None; free = False; cdft = off
        elif ln.strip() == 'SCF WAVEFUNCTION OPTIMIZATION':
            inner = outer = r['energy_Ha'] = r['spin'] = None
            if cdft is not None: cdft = off
        elif ln.strip() == 'CDFT EXTERNAL SCF WAVEFUNCTION OPTIMIZATION': cdft = r['energy_Ha'] = r['spin'] = None
        elif ln.strip() == 'No variables optimised in outer loop': free = True
        elif 'PROGRAM ENDED AT' in ln: r['ended'] = True
        elif 'SCF run converged in' in ln: inner = True
        elif 'Leaving inner SCF loop' in ln: inner = False
        elif 'SCF not converged!' in ln: inner = False
        elif 'CDFT SCF loop FAILED' in ln: cdft = False
        elif 'CDFT SCF loop converged' in ln: cdft = True
        elif 'outer SCF loop FAILED' in ln: outer = False
        elif 'outer SCF loop converged' in ln: outer = True
        elif 'ENERGY| Total FORCE_EVAL' in ln and ') energy [' in ln: r['energy_Ha'] = _energy(ln, path)
        elif 'Integrated absolute spin density' in ln: r['spin'] = _number(ln)
    c = inner if outer is None else (outer if free or not outer else inner is True)
    r['converged'] = c if cdft is off else None if cdft is None else cdft and c; return r


def _pdos_last(path):
    lines = Path(path).read_text(errors='replace').splitlines(); starts = [i for i, ln in enumerate(lines) if ln.startswith('# Projected DOS for atomic kind')]
    if not starts: raise ValueError(f'read_pdos_kind: {path}: no projected density of states header')
    head = lines[starts[-1]]; step = _STEP.search(head); names, E, f, w = None, [], [], []
    for ln in lines[starts[-1] + 1:]:
        if ln.startswith('#'):
            if 'Occupation' in ln: names = ln.split('Occupation', 1)[1].split()
            continue
        t = ln.split()
        if not t: continue
        if names is not None and len(t) != 3 + len(names): raise ValueError(f'read_pdos_kind: {path}: a row has {len(t)} columns, the header {3 + len(names)}')
        if names is None: raise ValueError(f'read_pdos_kind: {path}: no Occupation column in the header (a PDOS curve file?)')
        E.append(float(t[1])); f.append(float(t[2])); w.append(float(t[3 + names.index('Total')]) if 'Total' in names else sum(float(x) for x in t[3:]))
    return _KIND.search(head).group(1), int(step.group(1)) if step else None, np.array(E), np.array(f), np.array(w)


def read_pdos_kind(path):
    """Kind, eigenvalues (Ha), occupations and projected weight of every state in the last block of one PDOS file."""
    kind, _, E, f, w = _pdos_last(path); return kind, E, f, w


def read_spectrum(folder, spin='ALPHA', pattern=None):
    """Energies (eV), occupations and per-kind weights of every state of one spin channel, sorted by energy."""
    folder = Path(folder); pat = (pattern or ('*-{spin}_k*-1.pdos' if spin else '*-k*-1.pdos')).format(spin=spin); got = {}
    paths = sorted(p for p in folder.glob(pat) if not _CURVE.search(p.name))
    if not paths: raise FileNotFoundError(f'read_spectrum: no file matches {pat} in {folder}')
    for p in paths:
        kind, step, E, f, w = _pdos_last(p)
        if kind in got: raise ValueError(f'read_spectrum: {folder}: kind {kind} appears in two files')
        got[kind] = (step, E, f, w)
    kinds = sorted(got); step0, E0, f0 = got[kinds[0]][:3]
    if any(got[k][0] != step0 for k in kinds): raise ValueError(f'read_spectrum: {folder}: the last blocks come from different steps {sorted({got[k][0] for k in kinds}, key=str)}')
    if any(len(got[k][1]) != len(E0) or not np.allclose(got[k][1], E0) for k in kinds): raise ValueError(f'read_spectrum: {folder}: the kinds list different states')
    o = np.argsort(E0, kind='stable'); return E0[o] * HARTREE_EV, f0[o], {k: got[k][3][o] for k in kinds}


def mulliken_blocks(path, header='Mulliken Population Analysis'):
    """Every complete Mulliken block of an output log as (species, net charge, spin moment) arrays."""
    lines = Path(path).read_text(errors='replace').splitlines(); heads = [i for i, ln in enumerate(lines) if header in ln] + [len(lines)]; out = []
    for k, s in enumerate(heads[:-1]):
        idx, sp, q, m, done = [], [], [], [], False
        for ln in lines[s + 1:heads[k + 1]]:
            if '# Total charge' in ln: done = True; break
            u, r = _MULLIKEN_UKS.match(ln), _MULLIKEN_RKS.match(ln)
            if u: idx.append(int(u.group(1))); sp.append(u.group(2)); q.append(float(u.group(5))); m.append(float(u.group(6)))
            elif r: idx.append(int(r.group(1))); sp.append(r.group(2)); q.append(float(r.group(4))); m.append(0.0)
        if not done or not idx: continue
        if idx != list(range(1, len(idx) + 1)): raise ValueError(f'mulliken_blocks: {path}: block {k + 1} has unreadable atom rows')
        out.append((np.array(sp), np.array(q), np.array(m)))
    return out


def trajectory_energies(path):
    """Total energy (Ha) of every frame of a position trajectory, from its comment lines (XYZ "E =" or extended XYZ "Energy=")."""
    out = []
    for ln in Path(path).read_text(errors='replace').splitlines():
        if 'i =' in ln and 'E =' in ln: v = ln.split('E =')[1].split(',')[0]
        elif 'Energy=' in ln: v = (ln.split('Energy=')[1].split() or [''])[0]
        else: continue
        try: out.append(float(v))
        except ValueError: raise ValueError(f'trajectory_energies: {path}: unreadable energy in: {ln.strip()}') from None
    if not out: raise ValueError(f'trajectory_energies: {path}: no comment line with an energy')
    return out


def cube_header(path):
    """Atom count, grid shape, voxel vectors (A, rows) and origin (A) of a cube file."""
    with open(Path(path), errors='replace') as f: f.readline(); f.readline(); h = [f.readline().split() for _ in range(4)]
    if any(len(r) < 4 or not all(_num(x) for x in r[:4]) for r in h): raise ValueError(f'cube_header: {path}: the header is truncated or unreadable')
    n = [int(r[0]) for r in h[1:]]
    if 0 in n or len({v > 0 for v in n}) != 1: raise ValueError(f'cube_header: {path}: voxel counts {n} mix signs or are zero')
    s = 1.0 if n[0] < 0 else BOHR_A
    return dict(n_atoms=abs(int(h[0][0])), shape=tuple(abs(v) for v in n), voxel_A=np.array([[float(x) for x in r[1:4]] for r in h[1:]]) * s, origin_A=np.array([float(x) for x in h[0][1:4]]) * s)
