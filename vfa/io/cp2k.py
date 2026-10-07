"""CP2K files: the restart geometry, the output log, projected densities of states per kind, Mulliken blocks, cube headers."""
import re
import numpy as np

BOHR = 0.52917720859; HARTREE_EV = 27.211386245988
_UNIT = {'angstrom': 1.0, 'ang': 1.0, 'a': 1.0, 'bohr': BOHR, 'au': BOHR, 'a.u.': BOHR, 'nm': 10.0, 'pm': 0.01}
_KIND = re.compile(r'atomic kind\s+(\S+)'); _MULLIKEN_ROW = re.compile(r'^\s*(\d+)\s+([A-Za-z]{1,2})\s+(\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)(?:\s+(-?\d+\.\d+))?')


def _num(t):
    try: float(t); return True
    except ValueError: return False


def read_restart(path):
    """Symbols, positions (A) and cell (A) from the &COORD and &CELL blocks of a restart or input file; unit tokens honoured."""
    cellv, symbols, coords, scale_c, scale_p, in_cell, in_coord, depth = {}, [], [], 1.0, 1.0, False, False, 0
    for raw in path.read_text().splitlines():
        line = raw.split('#')[0].split('!')[0].strip()
        if not line: continue
        up = line.upper(); t = line.split()
        if up.startswith('&CELL') and not up.startswith('&CELL_OPT'): in_cell, depth = True, 1; continue
        if in_cell:
            if up.startswith('&END'): depth -= 1; in_cell = depth > 0; continue
            if up.startswith('&'): depth += 1; continue
            if t[0].upper() in 'ABC' and len(t) >= 4:
                u = None if _num(t[1]) else _UNIT.get(t[1].strip('[]').lower()); nums = [float(x) for x in t[1:] if _num(x)][:3]
                if len(nums) == 3: cellv[t[0].upper()] = np.array(nums) * (u or 1.0)
            elif t[0].upper() == 'UNIT' and len(t) >= 2: scale_c = _UNIT.get(t[1].strip('[]').lower(), 1.0)
            continue
        if up.startswith('&COORD'): in_coord = True; continue
        if in_coord:
            if up.startswith('&END'): in_coord = False; continue
            if t[0].upper() == 'UNIT' and len(t) >= 2: scale_p = _UNIT.get(t[1].strip('[]').lower(), 1.0); continue
            if len(t) >= 4 and all(_num(x) for x in t[1:4]): symbols.append(t[0]); coords.append([float(x) for x in t[1:4]])
    if not symbols or not {'A', 'B', 'C'} <= cellv.keys(): raise ValueError(f'{path}: incomplete &COORD or &CELL block')
    return np.array(symbols), np.array(coords) * scale_p, np.vstack([cellv['A'], cellv['B'], cellv['C']]) * scale_c


def read_output(path):
    """Ended, converged (outer SCF verdict first), total energy (Ha) and integrated absolute spin density of an output log."""
    r = dict(ended=False, converged=None, energy_Ha=None, spin=None); inner = outer = None
    for ln in path.read_text(errors='replace').splitlines():
        if 'PROGRAM ENDED AT' in ln: r['ended'] = True
        elif 'SCF run NOT converged' in ln: inner = False if inner is None else inner
        elif 'SCF run converged' in ln: inner = True
        elif 'outer SCF loop FAILED' in ln: outer = False
        elif 'outer SCF loop converged' in ln: outer = True
        elif 'ENERGY| Total FORCE_EVAL' in ln: r['energy_Ha'] = float(ln.split()[-1])
        elif 'Integrated absolute spin density' in ln: r['spin'] = float(ln.split()[-1])
    r['converged'] = (outer or bool(inner)) if outer is not None else inner; return r


def read_pdos_kind(path):
    """One projected density of states file: kind, eigenvalues (Ha), occupations and the summed projected weight of every state."""
    E, f, w = [], [], []
    with open(path, errors='replace') as fh:
        kind = _KIND.search(fh.readline()).group(1)
        for ln in fh:
            if ln.startswith('#'): continue
            t = ln.split(); E.append(float(t[1])); f.append(float(t[2])); w.append(sum(float(x) for x in t[3:]))
    return kind, np.array(E), np.array(f), np.array(w)


def read_spectrum(folder, spin='ALPHA', pattern='*-{spin}_k*-1.pdos'):
    """Energies (eV), occupations and per-kind weights of every state of one spin channel, sorted by energy; kinds matched on the file header."""
    got = {}
    for p in sorted(folder.glob(pattern.format(spin=spin))): kind, E, f, w = read_pdos_kind(p); got[kind] = (E, f, w)
    kinds = sorted(got); E0 = got[kinds[0]][0]
    if any(len(got[k][0]) != len(E0) or not np.allclose(got[k][0], E0) for k in kinds): raise ValueError(f'{folder}: the kinds list different states')
    o = np.argsort(E0); return E0[o] * HARTREE_EV, got[kinds[0]][1][o], {k: got[k][2][o] for k in kinds}


def mulliken_blocks(path, header='Mulliken Population Analysis'):
    """Every Mulliken block of an output log as (species, net charge, spin moment) arrays."""
    lines = path.read_text(errors='replace').splitlines()
    for s in [i for i, ln in enumerate(lines) if header in ln]:
        sp, q, m = [], [], []
        for ln in lines[s + 2:]:
            hit = _MULLIKEN_ROW.match(ln)
            if hit: sp.append(hit.group(2)); q.append(float(hit.group(6))); m.append(float(hit.group(7) or 0.0))
            elif sp: break
        if sp: yield np.array(sp), np.array(q), np.array(m)


def trajectory_energies(path):
    """Total energy (Ha) of every frame of a position trajectory, from its comment lines."""
    return [float(ln.split('E =')[1]) for ln in path.read_text(errors='replace').splitlines() if 'i =' in ln and 'E =' in ln]


def cube_header(path):
    """Number of atoms and grid shape of a cube file."""
    with open(path) as f:
        f.readline(); f.readline(); nat = abs(int(f.readline().split()[0])); return nat, tuple(int(f.readline().split()[0]) for _ in range(3))
