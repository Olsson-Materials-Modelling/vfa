"""Where a charge or spin sits on the atoms, from per-atom populations along a relaxation."""
import numpy as np


def participation(moments):
    """Effective number of atoms carrying a per-atom quantity: (sum |m|)^2 / sum m^2."""
    a = np.abs(np.asarray(moments, float)); return float(a.sum() ** 2 / (a ** 2).sum()) if a.any() else np.nan


def carriers(moments, share=0.5):
    """Atoms carrying the given share of the total absolute quantity, largest first (0-based)."""
    a = np.abs(np.asarray(moments, float)); o = np.argsort(-a); c = np.cumsum(a[o]) / a.sum(); return o[:int(np.searchsorted(c, share)) + 1]


def track(blocks):
    """Per block along a relaxation: the atom with the largest moment, its share, the top-five share and the participation."""
    rows = []
    for k, (sp, q, m) in enumerate(blocks):
        a = np.abs(m); o = np.argsort(-a); tot = a.sum()
        rows.append(dict(step=k, top_atom=int(o[0]), top_species=str(sp[o[0]]), top_moment=float(m[o[0]]), top_charge=float(q[o[0]]), share_top1=float(a[o[0]] / tot) if tot else np.nan, share_top5=float(a[o[:5]].sum() / tot) if tot else np.nan, participation_atoms=participation(m), second_atom=int(o[1]), second_moment=float(m[o[1]])))
    return rows


def moved(blocks):
    """Whether the atom carrying the largest moment differs between the first and the last block."""
    first, last = list(blocks)[0], list(blocks)[-1]; return bool(np.argmax(np.abs(first[2])) != np.argmax(np.abs(last[2])))
