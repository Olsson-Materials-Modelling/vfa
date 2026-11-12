"""Where a charge or spin sits on the atoms, from per-atom populations along a relaxation."""
import numpy as np


def participation(moments):
    """Effective number of atoms carrying a per-atom quantity: (sum |m|)^2 / sum m^2."""
    a = np.abs(np.asarray(moments, float)); return float(a.sum() ** 2 / (a ** 2).sum()) if a.any() else np.nan


def carriers(moments, share=0.5):
    """Atoms carrying the given share of the total absolute quantity, largest first, ties in atom order (0-based)."""
    a = np.abs(np.asarray(moments, float))
    if a.size == 0 or not np.all(np.isfinite(a)) or not a.any(): raise ValueError('carriers: the per-atom quantities must be finite, not all zero')
    if not 0 < share <= 1: raise ValueError(f'carriers: share must lie in (0, 1], not {share}')
    o = np.argsort(-a, kind='stable'); c = np.cumsum(a[o]) / a.sum(); return o[:int(np.searchsorted(c, share)) + 1]


def track(blocks):
    """Per block along a relaxation: the atom with the largest moment, its share, the top-five share, the participation, the runner-up."""
    rows = []
    for k, (sp, q, m) in enumerate(blocks):
        if len(m) == 0: raise ValueError(f'track: block {k} holds no atoms')
        a = np.abs(m); o = np.argsort(-a, kind='stable'); tot = a.sum()
        rows.append(dict(step=k, top_atom=int(o[0]), top_species=str(sp[o[0]]), top_moment=float(m[o[0]]), top_charge=float(q[o[0]]), share_top1=float(a[o[0]] / tot) if tot else np.nan, share_top5=float(a[o[:5]].sum() / tot) if tot else np.nan, participation_atoms=participation(m), second_atom=int(o[1]) if len(o) > 1 else -1, second_moment=float(m[o[1]]) if len(o) > 1 else np.nan))
    return rows


def moved(blocks):
    """Whether the atom carrying the largest moment differs between the first and the last of the blocks (any iterable)."""
    blocks = list(blocks)
    if not blocks: raise ValueError('moved: no blocks')
    return bool(np.argmax(np.abs(blocks[0][2])) != np.argmax(np.abs(blocks[-1][2])))
