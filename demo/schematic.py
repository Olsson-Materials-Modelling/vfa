#!/usr/bin/env python3
"""The workflow diagram (fig00) drawn from the demo's outputs, written to demo/out and demo/figures."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'demo'))
import style
from vfa.structure.cell import minimum_image
from vfa.structure.graph import bond_graph
from vfa.structure.rings import primitive_rings
from vfa.corrections.schemes import transition_levels
from vfa.units import HARTREE_EV

OUT, FIG = ROOT / 'demo/out', ROOT / 'demo/figures'
ACCENT = {'structure': style.OKABE_ITO['blue'], 'voids': style.OKABE_ITO['vermillion'], 'electronic': style.OKABE_ITO['bluishgreen'], 'corrections': style.OKABE_ITO['purple'], 'statistics': style.OKABE_ITO['orange'], 'grey': style.GREY}
TINT = {'structure': '#EDF3FA', 'voids': '#FBEFE9', 'electronic': '#EAF5F0', 'corrections': '#F6EFF4', 'statistics': '#FEF7E3', 'grey': '#F3F3F3'}
X = [0.015, 0.265, 0.515, 0.765]; W = 0.22; GAP = 0.03; TITLE, BODY = 8.2, 6.3; LINE = 0.026


def box(ax, x, y, w, h, title, lines, kind):
    """A tile with a coloured title bar; returns its texts and frame for the overflow check."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.01', facecolor=TINT[kind], edgecolor='none'))
    ax.add_patch(FancyBboxPatch((x, y + h - 0.038), w, 0.038, boxstyle='round,pad=0,rounding_size=0.01', facecolor=ACCENT[kind], edgecolor='none')); ax.add_patch(plt.Rectangle((x, y + h - 0.045), w, 0.01, facecolor=ACCENT[kind], edgecolor='none'))
    out = [ax.text(x + 0.012, y + h - 0.019, title, fontsize=TITLE, fontweight='bold', color='white', ha='left', va='center')]
    for i, l in enumerate(lines): out.append(bullet(ax, x, y + h - 0.062 - LINE * i, l))
    return out, (x, y, w, h)


def bullet(ax, x, y, line):
    """One bulleted line of a box; a line starting with two spaces continues the previous item."""
    if not line.startswith('  '): ax.text(x + 0.013, y, '\u2022', fontsize=BODY, color=style.INK, ha='left', va='center')
    return ax.text(x + 0.027, y, line.strip(), fontsize=BODY, color=style.INK, ha='left', va='center')


def arrow(ax, p, q, color=style.GREY):
    """Arrow between two points of the diagram, grey unless a colour is given."""
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=8, color=color, linewidth=1.0, shrinkA=0, shrinkB=0))


def frame(ia):
    """Grey frame without ticks around an inset picture."""
    ia.set_xticks([]); ia.set_yticks([]); ia.minorticks_off()
    for s in ia.spines.values(): s.set_linewidth(0.5); s.set_color(style.GREY)


def picture_structure(ia, z, cutoff):
    """The bond graph of the example cell projected, one primitive ring highlighted."""
    sym, pos, cell = z['symbols'], z['positions'], z['cell']; e, o, _ = bond_graph(sym, pos, cell, {('A', 'B'): cutoff}); rings = primitive_rings(len(sym), e, o, 16)
    sizes = np.array([len(r) for r in rings]); ring = rings[int(np.flatnonzero(sizes == np.bincount(sizes).argmax())[0])] if len(rings) else ()
    for (i, j), s in zip(e, o):
        if not s.any(): ia.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], color=style.GREY, lw=0.5, zorder=1)
    ia.scatter(pos[sym == 'A', 0], pos[sym == 'A', 1], s=9, c=style.INK, zorder=2); ia.scatter(pos[sym == 'B', 0], pos[sym == 'B', 1], s=4, c='white', edgecolors=style.INK, linewidths=0.4, zorder=2)
    if len(ring):
        p = [pos[ring[0]]]
        for a in ring[1:]: p.append(p[-1] + minimum_image(pos[a] - p[-1], cell))
        p = np.array(p + [p[0]]); ia.plot(p[:, 0], p[:, 1], color=ACCENT['structure'], lw=2.0, zorder=3); ia.scatter(p[:-1, 0], p[:-1, 1], s=16, c=ACCENT['structure'], zorder=4)
    L = np.diag(cell); ia.set_xlim(0, L[0]); ia.set_ylim(0, L[1]); ia.set_aspect('equal'); frame(ia)


def picture_voids(ia, z):
    """The clearance slice with the probe isoline and the pores of the network."""
    cell, C, probe = z['cell'], z['slice'], float(z['probe']); L = np.diag(cell)
    ia.imshow(C.T, origin='lower', extent=(0, L[0], 0, L[1]), cmap='cividis', vmin=-1.0, vmax=max(2.0, C.max()))
    ia.contour(np.linspace(0, L[0], C.shape[0], endpoint=False) + L[0] / C.shape[0] / 2, np.linspace(0, L[1], C.shape[1], endpoint=False) + L[1] / C.shape[1] / 2, C.T, levels=[probe], colors=[ACCENT['voids']], linewidths=1.0)
    for c, r, p in zip(z['pore_xyz'], z['pore_r'], z['pore_percolating']): ia.add_patch(plt.Circle(c[:2], r, facecolor='none', edgecolor='white', lw=1.2 if p else 0.7, ls='-' if p else '--'))
    ia.set_xlim(0, L[0]); ia.set_ylim(0, L[1]); frame(ia)


def picture_electronic(ia, z):
    """The charged spectrum: participation against energy, the trap marked."""
    E, a, n, t = z['E1'] - float(z['E0'][int(z['n0']) - 1]), z['a1'], int(z['n1']), int(z['trap']); keep = (E > -7) & (E < 13); occ = np.arange(E.size) < n
    ia.scatter(E[keep & occ], a[keep & occ], s=5, c=style.OKABE_ITO['blue'], lw=0); ia.scatter(E[keep & ~occ], a[keep & ~occ], s=5, c=style.OKABE_ITO['vermillion'], lw=0); ia.scatter([E[t]], [a[t]], s=26, c=ACCENT['electronic'], edgecolors=style.INK, linewidths=0.5, zorder=3)
    ia.axhline(float(z['n_atoms']) / 5, color=style.GREY, ls=':', lw=0.8); ia.set_yscale('log'); frame(ia)


def picture_corrections(ia):
    """Formation energy against Fermi level of the oxygen vacancy in MgO, uncorrected (dashed) and with the static LZ correction (solid), and the transition levels."""
    m = json.loads((ROOT / 'demo/data/mgo_vacancy.json').read_text()); e0 = next(s['energy_Ha'] for s in m['states'] if s['charge'] == 0); gap = m['host_gap_eV']; fermi = np.array([0.0, gap])
    colour = {0: style.INK, 1: style.OKABE_ITO['blue'], 2: style.OKABE_ITO['vermillion']}
    for corrected, ls, face in ((False, (0, (2, 1.5)), 'white'), (True, '-', style.INK)):
        f = {s['charge']: (s['energy_Ha'] - e0) * HARTREE_EV + s['charge'] * m['host_vbm_eV'] + corrected * s['correction_eV'] for s in m['states']}
        for q, e in f.items(): ia.plot(fermi, e + q * fermi, color=colour[q], ls=ls, lw=1.0 if corrected else 0.7)
        for t in transition_levels(f):
            if 0 <= t['level_eV'] <= gap: ia.plot([t['level_eV']], [f[t['charges'][1]] + t['charges'][1] * t['level_eV']], 'o', ms=3, mfc=face, mec=style.INK, mew=0.6, zorder=3)
    for q, x, y in ((0, 0.97, -0.55), (1, 0.97, 1.1), (2, 0.5, 2.2)): ia.text(gap * x, y, f'+{q}' if q else '0', color=colour[q], fontsize=BODY, ha='right', va='center')
    ia.text(0.04, 0.95, 'MgO $V_O$, LZ', transform=ia.transAxes, fontsize=BODY, ha='left', va='top'); ia.set_xlim(0, gap); ia.set_ylim(-4.3, 5.3); frame(ia)


def picture_statistics(ia, c, curves):
    """A trend along the ordering column: the cells, the smoothed curve."""
    ia.scatter(c.number_density, c['fraction_p0.8'], s=5, c=style.GREY, lw=0); ia.plot(curves.number_density, curves['fraction_p0.8'], color=ACCENT['statistics'], lw=1.6); frame(ia)


def main():
    """The workflow diagram, with an overflow check on every text box."""
    style.apply(); FIG.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True); fig, ax = plt.subplots(figsize=(style.TWO_COLUMN, 5.9)); fig.subplots_adjust(0, 0, 1, 1); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off'); items = []
    z = np.load(OUT / 'example_cell.npz'); sp = np.load(OUT / 'spectrum_example.npz'); cutoff = float(np.load(OUT / 'rdf.npz')['cutoff']); c = pd.read_csv(OUT / 'cells.csv'); curves = pd.read_csv(OUT / 'trend_curves.csv')
    y1, h1 = 0.855, 0.115; y2, h2 = 0.395, 0.43; y3, h3 = 0.115, 0.24; y4, h4 = 0.015, 0.08
    items.append(box(ax, X[0], y1, 2 * W + GAP, h1, 'structures and radii', ['extended XYZ, CP2K restart, LAMMPS data', 'radii: centre, covalent, ionic, or your own table'], 'grey'))
    items.append(box(ax, X[2], y1, 2 * W + GAP, h1, 'spectra, energies, corrector output', ['energies, occupations, weights, participation', 'total energies; potential and density JSON'], 'grey'))
    parts = [('structure', ['bond graph, offsets', 'coordination, angles,', '  tetrahedral order', 'rings: Franzblau, King,', '  Guttman; extents', 'g(r), cutoff; S(Q), G(r)']),
             ('voids', ['network: Di, Df, Dif,', '  pores, percolation', 'clearance, grid or', '  points: fraction, surface', 'free volume per atom', 'cavity distance']),
             ('electronic', ['species character', 'semicore alignment', 'band ladder, tail check', 'trap by character', 'onset, thresholds swept', 'trapping energies']),
             ('corrections', ['gates on single-point pairs', 'corrector output by path:', '  selected totals, schemes,', '  reliability, level shift', 'corrected formation', '  energies, transition levels'])]
    for x, (t, l) in zip(X, parts): items.append(box(ax, x, y2, W, h2, t, l, t))
    pics = [ax.inset_axes([x + 0.012, y2 + 0.012, W - 0.024, 0.195]) for x in X[:3]] + [ax.inset_axes([X[3] + 0.012, y2 + 0.034, W - 0.024, 0.173])]
    picture_structure(pics[0], z, cutoff); picture_voids(pics[1], z); picture_electronic(pics[2], sp); picture_corrections(pics[3])
    items.append(box(ax, X[0], y3, 4 * W + 3 * GAP, h3, 'statistics', [], 'statistics'))
    cols = [['intervals: bootstrap,', '  Clopper-Pearson,', '  simultaneous, Wilson'], ["contrasts: Cliff's delta,", '  scale ratio, shift'], ['trends: smoothing in', '  order, held-out window', 'change: Theil-Sen and', '  least squares']]
    for k, lines in enumerate(cols):
        for i, l in enumerate(lines): items[-1][0].append(bullet(ax, X[k], y3 + h3 - 0.062 - LINE * i, l))
    items[-1][0].append(bullet(ax, X[0], y3 + h3 - 0.062 - LINE * 3, 'held-out models; the cell is the sampling unit'))
    picture_statistics(ax.inset_axes([X[3] + 0.012, y3 + 0.012, W - 0.024, h3 - 0.06]), c, curves)
    items.append(box(ax, X[0], y4, 4 * W + 3 * GAP, h4, 'provenance', [], 'grey')); items[-1][0].append(bullet(ax, X[0], y4 + 0.019, 'run context (code and inputs hashed, settings recorded) and a sidecar beside every table'))
    for x in X: arrow(ax, (x + W / 2, y1), (x + W / 2, y2 + h2)); arrow(ax, (x + W / 2, y2), (x + W / 2, y3 + h3))
    arrow(ax, (X[2] + W, y2 + h2 - 0.05), (X[3], y2 + h2 - 0.05), ACCENT['electronic']); arrow(ax, (0.5, y3), (0.5, y4 + h4), ACCENT['statistics'])
    fig.canvas.draw(); r = fig.canvas.get_renderer(); bad = []
    for texts, (x, y, w, _) in items:
        for t in texts:
            bb = t.get_window_extent(r).transformed(ax.transAxes.inverted())
            if bb.x1 > x + w - 0.006 or bb.y0 < y + 0.004: bad.append(t.get_text())
    if bad: raise SystemExit('text does not fit its box: ' + '; '.join(bad))
    style.save(fig, OUT, FIG, 'fig00_workflow', stamp=None)


if __name__ == '__main__':
    main()
