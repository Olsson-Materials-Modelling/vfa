#!/usr/bin/env python3
"""Figures of the demo (demo/out/, demo/figures/): the clearance field and pore network of one cell, the probe sweep, trends with density, per-atom quantities, structural descriptors."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'demo'))
import style
from vfa.structure.cell import minimum_image

OUT, FIG = ROOT / 'demo/out', ROOT / 'demo/figures'


def ecdf(ax, x, **kw):
    x = np.sort(np.asarray(x)[np.isfinite(x)]); ax.step(x, np.arange(1, len(x) + 1) / len(x), where='post', **kw)


def fig01(z):
    """The clearance field of one cell in a slice, and its pore network."""
    cell, pos, sym, C, probe = z['cell'], z['positions'], z['symbols'], z['slice'], float(z['probe']); L = np.diag(cell); fig, (a, b) = plt.subplots(1, 2, figsize=(style.TWO_COLUMN, 3.5), constrained_layout=True)
    im = a.imshow(C.T, origin='lower', extent=(0, L[0], 0, L[1]), cmap='cividis', vmin=-1.0, vmax=max(2.0, C.max())); a.contour(np.linspace(0, L[0], C.shape[0], endpoint=False) + L[0] / C.shape[0] / 2, np.linspace(0, L[1], C.shape[1], endpoint=False) + L[1] / C.shape[1] / 2, C.T, levels=[probe], colors=[style.OKABE_ITO['vermillion']], linewidths=1.2)
    dz = pos[:, 2] - float(z['slice_z']); slab = np.abs(dz - np.rint(dz / L[2]) * L[2]) < 1.0
    for s, mk, fc in (('A', 'o', style.INK), ('B', 'o', 'white')): m = slab & (sym == s); a.scatter(pos[m, 0], pos[m, 1], s=22 if s == 'A' else 12, marker=mk, facecolors=fc, edgecolors=style.INK, linewidths=0.7, zorder=3)
    cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.02); cb.set_label('clearance (Å)'); a.set_xlabel('x (Å)'); a.set_ylabel('y (Å)'); a.set_xlim(0, L[0]); a.set_ylim(0, L[1]); style.style_axes(a, minor_x=False, minor_y=False); style.panel_label(a, 'a')
    a.text(0.03, 0.97, f'slice at z = {float(z["slice_z"]):.1f} Å\nisoline: probe {probe:g} Å', transform=a.transAxes, ha='left', va='top', fontsize=8, bbox=dict(facecolor='white', edgecolor=style.INK, linewidth=0.6, pad=2))
    P, R, perc, tp = z['pore_xyz'], z['pore_r'], z['pore_percolating'], z['throat_pairs']
    for i, j in tp:
        d = minimum_image(P[j] - P[i], cell); b.plot([P[i, 0], P[i, 0] + d[0]], [P[i, 1], P[i, 1] + d[1]], color=style.GREY, lw=0.8, zorder=1); b.plot([P[j, 0], P[j, 0] - d[0]], [P[j, 1], P[j, 1] - d[1]], color=style.GREY, lw=0.8, zorder=1)
    for i in range(len(R)): b.add_patch(plt.Circle(P[i, :2], R[i], facecolor=style.OKABE_ITO['vermillion'] if perc[i] else 'white', edgecolor=style.OKABE_ITO['vermillion'] if perc[i] else style.OKABE_ITO['blue'], lw=1.0, alpha=0.9, zorder=2))
    b.plot([], [], 'o', mfc=style.OKABE_ITO['vermillion'], mec=style.OKABE_ITO['vermillion'], label='percolating pore'); b.plot([], [], 'o', mfc='white', mec=style.OKABE_ITO['blue'], label='isolated pore'); b.plot([], [], color=style.GREY, lw=0.8, label='throat')
    b.set_xlim(0, L[0]); b.set_ylim(0, L[1]); b.set_aspect('equal'); b.set_xlabel('x (Å)'); b.set_ylabel('y (Å)'); style.style_axes(b, minor_x=False, minor_y=False); style.legend(b, loc='upper right', fontsize=8); style.panel_label(b, 'b')
    b.text(0.03, 0.03, f'probe {probe:g} Å: {len(R)} pores, {int(perc.sum())} percolating\n$D_i$ {float(z["Di_A"]):.2f} Å, $D_f$ {float(z["Df_A"]):.2f} Å', transform=b.transAxes, ha='left', va='bottom', fontsize=8, bbox=dict(facecolor='white', edgecolor=style.INK, linewidth=0.6, pad=2))
    style.save(fig, OUT, FIG, 'fig01_clearance_and_pores')


def fig02(sw):
    """Accessible void fraction and specific surface against the probe radius at the lowest and highest density."""
    lo, hi = sw.number_density.min(), sw.number_density.max(); col, mk = style.series(['low', 'high']); fig, axs = plt.subplots(1, 2, figsize=(style.TWO_COLUMN, 3.0), constrained_layout=True)
    for ax, col_name, lab in zip(axs, ('fraction', 'specific_surface'), ('accessible void fraction', 'void surface (Å$^2$ per 1000 Å$^3$)')):
        for rho in sorted(sw.number_density.unique()):
            g = sw[sw.number_density == rho].groupby('probe_A')[col_name]; m, a_, b_ = g.median(), g.min(), g.max(); key = 'low' if rho == lo else 'high' if rho == hi else None
            if key: ax.fill_between(m.index, a_, b_, color=col[key], alpha=0.2, lw=0); ax.plot(m.index, m, marker=mk[key], color=col[key], label=f'{key} density, {rho:.3f} atoms Å$^{{-3}}$')
            else: ax.plot(m.index, m, color=style.GREY, lw=0.8, zorder=0)
        ax.set_xlabel('probe radius (Å)'); ax.set_ylabel(lab); style.style_axes(ax)
    axs[0].plot([], [], color=style.GREY, lw=0.8, label='intermediate densities'); style.legend(axs[0], loc='upper right', fontsize=8); style.panel_label(axs[0], 'a'); style.panel_label(axs[1], 'b'); style.save(fig, OUT, FIG, 'fig02_probe_sweep')


def fig03(c):
    """Largest included and free spheres, the accessible fraction and the ring-size shares against density."""
    fig, axs = plt.subplots(1, 3, figsize=(style.TWO_COLUMN, 3.0), constrained_layout=True); g = c.groupby('number_density'); x = np.array(sorted(c.number_density.unique())); col, mk = style.series(['Di', 'Df', 'low', 'high'])
    for key, name, lab in (('Di', 'Di_A', 'largest included sphere $D_i$'), ('Df', 'Df_A', 'largest free sphere $D_f$')):
        q = g[name].quantile([.25, .5, .75]).unstack(); axs[0].errorbar(x, q[.5], yerr=[q[.5] - q[.25], q[.75] - q[.5]], marker=mk[key], color=col[key], capsize=2, lw=1.2, label=lab)
    lo_, hi_ = g['Df_A'].quantile(.25).min(), g['Di_A'].quantile(.75).max(); axs[0].set_ylim(lo_ - 0.4, hi_ + 1.4); axs[0].set_xlabel('number density (atoms Å$^{-3}$)'); axs[0].set_ylabel('diameter (Å)'); style.legend(axs[0], loc='upper right', fontsize=8)
    q = g['fraction_p0.8'].quantile([.25, .5, .75]).unstack(); axs[1].errorbar(x, q[.5], yerr=[q[.5] - q[.25], q[.75] - q[.5]], marker='o', color=style.INK, capsize=2, lw=1.2); axs[1].set_xlabel('number density (atoms Å$^{-3}$)'); axs[1].set_ylabel('accessible void fraction\n(probe 0.8 Å)')
    sizes = sorted(int(k[4:-6]) for k in c.columns if k.startswith('ring') and k.endswith('_share')); w = 0.38
    for i, (key, rho) in enumerate((('low', x[0]), ('high', x[-1]))):
        sh = c[c.number_density == rho][[f'ring{s}_share' for s in sizes]].mean() * 100; axs[2].bar(np.array(sizes) + (i - 0.5) * w, sh, width=w, color=col[key], edgecolor=style.INK, lw=0.5, label=f'{key} density')
    axs[2].set_xlabel('ring size (A atoms)'); axs[2].set_ylabel('share of primitive rings (%)'); axs[2].set_xticks(sizes); style.legend(axs[2], loc='upper left', fontsize=8)
    for ax, l in zip(axs, 'abc'): style.style_axes(ax, minor_x=ax is not axs[2]); style.panel_label(ax, l)
    style.save(fig, OUT, FIG, 'fig03_density_trends')


def fig04(at):
    """Per-atom free volume by species at the lowest and highest density, and the cavity distance of three- and four-fold A atoms."""
    lo, hi = at.number_density.min(), at.number_density.max(); col, mk = style.series(['low', 'high']); fig, axs = plt.subplots(1, 2, figsize=(style.TWO_COLUMN, 3.0), constrained_layout=True)
    for key, rho in (('low', lo), ('high', hi)):
        for sp, ls in (('A', '-'), ('B', '--')): ecdf(axs[0], at[(at.number_density == rho) & (at.species == sp)].free_volume_A3, color=col[key], ls=ls, label=f'{sp}, {key} density')
    axs[0].set_xlabel('free volume at the atom, probe 0.8 Å (Å$^3$)'); axs[0].set_ylabel('cumulative fraction of atoms'); style.legend(axs[0], loc='lower right', fontsize=8)
    a = at[at.species == 'A']; groups = [a[a.coordination == z].cavity_distance_A.dropna() for z in (3, 4)]
    bp = axs[1].boxplot(groups, widths=0.5, patch_artist=True, medianprops=dict(color=style.INK), flierprops=dict(marker='o', markersize=3, markerfacecolor='none', markeredgecolor=style.GREY))
    for patch, c in zip(bp['boxes'], (style.OKABE_ITO['vermillion'], style.OKABE_ITO['blue'])): patch.set(facecolor=c, alpha=0.5, edgecolor=style.INK)
    axs[1].set_xticks([1, 2]); axs[1].set_xticklabels(['three-fold A', 'four-fold A']); axs[1].set_ylabel('distance to the nearest pore centre (Å)'); axs[1].text(0.97, 0.03, f'n = {len(groups[0])}, {len(groups[1])}', transform=axs[1].transAxes, ha='right', va='bottom', fontsize=8)
    for ax, l in zip(axs, 'ab'): style.style_axes(ax, minor_x=ax is axs[0]); style.panel_label(ax, l)
    style.save(fig, OUT, FIG, 'fig04_sites')


def fig05(c, rdf, ang):
    """The A-B pair correlation with the measured bond cutoff, the bond-angle distributions and the tetrahedral order."""
    lo, hi = c.number_density.min(), c.number_density.max(); col, mk = style.series(['low', 'high']); fig, axs = plt.subplots(1, 3, figsize=(style.TWO_COLUMN, 3.0), constrained_layout=True); keys = list(rdf['keys'])
    for key, rho in (('low', lo), ('high', hi)): axs[0].plot(rdf['r'], rdf['curves'][keys.index(f'{rho:.3f}:AB')], color=col[key], label=f'{key} density')
    axs[0].axvline(float(rdf['cutoff']), color=style.INK, ls=':', lw=1); axs[0].text(float(rdf['cutoff']) + 0.1, axs[0].get_ylim()[1] * 0.5, f'bond cutoff\n{float(rdf["cutoff"]):.2f} Å', fontsize=8, va='center'); axs[0].set_xlim(1.0, 6.0); axs[0].set_xlabel('r (Å)'); axs[0].set_ylabel('$g_{AB}(r)$'); style.legend(axs[0], loc='upper right', fontsize=8)
    cells = c.cell.tolist(); bab = np.concatenate([ang[f'{n}:BAB'] for n in cells]); aba = np.concatenate([ang[f'{n}:ABA'] for n in cells]); q = np.concatenate([ang[f'{n}:q'] for n in cells])
    axs[1].hist(bab, bins=np.linspace(60, 180, 61), density=True, histtype='step', color=col['low'], lw=1.3, label='B–A–B'); axs[1].hist(aba, bins=np.linspace(60, 180, 61), density=True, histtype='step', color=col['high'], lw=1.3, label='A–B–A')
    axs[1].axvline(np.degrees(np.arccos(-1 / 3)), color=style.GREY, ls=':', lw=1, label='tetrahedral angle'); axs[1].set_xlabel('bond angle (°)'); axs[1].set_ylabel('probability density (°$^{-1}$)'); style.legend(axs[1], loc='upper left', fontsize=8)
    axs[2].hist(q[np.isfinite(q)], bins=np.linspace(0.6, 1.0, 41), density=True, color=style.OKABE_ITO['blue'], alpha=0.6, edgecolor=style.INK, lw=0.4); axs[2].set_xlabel('tetrahedral order $q$ of A'); axs[2].set_ylabel('probability density')
    for ax, l in zip(axs, 'abc'): style.style_axes(ax); style.panel_label(ax, l)
    style.save(fig, OUT, FIG, 'fig05_structure')


def main():
    style.apply(); FIG.mkdir(exist_ok=True); c = pd.read_csv(OUT / 'cells.csv'); sw = pd.read_csv(OUT / 'sweep.csv'); at = pd.read_csv(OUT / 'atoms.csv')
    fig01(np.load(OUT / 'example_cell.npz', allow_pickle=True)); fig02(sw); fig03(c); fig04(at); fig05(c, np.load(OUT / 'rdf.npz', allow_pickle=True), np.load(OUT / 'angles.npz', allow_pickle=True))
    fig06(np.load(OUT / 'spectrum_example.npz', allow_pickle=True)); ev = pd.read_csv(OUT / 'events.csv'); fig07(c.merge(ev.drop(columns='number_density'), on='cell'), pd.read_csv(OUT / 'trend_curves.csv'), pd.read_csv(OUT / 'trend_change.csv'), pd.read_csv(OUT / 'model.csv'), pd.read_csv(OUT / 'learning_curve.csv')); fig08(pd.read_csv(OUT / 'modes.csv'), pd.read_csv(OUT / 'convergence.csv')); fig09(np.load(OUT / 'example_cell.npz', allow_pickle=True))



def fig06(z):
    """One cell before and after trapping: the band ladder, the precursor and the trap by character, the onsets."""
    fig, axs = plt.subplots(1, 2, figsize=(style.TWO_COLUMN, 3.2), constrained_layout=True); col, _ = style.series(['A', 'B']); cmap = {True: style.OKABE_ITO['blue'], False: style.OKABE_ITO['vermillion']}
    ref = float(z['E0'][int(z['n0']) - 1])                                                                      # the neutral HOMO on the aligned scale
    for ax, tag, lab in ((axs[0], 0, 'neutral'), (axs[1], 1, 'charged')):
        E, f, a, n = z[f'E{tag}'] - ref, z[f'f{tag}'], z[f'a{tag}'], int(z[f'n{tag}']); occ = np.arange(E.size) < n; keep = (E > -8) & (E < 14)
        ax.scatter(E[keep], a[keep], c=[cmap[o] for o in occ[keep]], s=14 + 40 * f[keep], marker='o', edgecolors=style.INK, linewidths=0.4, alpha=0.85)
        ax.axhline(float(z['n_atoms']) / 5, color=style.GREY, ls=':', lw=1); ax.axhline(float(z['n_atoms']) / 3, color=style.GREY, ls='--', lw=1); ax.set_yscale('log'); ax.set_xlabel('energy from the neutral HOMO, aligned (eV)'); ax.set_ylabel('participation (atoms)'); ax.set_title(lab, fontsize=10)
        style.style_axes(ax, minor_y=False)
    E0, n0 = z['E0'] - ref, int(z['n0']); axs[0].annotate('precursor', (E0[n0], z['a0'][n0]), (E0[n0] - 5.0, z['a0'][n0] * 0.55), fontsize=8, arrowprops=dict(arrowstyle='-', lw=0.6)); axs[0].annotate(f'onset 1/5', (E0[int(z['onset5'])], z['a0'][int(z['onset5'])]), (E0[int(z['onset5'])] + 0.6, z['a0'][int(z['onset5'])] * 0.4), fontsize=8, arrowprops=dict(arrowstyle='-', lw=0.6))
    E1, t = z['E1'] - ref, int(z['trap']); axs[1].annotate(f'trap, A character {float(z["f1"][t]):.2f}', (E1[t], z['a1'][t]), (E1[t] + 1.2, z['a1'][t] * 1.6), fontsize=8, arrowprops=dict(arrowstyle='-', lw=0.6))
    axs[0].scatter([], [], c=cmap[True], s=20, edgecolors=style.INK, linewidths=0.4, label='occupied'); axs[0].scatter([], [], c=cmap[False], s=20, edgecolors=style.INK, linewidths=0.4, label='empty'); axs[0].plot([], [], color=style.GREY, ls=':', label='1/5 of the atoms'); axs[0].plot([], [], color=style.GREY, ls='--', label='1/3 of the atoms'); style.legend(axs[0], loc='lower left', fontsize=7.5)
    axs[1].text(0.03, 0.97, 'marker size: A character', transform=axs[1].transAxes, ha='left', va='top', fontsize=7.5); style.panel_label(axs[0], 'a'); style.panel_label(axs[1], 'b'); style.save(fig, OUT, FIG, 'fig06_spectrum_reading')


def fig07(c, curves, change, model, lc):
    """Trends with density with the held-out window and both straight-line changes; the held-out prediction of the trap kind."""
    fig, axs = plt.subplots(1, 3, figsize=(style.TWO_COLUMN, 3.0), constrained_layout=True); col, mk = style.series(['smoothed', 'theil_sen', 'least_squares', 'observed', 'predicted'])
    for ax, q, lab in ((axs[0], 'fraction_p0.8', 'accessible void fraction, probe 0.8 Å'), (axs[1], 'trap_below_edge_eV', 'trap level below the band edge (eV)')):
        ax.scatter(c.number_density, c[q], s=12, color=style.GREY, edgecolors=style.INK, linewidths=0.3, label='cells'); ax.plot(curves.number_density, curves[q], color=col['smoothed'], lw=1.6, label=f"smoothed, {int(change.window_cells[0])} cells")
        r = change[change.quantity == q].iloc[0]; x = np.array([r.density_low, r.density_high])
        for key, lab2, ls in (('theil_sen', 'Theil–Sen', '-'), ('least_squares', 'least squares', '--')): ax.plot(x, [r[f'{key}_low'], r[f'{key}_high']], color=col[key], lw=1.2, ls=ls, label=lab2)
        txt = '\n'.join(f"{lab2} {r[f'{key}_change']:+.2f} [{r[f'{key}_lower']:+.2f}, {r[f'{key}_upper']:+.2f}]" for key, lab2 in (('theil_sen', 'Theil–Sen'), ('least_squares', 'least squares')))
        ax.text(*((0.03, 0.03) if q == 'fraction_p0.8' else (0.97, 0.03)), txt, transform=ax.transAxes, ha='left' if q == 'fraction_p0.8' else 'right', va='bottom', fontsize=6.5, bbox=dict(facecolor='white', edgecolor=style.INK, linewidth=0.5, pad=2))
        if q != 'fraction_p0.8': ax.set_ylim(c[q].min() - 0.35, c[q].max() + 0.3)
        ax.set_xlabel('number density (atoms Å$^{-3}$)'); ax.set_ylabel(lab); style.legend(ax, loc='upper right' if q == 'fraction_p0.8' else 'upper left', fontsize=6.5)
    x = np.arange(len(model)); w = 0.38; axs[2].bar(x - w / 2, model.observed, w, color=col['observed'], edgecolor=style.INK, lw=0.5, label='observed, test half'); axs[2].errorbar(x - w / 2, model.observed, yerr=[model.observed - model.wilson_lower, model.wilson_upper - model.observed], fmt='none', ecolor=style.INK, capsize=2)
    axs[2].bar(x + w / 2, model.predicted, w, color=col['predicted'], edgecolor=style.INK, lw=0.5, label='predicted, fitted on the other half'); axs[2].errorbar(x + w / 2, model.predicted, yerr=[model.predicted - model.lower, model.upper - model.predicted], fmt='none', ecolor=style.INK, capsize=2)
    axs[2].set_xticks(x); axs[2].set_xticklabels(model.kind); axs[2].set_ylabel('share of cells'); axs[2].set_ylim(0, 1); style.legend(axs[2], loc='upper right', fontsize=7)
    for ax, l in zip(axs, 'abc'): style.style_axes(ax, minor_x=ax is not axs[2]); style.panel_label(ax, l)
    style.save(fig, OUT, FIG, 'fig07_trends_and_model')


def fig08(m, conv):
    """The modes side by side: grid against pointwise under three radii conventions, the Voronoi sphere against the grid's largest clearance, the ring criteria, the convergence checks."""
    fig, axs = plt.subplots(1, 4, figsize=(style.TWO_COLUMN, 2.9), constrained_layout=True); col, mk = style.series(['centre', 'covalent', 'ionic', 'primitive', 'king', 'guttman'])
    for r in ('centre', 'covalent', 'ionic'): axs[0].scatter(m[f'{r}_grid_fraction'], m[f'{r}_pointwise_fraction'], s=14, marker=mk[r], color=col[r], edgecolors=style.INK, linewidths=0.3, label=r)
    lim = (0, 1); axs[0].plot(lim, lim, color=style.GREY, lw=0.8, zorder=0); axs[0].set_xlim(lim); axs[0].set_ylim(lim); axs[0].set_xlabel('accessible fraction, grid'); axs[0].set_ylabel('accessible fraction, pointwise'); style.legend(axs[0], loc='upper left', fontsize=7.5)
    for r in ('centre', 'covalent', 'ionic'): axs[1].scatter(m[f'{r}_Di_A'], m[f'{r}_grid_largest_A'], s=14, marker=mk[r], color=col[r], edgecolors=style.INK, linewidths=0.3)
    lo, hi = min(m[[f'{r}_Di_A' for r in ('centre', 'covalent', 'ionic')]].min()), max(m[[f'{r}_Di_A' for r in ('centre', 'covalent', 'ionic')]].max()); axs[1].plot([lo, hi], [lo, hi], color=style.GREY, lw=0.8, zorder=0); axs[1].set_xlabel('$D_i$, Voronoi network (Å)'); axs[1].set_ylabel('largest grid clearance, ×2 (Å)')
    sizes = range(3, 13); w = 0.27
    for i, crit in enumerate(('primitive', 'king', 'guttman')): axs[2].bar(np.array(sizes) + (i - 1) * w, [m[f'{crit}_size{s}'].mean() for s in sizes], w, color=col[crit], edgecolor=style.INK, lw=0.4, label=crit)
    axs[2].set_ylim(0, 1.25 * max(m[f'{crit}_size{s}'].mean() for crit in ('primitive', 'king', 'guttman') for s in sizes)); axs[2].set_xlabel('ring size (A atoms)'); axs[2].set_ylabel('rings per cell, up to 24 atoms'); axs[2].set_xticks(list(sizes)[::3]); style.legend(axs[2], loc='upper right', fontsize=7.5)
    p = conv[conv['mode'] == 'pointwise']; g = conv[conv['mode'] == 'grid']; V = float(pd.read_csv(ROOT / 'example/data/cells.csv').set_index('cell').volume_A3[conv.cell.iloc[0]])
    axs[3].plot(p.n, p.fraction, marker='o', color=col['covalent'], label='pointwise: sample points'); axs[3].plot(V / g.dx ** 3, g.fraction, marker='s', color=col['ionic'], label='grid: grid points'); axs[3].set_xscale('log'); axs[3].set_xlabel('evaluation points'); axs[3].set_ylabel('accessible fraction, probe 0.8 Å')
    y = conv.fraction; axs[3].set_ylim(y.min() - 0.004, y.max() + 0.006); style.legend(axs[3], loc='upper left', fontsize=7)
    for ax, l in zip(axs, 'abcd'): style.style_axes(ax, minor_x=ax not in (axs[2], axs[3])); style.panel_label(ax, l)
    style.save(fig, OUT, FIG, 'fig08_modes')


def fig09(z):
    """The void space of one cell in three dimensions: the surface at the probe radius, the atoms, and the pore network with its percolating pores."""
    from skimage.measure import marching_cubes
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    cell, pos, sym, C, probe, n = z['cell'], z['positions'], z['symbols'], z['field'].astype(float), float(z['probe']), z['shape']; L = np.diag(cell)
    verts, faces, normals, _ = marching_cubes(C, level=probe); xyz = (verts / np.asarray(n, float)) @ cell                      # fractional -> cartesian, one period
    fig = plt.figure(figsize=(style.TWO_COLUMN, 3.6)); light = np.array([0.4, 0.3, 0.85]); light /= np.linalg.norm(light)
    for k, (title, show_atoms, show_pores) in enumerate((('void surface and atoms', True, False), ('void surface and pore network', False, True))):
        ax = fig.add_subplot(1, 2, k + 1, projection='3d'); tri = xyz[faces]; fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); fn /= np.linalg.norm(fn, axis=1)[:, None] + 1e-12
        shade = 0.55 + 0.45 * np.clip(np.abs(fn @ light), 0, 1); cols = np.c_[shade * 0.47, shade * 0.70, shade * 0.86, np.full(len(tri), 0.55 if show_atoms else 0.35)]
        ax.add_collection3d(Poly3DCollection(tri, facecolors=cols, edgecolors='none'))
        if show_atoms:
            for s, size, fc in (('A', 26, style.INK), ('B', 12, 'white')): m = sym == s; ax.scatter(pos[m, 0], pos[m, 1], pos[m, 2], s=size, c=fc, edgecolors=style.INK, linewidths=0.5, depthshade=False)
        if show_pores:
            P, R, perc, tp = z['pore_xyz'], z['pore_r'], z['pore_percolating'], z['throat_pairs']; P = P % L
            for i, j in tp: d = minimum_image(P[j] - P[i], cell); ax.plot([P[i, 0], P[i, 0] + d[0]], [P[i, 1], P[i, 1] + d[1]], [P[i, 2], P[i, 2] + d[2]], color=style.GREY, lw=1.0)
            ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=(R * 14) ** 2, c=[style.OKABE_ITO['vermillion'] if p else 'white' for p in perc], edgecolors=style.OKABE_ITO['vermillion'], linewidths=1.0, alpha=0.9, depthshade=False)
        for a, b in [(i, j) for i in range(8) for j in range(8) if bin(i ^ j).count('1') == 1 and i < j]:                       # the cell edges
            p, q = np.array([(a >> k) & 1 for k in range(3)]) * L, np.array([(b >> k) & 1 for k in range(3)]) * L; ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], color=style.INK, lw=0.6)
        ax.set_xlim(0, L[0]); ax.set_ylim(0, L[1]); ax.set_zlim(0, L[2]); ax.set_box_aspect(tuple(L)); ax.view_init(elev=22, azim=-55); ax.set_axis_off(); ax.text2D(0.5, 0.97, title, transform=ax.transAxes, fontsize=9, ha='center', va='top')
        ax.text2D(0.0, 0.98, 'ab'[k], transform=ax.transAxes, fontsize=12, fontweight='bold', ha='left', va='top')
    fig.text(0.5, 0.02, f"{str(z['name'])}: surface at probe {probe:g} Å (clearance grid), pores of the Voronoi network at the same probe, percolating pores filled; cell {L[0]:.1f} Å", ha='center', fontsize=7.5)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.06, top=0.96, wspace=0.0); fig.text(0.995, 0.995, 'synthetic data', ha='right', va='top', fontsize=7, color=style.GREY)
    fig.savefig(OUT / 'fig09_void_cube.pdf'); fig.savefig(OUT / 'fig09_void_cube.png', dpi=300); fig.savefig(FIG / 'fig09_void_cube.png', dpi=200); plt.close(fig); print('wrote fig09_void_cube')

if __name__ == '__main__':
    main()
