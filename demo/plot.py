#!/usr/bin/env python3
"""Figures fig01 to fig12 from the tables in demo/out, written to demo/out and demo/figures."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'demo'))
import style
import json
from vfa.structure.cell import minimum_image
from vfa.corrections.schemes import transition_levels
from vfa.units import HARTREE_EV

OUT, FIG = ROOT / 'demo/out', ROOT / 'demo/figures'


def ecdf(ax, x, **kw):
    """Empirical cumulative distribution of the finite values as a step line."""
    x = np.sort(np.asarray(x)[np.isfinite(x)]); ax.step(x, np.arange(1, len(x) + 1) / len(x), where='post', **kw)


def fig01(z):
    """The clearance field of one cell in a slice, with the isoline at the probe radius."""
    cell, pos, sym, C, probe = z['cell'], z['positions'], z['symbols'], z['slice'], float(z['probe']); L = np.diag(cell); fig, a = plt.subplots(figsize=(4.2, 3.6), constrained_layout=True)
    im = a.imshow(C.T, origin='lower', extent=(0, L[0], 0, L[1]), cmap='cividis', vmin=-1.0, vmax=max(2.0, C.max())); a.contour(np.linspace(0, L[0], C.shape[0], endpoint=False) + L[0] / C.shape[0] / 2, np.linspace(0, L[1], C.shape[1], endpoint=False) + L[1] / C.shape[1] / 2, C.T, levels=[probe], colors='white', linewidths=1.0)
    dz = pos[:, 2] - float(z['slice_z']); slab = np.abs(dz - np.rint(dz / L[2]) * L[2]) < 1.0
    for s_, mk, fc in (('A', 'o', style.INK), ('B', 'o', 'white')): m = slab & (sym == s_); a.scatter(pos[m, 0], pos[m, 1], s=22 if s_ == 'A' else 12, marker=mk, facecolors=fc, edgecolors=style.INK, linewidths=0.7, zorder=3)
    cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.02); cb.set_label('clearance (Å)'); a.set_xlabel('x (Å)'); a.set_ylabel('y (Å)'); a.set_xlim(0, L[0]); a.set_ylim(0, L[1]); style.style_axes(a, minor_x=False, minor_y=False)
    a.set_title(f'slice at z = {float(z["slice_z"]):.1f} Å; isoline: probe {probe:g} Å', fontsize=8)
    fig.legend(handles=[plt.Line2D([], [], ls='', marker='o', mfc=fc, mec=style.INK, label=f'{s_} within 1 Å of the slice') for s_, fc in (('A', style.INK), ('B', 'white'))], loc='outside lower center', ncol=2, fontsize=7.5, frameon=False)
    style.save(fig, OUT, FIG, 'fig01_clearance')

def fig02(sw, name):
    """Accessible void fraction and void surface of the example cell against the probe radius."""
    fig, axs = plt.subplots(1, 2, figsize=(style.TWO_COLUMN, 3.0), constrained_layout=True); g = sw[sw.cell == name].sort_values('probe_A')
    for ax, col_name, lab, l in zip(axs, ('fraction', 'surface_m2_cm3'), ('accessible void fraction', 'void surface (m$^2$ cm$^{-3}$)'), 'ab'):
        ax.plot(g.probe_A, g[col_name], marker='o', color=style.INK); ax.set_xlabel('probe radius (Å)'); ax.set_ylabel(lab); style.style_axes(ax); style.panel_label(ax, l)
    axs[0].set_ylim(0, 1.0); style.save(fig, OUT, FIG, 'fig02_probe_sweep')

def fig04(at, name):
    """Per-atom free volume by species, and the cavity distance of three- and four-fold A, in the example cell."""
    at = at[at.cell == name]; fig, axs = plt.subplots(1, 2, figsize=(style.TWO_COLUMN, 3.0), constrained_layout=True); col = {'A': style.OKABE_ITO['blue'], 'B': style.OKABE_ITO['vermillion']}
    for sp in ('A', 'B'): ecdf(axs[0], at[at.species == sp].free_volume_A3, color=col[sp], label=sp)
    axs[0].set_xlabel('free volume at the atom, probe 0.8 Å (Å$^3$)'); axs[0].set_ylabel('cumulative fraction of atoms'); style.legend(axs[0], loc='lower right', fontsize=8)
    a = at[at.species == 'A']; groups = [a[a.coordination == z].cavity_distance_A.dropna() for z in (3, 4)]
    bp = axs[1].boxplot(groups, widths=0.5, patch_artist=True, medianprops=dict(color=style.INK), flierprops=dict(marker='o', markersize=3, markerfacecolor='none', markeredgecolor=style.GREY))
    for patch, c in zip(bp['boxes'], (style.OKABE_ITO['bluishgreen'], style.OKABE_ITO['purple'])): patch.set(facecolor=c, alpha=0.5, edgecolor=style.INK)
    axs[1].set_xticks([1, 2]); axs[1].set_xticklabels([f'three-fold A\nn = {len(groups[0])}', f'four-fold A\nn = {len(groups[1])}']); axs[1].set_ylabel('distance to the nearest pore (Å)')
    for ax, l in zip(axs, 'ab'): style.style_axes(ax, minor_x=ax is axs[0]); style.panel_label(ax, l)
    style.save(fig, OUT, FIG, 'fig04_sites')

def fig05(rdf, rho, comp):
    """The A-B pair correlation of the cells at the example's density with the measured bond cutoff, and the running A-B coordination that the cutoff reads."""
    fig, axs = plt.subplots(1, 2, figsize=(style.TWO_COLUMN, 3.0), constrained_layout=True); dens = list(np.round(rdf['densities'], 6))
    cut = float(rdf['cutoff']); r0 = rdf['r']; r = r0.reshape(-1, 4).mean(1); g = rdf['curves_AB'][dens.index(round(rho, 6))]; top = 2.0 * g.reshape(-1, 4).mean(1)[r > cut].max()
    axs[0].plot(r, g.reshape(-1, 4).mean(1), color=style.INK); axs[0].set_ylim(-0.04 * top, top)
    axs[0].axvline(cut, color=style.INK, ls=':', lw=1); axs[0].text(cut + 0.1, 0.74 * top, f'bond cutoff {cut:.3f} Å\nfirst peak off scale', fontsize=7.5, va='top'); axs[0].set_xlim(1.0, 6.0); axs[0].set_xlabel('r (Å)'); axs[0].set_ylabel('$g_{AB}(r)$')
    dr = r0[1] - r0[0]; axs[1].plot(r0, np.cumsum(g * 4 * np.pi * r0 ** 2 * dr) * rho * comp, color=style.INK)
    axs[1].axvline(cut, color=style.INK, ls=':', lw=1); axs[1].axhline(4, color=style.GREY, lw=0.8, zorder=0); axs[1].text(1.05, 4.15, 'four B', ha='left', va='bottom', fontsize=7.5, color=style.GREY); axs[1].set_xlim(1.0, 3.5); axs[1].set_ylim(0, 8); axs[1].set_xlabel('r (Å)'); axs[1].set_ylabel('B neighbours of an A atom within r')
    for ax, l in zip(axs, 'ab'): style.style_axes(ax); style.panel_label(ax, l)
    style.save(fig, OUT, FIG, 'fig05_structure')

def fig06(z):
    """One cell before and after trapping: participation against energy, the precursor, the trap by character, the onsets."""
    fig, axs = plt.subplots(1, 2, figsize=(style.TWO_COLUMN, 3.2), constrained_layout=True); cmap = {True: style.OKABE_ITO['blue'], False: style.OKABE_ITO['vermillion']}
    ref = float(z['E0'][int(z['n0']) - 1])
    for ax, tag, lab in ((axs[0], 0, 'neutral'), (axs[1], 1, 'charged')):
        E, f, a, n = z[f'E{tag}'] - ref, z[f'f{tag}'], z[f'a{tag}'], int(z[f'n{tag}']); occ = np.arange(E.size) < n; keep = (E > -8) & (E < 14)
        ax.scatter(E[keep], a[keep], c=[cmap[o] for o in occ[keep]], s=14 + 40 * f[keep], marker='o', edgecolors=style.INK, linewidths=0.4, alpha=0.85)
        ax.axhline(float(z['n_atoms']) / 5, color=style.GREY, ls=':', lw=1); ax.axhline(float(z['n_atoms']) / 3, color=style.GREY, ls='--', lw=1); ax.set_yscale('log'); ax.set_xlabel('energy from the neutral HOMO, aligned (eV)'); ax.set_ylabel('participation (atoms)'); ax.set_title(lab, fontsize=10)
        style.style_axes(ax, minor_y=False)
    E0, n0 = z['E0'] - ref, int(z['n0']); axs[0].annotate('precursor', (E0[n0], z['a0'][n0]), (E0[n0] - 5.0, z['a0'][n0] * 0.55), fontsize=8, arrowprops=dict(arrowstyle='-', lw=0.6)); axs[0].annotate('onset 1/5', (E0[int(z['onset5'])], z['a0'][int(z['onset5'])]), (E0[int(z['onset5'])] + 0.6, z['a0'][int(z['onset5'])] * 0.4), fontsize=8, arrowprops=dict(arrowstyle='-', lw=0.6))
    E1, t = z['E1'] - ref, int(z['trap']); axs[1].annotate(f'trap, A character {float(z["f1"][t]):.2f}', (E1[t], z['a1'][t]), (E1[t] - 5.5, z['a1'][t] * 2.5), fontsize=8, arrowprops=dict(arrowstyle='-', lw=0.6))
    axs[0].scatter([], [], c=cmap[True], s=20, edgecolors=style.INK, linewidths=0.4, label='occupied'); axs[0].scatter([], [], c=cmap[False], s=20, edgecolors=style.INK, linewidths=0.4, label='empty'); axs[0].plot([], [], color=style.GREY, ls=':', label='1/5 of the atoms'); axs[0].plot([], [], color=style.GREY, ls='--', label='1/3 of the atoms'); style.legend(axs[0], loc='upper left', bbox_to_anchor=(0.29, 0.97), fontsize=7.5)
    axs[1].text(0.03, 0.97, 'marker size: A character', transform=axs[1].transAxes, ha='left', va='top', fontsize=7.5); style.panel_label(axs[0], 'a'); style.panel_label(axs[1], 'b'); style.save(fig, OUT, FIG, 'fig06_spectrum_reading')


def fig07(c, curves, change, shares):
    """A value along the ordering column with its smoothed curve and straight-line change, and a share of sites per group over the cells only and with the sites counted."""
    fig, axs = plt.subplots(1, 2, figsize=(style.TWO_COLUMN, 3.0), constrained_layout=True); col, _ = style.series(['smoothed', 'theil_sen', 'least_squares']); q = 'trap_below_lumo_eV'; r = change[change.quantity == q].iloc[0]; x = np.array([r.x_low, r.x_high])
    axs[0].scatter(c.number_density, c[q], s=12, color=style.GREY, edgecolors=style.INK, linewidths=0.3, label='cells'); axs[0].plot(curves.number_density, curves[q], color=col['smoothed'], lw=1.6, label=f'smoothed, window {int(r.window_cells)} cells')
    for key, lab, ls in (('theil_sen', 'Theil–Sen', '-'), ('least_squares', 'least squares', '--')): axs[0].plot(x, [r[f'{key}_low'], r[f'{key}_high']], color=col[key], lw=1.2, ls=ls, label=f"{lab}: change {r[f'{key}_change']:+.2f} [{r[f'{key}_lower']:+.2f}, {r[f'{key}_upper']:+.2f}] eV")
    axs[0].set_ylim(c[q].min() - 0.9, c[q].max() + 0.25); axs[0].set_xlabel('number density (atoms Å$^{-3}$)'); axs[0].set_ylabel('trap below the lowest empty state (eV)'); style.legend(axs[0], loc='lower left', fontsize=7)
    dx = 0.0006; xs = shares.number_density.to_numpy(); s = shares.share.to_numpy()
    axs[1].errorbar(xs - dx, s, yerr=[s - shares.cells_lower, shares.cells_upper - s], fmt='o', ms=4, mfc='white', mec=style.GREY, color=style.GREY, capsize=2, label='interval from the cell values')
    axs[1].errorbar(xs + dx, s, yerr=[s - shares.sites_lower, shares.sites_upper - s], fmt='o', ms=4, color=style.INK, capsize=2, label='interval from the counted sites')
    axs[1].set_ylim(0, 1); axs[1].set_xlabel('group: number density (atoms Å$^{-3}$)'); axs[1].set_ylabel('share of three-fold A among A'); style.legend(axs[1], loc='upper right', fontsize=7.5)
    for ax, l in zip(axs, 'ab'): style.style_axes(ax); style.panel_label(ax, l)
    style.save(fig, OUT, FIG, 'fig07_statistics')

def fig08(m, conv):
    """The modes side by side: grid and pointwise, network and grid spheres, convergence checks."""
    fig, axs = plt.subplots(1, 3, figsize=(style.TWO_COLUMN, 2.9), constrained_layout=True); col, mk = style.series(['centre', 'covalent', 'ionic'])
    for r in ('centre', 'covalent', 'ionic'): axs[0].scatter(m[f'{r}_grid_fraction'], m[f'{r}_pointwise_fraction'], s=14, marker=mk[r], color=col[r], edgecolors=style.INK, linewidths=0.3, label=r)
    lim = (0, 1); axs[0].plot(lim, lim, color=style.GREY, lw=0.8, zorder=0); axs[0].set_xlim(lim); axs[0].set_ylim(lim); axs[0].set_xlabel('accessible fraction, grid'); axs[0].set_ylabel('accessible fraction, pointwise'); style.legend(axs[0], loc='upper left', fontsize=7.5)
    for r in ('centre', 'covalent', 'ionic'): axs[1].scatter(m[f'{r}_Di_A'], m[f'{r}_grid_largest_A'], s=14, marker=mk[r], color=col[r], edgecolors=style.INK, linewidths=0.3)
    lo, hi = min(m[[f'{r}_Di_A' for r in ('centre', 'covalent', 'ionic')]].min()), max(m[[f'{r}_Di_A' for r in ('centre', 'covalent', 'ionic')]].max()); axs[1].plot([lo, hi], [lo, hi], color=style.GREY, lw=0.8, zorder=0); axs[1].set_xlabel('$D_i$, network (Å)'); axs[1].set_ylabel('$D_i$, grid (Å)')
    p = conv[conv['mode'] == 'pointwise']; g = conv[conv['mode'] == 'grid']; V = float(pd.read_csv(ROOT / 'example/data/cells.csv').set_index('cell').volume_A3[conv.cell.iloc[0]])
    axs[2].plot(p.n, p.fraction, marker='o', color=style.INK, label='pointwise'); axs[2].plot(V / g.dx ** 3, g.fraction, marker='s', color=style.GREY, label='grid'); axs[2].set_xscale('log'); axs[2].set_xlabel('evaluation points'); axs[2].set_ylabel('accessible fraction')
    y = conv.fraction; axs[2].set_ylim(y.min() - 0.004, y.max() + 0.006); style.legend(axs[2], loc='lower right', fontsize=7)
    for ax, l in zip(axs, 'abc'): style.style_axes(ax, minor_x=ax is not axs[2]); style.panel_label(ax, l)
    style.save(fig, OUT, FIG, 'fig08_modes')


def fig09(z):
    """The void surface of one cell in three dimensions, with the atoms and with the pore network."""
    cell, pos, sym, C, probe, n = z['cell'], z['positions'], z['symbols'], z['field'].astype(float), float(z['probe']), z['shape']; L = np.diag(cell)
    verts, faces, _, _ = marching_cubes(C, level=probe); xyz = (verts / np.asarray(n, float)) @ cell
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
        for a, b in [(i, j) for i in range(8) for j in range(8) if bin(i ^ j).count('1') == 1 and i < j]:
            p, q = np.array([(a >> k) & 1 for k in range(3)]) * L, np.array([(b >> k) & 1 for k in range(3)]) * L; ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], color=style.INK, lw=0.6)
        ax.set_xlim(0, L[0]); ax.set_ylim(0, L[1]); ax.set_zlim(0, L[2]); ax.set_box_aspect(tuple(L)); ax.view_init(elev=22, azim=-55); ax.set_axis_off(); ax.text2D(0.5, 0.97, title, transform=ax.transAxes, fontsize=9, ha='center', va='top')
        ax.text2D(0.0, 0.98, 'ab'[k], transform=ax.transAxes, fontsize=12, fontweight='bold', ha='left', va='top')
    fig.text(0.5, 0.02, f"{str(z['name'])}: surface at probe {probe:g} Å (clearance grid), pores of the network at the same probe, percolating pores filled; cell {L[0]:.1f} Å", ha='center', fontsize=7.5)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.06, top=0.96, wspace=0.0); style.save(fig, OUT, FIG, 'fig09_void_cube', tight=False)


def fig10(c, at, ang):
    """Coordination by species at the measured cutoff, the bond-angle distributions and the tetrahedral order of A."""
    fig, axs = plt.subplots(1, 3, figsize=(style.TWO_COLUMN, 3.0), constrained_layout=True); col = {'A': style.OKABE_ITO['blue'], 'B': style.OKABE_ITO['vermillion']}; zs = np.arange(0, 7); w = 0.4
    for i, sp in enumerate(('A', 'B')): v = at[at.species == sp].coordination; axs[0].bar(zs + (i - 0.5) * w, [100 * np.mean(v == k) for k in zs], w, color=col[sp], edgecolor=style.INK, lw=0.4, label=sp)
    axs[0].set_xlabel('coordination at the cutoff'); axs[0].set_ylabel('share of atoms (%)'); style.legend(axs[0], loc='upper left', fontsize=8)
    cells = c.cell.tolist(); bab = np.concatenate([ang[f'{n}:BAB'] for n in cells]); aba = np.concatenate([ang[f'{n}:ABA'] for n in cells]); q = np.concatenate([ang[f'{n}:q'] for n in cells])
    axs[1].hist(bab, bins=np.linspace(60, 180, 61), density=True, histtype='step', color=style.OKABE_ITO['bluishgreen'], lw=1.3, label='B–A–B'); axs[1].hist(aba, bins=np.linspace(60, 180, 61), density=True, histtype='step', color=style.OKABE_ITO['orange'], lw=1.3, label='A–B–A')
    axs[1].axvline(np.degrees(np.arccos(-1 / 3)), color=style.GREY, ls=':', lw=1, label='tetrahedral angle'); axs[1].set_xlabel('bond angle (°)'); axs[1].set_ylabel('probability density (°$^{-1}$)'); axs[1].set_ylim(0, 1.5 * axs[1].get_ylim()[1]); style.legend(axs[1], loc='upper right', fontsize=8)
    axs[2].hist(q[np.isfinite(q)], bins=np.linspace(0.6, 1.0, 41), density=True, color=style.OKABE_ITO['blue'], alpha=0.6, edgecolor=style.INK, lw=0.4); axs[2].set_xlabel('tetrahedral order $q$ of A'); axs[2].set_ylabel('probability density')
    for ax, l in zip(axs, 'abc'): style.style_axes(ax, minor_x=ax is not axs[0]); style.panel_label(ax, l)
    style.save(fig, OUT, FIG, 'fig10_local_geometry')


def fig11(m):
    """Ring sizes under the three criteria, per cell, up to 24 atoms."""
    fig, ax = plt.subplots(figsize=(4.2, 3.0), constrained_layout=True); col, _ = style.series(['centre', 'covalent', 'ionic', 'primitive', 'king', 'guttman']); sizes = range(3, 13); w = 0.27
    for i, crit in enumerate(('primitive', 'king', 'guttman')): ax.bar(np.array(sizes) + (i - 1) * w, [m[f'{crit}_size{s}'].mean() for s in sizes], w, color=col[crit], edgecolor=style.INK, lw=0.4, label={'primitive': 'primitive', 'king': 'King', 'guttman': 'Guttman'}[crit])
    ax.set_ylim(0, 1.45 * max(m[f'{crit}_size{s}'].mean() for crit in ('primitive', 'king', 'guttman') for s in sizes)); ax.set_xlabel('ring size (A atoms)'); ax.set_ylabel('rings per cell'); ax.set_title('mean of 30 cells; rings up to 24 atoms', fontsize=8, loc='left'); ax.set_xticks(list(sizes)[::3]); style.legend(ax, loc='upper left', fontsize=8); style.style_axes(ax, minor_x=False)
    style.save(fig, OUT, FIG, 'fig11_rings')

def fig12(sc):
    """Total S(Q) of the cells made like the example cell by the reciprocal-lattice sum and by the transform, and G(r), equal weights."""
    fig, axs = plt.subplots(1, 2, figsize=(style.TWO_COLUMN, 3.0), constrained_layout=True)
    axs[0].plot(sc['q_transform'], sc['S_transform'], color=style.OKABE_ITO['blue'], lw=1.2, label='transform of g(r), Lorch'); axs[0].plot(sc['q_direct'], sc['S_direct'], 'o', ms=2.5, color=style.INK, label='reciprocal-lattice sum')
    axs[0].axhline(1, color=style.GREY, lw=0.8, zorder=0); axs[0].set_xlabel('Q (Å$^{-1}$)'); axs[0].set_ylabel('S(Q)'); axs[0].set_xlim(0, 20); style.legend(axs[0], loc='upper right', fontsize=8)
    axs[1].plot(sc['r'], sc['G'], color=style.INK, lw=1.0); axs[1].axhline(0, color=style.GREY, lw=0.8, zorder=0); axs[1].set_ylim(-1.5, 8); axs[1].text(0.97, 0.95, 'first peak off scale', transform=axs[1].transAxes, ha='right', va='top', fontsize=7.5); axs[1].set_xlabel('r (Å)'); axs[1].set_ylabel('G(r)')
    for ax, l in zip(axs, 'ab'): style.style_axes(ax); style.panel_label(ax, l)
    style.save(fig, OUT, FIG, 'fig12_total_scattering')


def fig03():
    """Potential alignment and transition levels of the oxygen vacancy in MgO, from the cp2k-charge-corrections example."""
    m = json.loads((ROOT / 'demo/data/mgo_vacancy.json').read_text()); al = m['alignment']; x = np.array(al['z_A']); half, start = 0.5 * (x.max() - x.min()), al['far_from_A']; colour = {0: style.INK, 1: style.OKABE_ITO['blue'], 2: style.OKABE_ITO['vermillion']}
    fig, axs = plt.subplots(1, 2, figsize=(style.TWO_COLUMN, 3.2), constrained_layout=True)
    for lo, hi in ((-half, -start), (start, half)): axs[0].axvspan(lo, hi, color=style.GREY, alpha=0.15, lw=0)
    for q in (1, 2):
        t = al['terms'][str(q)]; axs[0].plot(x, al['potential_V'][str(q)], color=colour[q], label=f'+{q}: {t["scale"] * t["point_charge_eV"]:.3f} - {q} x {t["alignment_V"]:.3f} = {t["scale"] * t["point_charge_eV"] - q * t["alignment_V"]:.3f} eV')
        for lo, hi in ((-half, -start), (start, half)): axs[0].plot([lo, hi], [t['alignment_V']] * 2, color=colour[q], ls=(0, (3, 2)), lw=1.0)
    axs[0].axhline(0, color=style.GREY, lw=0.6, ls=':'); axs[0].set_xlim(-half, half); axs[0].set_ylim(-0.2, 0.2); axs[0].set_xlabel('distance from the vacancy (Å)'); axs[0].set_ylabel('charged defect - host (V)')
    style.legend(axs[0], loc='upper center', fontsize=7.5, title='LZ = s E$_{PC}$ - q ΔV; dashed: ΔV', title_fontsize=7.5)
    e0 = next(st['energy_Ha'] for st in m['states'] if st['charge'] == 0); gap = m['host_gap_eV']; fermi = np.array([0.0, gap])
    for corrected, ls, face in ((False, (0, (3, 2)), 'white'), (True, '-', style.INK)):
        f = {st['charge']: (st['energy_Ha'] - e0) * HARTREE_EV + st['charge'] * m['host_vbm_eV'] + corrected * st['correction_eV'] for st in m['states']}
        for q, e in f.items(): axs[1].plot(fermi, e + q * fermi, color=colour[q], ls=ls, lw=1.3 if corrected else 0.9, label=(f'+{q}' if q else '0') if corrected else None)
        for t in transition_levels(f):
            if 0 <= t['level_eV'] <= gap: axs[1].plot([t['level_eV']], [f[t['charges'][1]] + t['charges'][1] * t['level_eV']], 'o', ms=5, mfc=face, mec=style.INK, zorder=3)
    axs[1].plot([], [], color=style.GREY, ls=(0, (3, 2)), lw=0.9, label='uncorrected'); axs[1].plot([], [], color=style.GREY, lw=1.3, label='LZ corrected')
    axs[1].set_xlim(0, gap); axs[1].set_xlabel('Fermi level above the valence band (eV)'); axs[1].set_ylabel('formation energy, neutral = 0 (eV)'); style.legend(axs[1], loc='upper left', fontsize=7.5)
    for ax, l in zip(axs, 'ab'): style.style_axes(ax); style.panel_label(ax, l)
    style.save(fig, OUT, FIG, 'fig03_corrections', stamp='MgO V_O, cp2k-charge-corrections example')

def main():
    """Every figure from the tables in demo/out."""
    style.apply(); FIG.mkdir(exist_ok=True); c = pd.read_csv(OUT / 'cells.csv'); sw = pd.read_csv(OUT / 'sweep.csv'); at = pd.read_csv(OUT / 'atoms.csv')
    z = np.load(OUT / 'example_cell.npz'); name = str(z['name']); fig01(z); fig02(sw, name); fig04(at, name); fig05(np.load(OUT / 'rdf.npz'), float(c.set_index('cell').number_density[name]), float(np.mean(z['symbols'] == 'B'))); fig10(c, at, np.load(OUT / 'angles.npz')); fig11(pd.read_csv(OUT / 'modes.csv')); fig12(np.load(OUT / 'scattering.npz'))
    fig06(np.load(OUT / 'spectrum_example.npz')); ev = pd.read_csv(OUT / 'events.csv'); fig07(c.merge(ev.drop(columns='number_density'), on='cell'), pd.read_csv(OUT / 'trend_curves.csv'), pd.read_csv(OUT / 'trend_change.csv'), pd.read_csv(OUT / 'shares.csv')); fig08(pd.read_csv(OUT / 'modes.csv'), pd.read_csv(OUT / 'convergence.csv')); fig09(np.load(OUT / 'example_cell.npz')); fig03()


if __name__ == '__main__':
    main()
