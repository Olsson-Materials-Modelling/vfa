#!/usr/bin/env python3
"""Statistics on the example cells: a trend along an ordering column, group means pointwise and simultaneous, a share of sites, with provenance sidecars (demo/out)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from vfa.stats.trends import smooth_in_order, held_out_window, positions, linear_change
from vfa.stats.intervals import percentile_bootstrap, bounded_prevalence, influence_mean, maxt_simultaneous
from vfa.provenance import RunContext, write_csv

OUT = ROOT / 'demo/out'; ORDER = 'number_density'; GROUP = 'number_density'; VALUE = 'trap_below_lumo_eV'
QUANTITIES = ['fraction_p0.8', 'Di_A', 'rings_per_1000A3', 'trap_below_lumo_eV', 'addition_corrected_eV']


def trends(c, ctx):
    """Smoothed curves, each at its own held-out window, and straight-line changes of every quantity along the ordering column."""
    x = c[ORDER].to_numpy(); Y = c[QUANTITIES].to_numpy(); q, _ = positions(x); w = held_out_window(Y, tuple(int(v) for v in np.unique(np.round(np.geomspace(1, 2 * len(c), 10))))); rows = []
    for k, name in enumerate(QUANTITIES):
        ch = linear_change(x, Y[:, k], q[0], q[1], n_replicates=1000)
        rows.append(dict(quantity=name, window_cells=w['column_chosen'][k], x_low=q[0], x_high=q[1], **{f'{m}_{a}': ch[m][a] for m in ('theil_sen', 'least_squares') for a in ('low', 'high', 'change', 'lower', 'upper')}, spearman=ch['spearman'], p=ch['p']))
    xs = np.unique(x); at = np.array([np.flatnonzero(x == v).mean() for v in xs]); curve = pd.DataFrame({name: smooth_in_order(Y[:, k], at, w['column_chosen'][k])[:, 0] for k, name in enumerate(QUANTITIES)}).assign(**{ORDER: xs})
    write_csv(pd.DataFrame(rows), OUT / 'trend_change.csv', ctx, float_format=None); write_csv(curve, OUT / 'trend_curves.csv', ctx, float_format=None)
    return w


def groups(c, ctx):
    """The mean of one value in every group, with pointwise and simultaneous intervals."""
    y, g = c[VALUE].to_numpy(), c[GROUP].to_numpy(); labels = np.unique(g); est = np.array([y[g == u].mean() for u in labels]); inf = np.zeros((len(y), len(labels)))
    for j, u in enumerate(labels): inf[g == u, j] = influence_mean(y[g == u])[1]
    band = maxt_simultaneous(inf, est, groups=g); point = [percentile_bootstrap(y[g == u]) for u in labels]
    t = pd.DataFrame({'quantity': VALUE, GROUP: labels, 'n_cells': [int((g == u).sum()) for u in labels], 'mean': est, 'lower': [p['lower'] for p in point], 'upper': [p['upper'] for p in point],
                      'simultaneous_lower': band['lower'], 'simultaneous_upper': band['upper'], 'critical_value': band['critical_value']})
    write_csv(t, OUT / 'groups.csv', ctx, float_format=None); return t


def shares(c, at, ctx):
    """The share of three-fold A among A in every group, over the cells only and with the sites counted."""
    a = at[at.species == 'A']; per = a.groupby('cell').agg(k=('coordination', lambda z: int((z == 3).sum())), m=('coordination', 'size')).reindex(c.cell); g = c[GROUP].to_numpy(); rows = []
    for u in np.unique(g):
        k, m = per.k.to_numpy()[g == u], per.m.to_numpy()[g == u]; cells, sites = bounded_prevalence(k / m), bounded_prevalence(k / m, sites=m)
        rows.append({GROUP: u, 'n_cells': len(k), 'n_sites': int(m.sum()), 'share': sites['estimate'], 'cells_lower': cells['lower'], 'cells_upper': cells['upper'], 'sites_lower': sites['lower'], 'sites_upper': sites['upper'], 'n_effective': sites['n_effective'], 'kind': sites['kind']})
    t = pd.DataFrame(rows); write_csv(t, OUT / 'shares.csv', ctx, float_format=None); return t


def main():
    """Trends, group means and shares of sites, written with sidecars."""
    ctx = RunContext().with_config(dict(order=ORDER, group=GROUP, value=VALUE, quantities=QUANTITIES)).with_inputs([OUT / 'cells.csv', OUT / 'events.csv', OUT / 'atoms.csv'])
    c = pd.read_csv(OUT / 'cells.csv').merge(pd.read_csv(OUT / 'events.csv').drop(columns='number_density'), on='cell').sort_values([ORDER, 'cell'], kind='stable').reset_index(drop=True)
    w = trends(c, ctx); gr = groups(c, ctx); sh = shares(c, pd.read_csv(OUT / 'atoms.csv'), ctx)
    pd.set_option('display.width', 220); print(f"smoothing windows (cells) {dict(zip(QUANTITIES, w['column_chosen']))}"); print(gr.round(3).to_string(index=False)); print(sh.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
