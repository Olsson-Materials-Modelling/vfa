#!/usr/bin/env python3
"""Trends with density by smoothing in order, straight-line changes, contrasts between the density halves and a held-out model of the trap kind (demo/out/trend_*.csv, contrasts.csv, model.csv, learning_curve.csv)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from vfa.stats.trends import smooth_in_order, held_out_window, positions, linear_change
from vfa.stats.contrasts import cliff_delta_interval, scale_ratio
from vfa.stats.intervals import percentile_bootstrap, wilson
from vfa.stats.models import held_out_shares, learning_curve, softmax_fit, softmax_predict, standardise, log_loss

OUT = ROOT / 'demo/out'; QUANTITIES = ['fraction_p0.8', 'Di_A', 'rings_per_1000A3', 'trap_below_edge_eV', 'relaxation_corrected_eV']


def main():
    c = pd.read_csv(OUT / 'cells.csv').merge(pd.read_csv(OUT / 'events.csv').drop(columns='number_density'), on='cell').sort_values(['number_density', 'cell'], kind='stable').reset_index(drop=True); rho = c.number_density.to_numpy(); q, at = positions(rho)      # cells of equal density ordered by name: the order is part of the result
    Y = c[QUANTITIES].to_numpy(); w = held_out_window(Y, (2, 4, 6, 10, 15)); rows = []
    for k, name in enumerate(QUANTITIES):
        ends = smooth_in_order(Y[:, [k]], at, w['chosen'])[:, 0]; ch = linear_change(rho, Y[:, k], q[0], q[1], n_replicates=1000)
        rows.append(dict(quantity=name, window_cells=w['chosen'], low=ends[0], high=ends[1], change_smoothed=ends[1] - ends[0], **{f'theil_sen_{a}': ch['theil_sen'][a] for a in ('low', 'high', 'change', 'lower', 'upper')}, **{f'least_squares_{a}': ch['least_squares'][a] for a in ('low', 'high', 'change', 'lower', 'upper')}, density_low=q[0], density_high=q[1], spearman=ch['spearman'], p=ch['p']))
    t = pd.DataFrame(rows); t.to_csv(OUT / 'trend_change.csv', index=False); curve = pd.DataFrame(smooth_in_order(Y, np.arange(len(c)), w['chosen']), columns=QUANTITIES).assign(number_density=rho); curve.to_csv(OUT / 'trend_curves.csv', index=False)
    lo, hi = c[rho <= np.median(rho)], c[rho > np.median(rho)]; con = []
    for name in QUANTITIES:
        d = cliff_delta_interval(hi[name].to_numpy(), lo[name].to_numpy(), n_replicates=1000); s = scale_ratio(hi[name].to_numpy(), lo[name].to_numpy(), n_replicates=1000); con.append(dict(quantity=name, cliff_delta=d['delta'], lower=d['lower'], upper=d['upper'], p=d['p'], sd_ratio=s['ratio'], sd_lower=s['lower'], sd_upper=s['upper'], brown_forsythe_p=s['brown_forsythe_p']))
    pd.DataFrame(con).to_csv(OUT / 'contrasts.csv', index=False)
    X = c[['free_volume_trap_A3', 'Di_A']].to_numpy(); y = c.kind.to_numpy(); classes = sorted(set(y)); train = np.arange(len(c)) % 2 == 0                     # every other cell in density order: the two halves span the same densities
    sh = held_out_shares(X[train], y[train], X[~train], classes, n_replicates=400); obs = [(y[~train] == k).mean() for k in classes]; P = softmax_predict(sh['weights'], standardise(X[train], X[~train]))
    model = pd.DataFrame(dict(kind=classes, observed=obs, predicted=sh['predicted'], lower=sh['lower'], upper=sh['upper'], wilson_lower=[wilson(int(o * (~train).sum()), int((~train).sum()))[0] for o in obs], wilson_upper=[wilson(int(o * (~train).sum()), int((~train).sum()))[1] for o in obs])); model.to_csv(OUT / 'model.csv', index=False)
    lc = pd.DataFrame(learning_curve(X[train], y[train], X[~train], classes, (6, 9, 12, 15), n_replicates=100)); lc.to_csv(OUT / 'learning_curve.csv', index=False)
    pd.set_option('display.width', 220); print(f"window {w['chosen']} cells (losses {dict(zip(w['widths'], np.round(w['losses'], 4)))})"); print(t.round(3).to_string(index=False)); print(pd.DataFrame(con).round(3).to_string(index=False)); print(model.round(3).to_string(index=False)); print('held-out log loss', round(log_loss(P, y[~train], classes), 3))


if __name__ == '__main__':
    main()
