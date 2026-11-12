"""Tangent spheres of four balls and the trisector curve of three against their defining equations and an independent numerical solve."""
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vfa.voids.spheres import Trisector, tangent_spheres


def _quadruples(seed, n):
    rng = np.random.default_rng(seed); return rng.normal(0, 2, (n, 4, 3)), rng.uniform(0, 1.2, (n, 4))


def _residual(X, R, C, r):
    return np.abs(np.linalg.norm(X[:, :, None] - C[:, None], axis=3) - (R[:, :, None] + r[:, None]))


def _numerical(C, r, rng, starts=20):
    def f(z): return np.linalg.norm(z[:3] - C, axis=1) - z[3] - r
    sols = []
    for _ in range(starts):
        z = least_squares(f, np.r_[C.mean(0) + rng.normal(0, 4, 3), abs(rng.normal(0, 3))], xtol=1e-14, ftol=1e-14, gtol=1e-14).x
        if np.abs(f(z)).max() < 1e-8 and z[3] + r.min() >= -1e-9 and not any(np.linalg.norm(z - q) < 1e-5 for q in sols): sols.append(z)
    return sols


def test_tangent_spheres_touch_all_four_balls():
    C, r = _quadruples(3, 400); X, R = tangent_spheres(C, r); ok = np.isfinite(R)
    assert ok.sum() > 300 and _residual(X, R, C, r)[ok].max() < 1e-9 and (R[ok] + r.min(1).repeat(2).reshape(-1, 2)[ok] >= -1e-9).all() and np.isnan(X[~ok]).all()


def test_tangent_spheres_match_an_independent_solve():
    C, r = _quadruples(3, 400); X, R = tangent_spheres(C, r); n = np.isfinite(R).sum(1); rng = np.random.default_rng(7); counts = []
    for k in np.r_[np.flatnonzero(n == 0)[:4], np.flatnonzero(n == 1)[:4], np.flatnonzero(n == 2)[:4]]:
        got = [np.r_[X[k, j], R[k, j]] for j in range(2) if np.isfinite(R[k, j])]; sols = _numerical(C[k], r[k], rng); counts.append(len(got))
        assert all(any(np.linalg.norm(s - g) < 1e-5 for g in got) for s in sols) and all(any(np.linalg.norm(s - g) < 1e-5 for s in sols) for g in got)
    assert set(counts) == {0, 1, 2}


def test_coplanar_and_collinear_quadruples():
    C, r = _quadruples(3, 400); C[:, :, 2] = 0.0; X, R = tangent_spheres(C, r); ok = np.isfinite(R)
    assert ok.sum() > 200 and _residual(X, R, C, r)[ok].max() < 1e-9 and np.allclose(X[ok[:, 0] & ok[:, 1]][:, 0, :2], X[ok[:, 0] & ok[:, 1]][:, 1, :2])
    th = np.random.default_rng(1).uniform(0, 2 * np.pi, (50, 4)); Cc = np.stack([2 * np.cos(th), 2 * np.sin(th), np.zeros_like(th)], 2); Xc, Rc = tangent_spheres(Cc, np.full((50, 4), 0.5))
    assert np.allclose(Rc[:, 0], 1.5) and np.allclose(Xc[:, 0], 0.0) and np.isnan(Rc[:, 1]).all()
    t = np.random.default_rng(2).normal(0, 2, (50, 4, 1)); Cl = t * np.array([1.0, 0.3, -0.2]); Xl, Rl = tangent_spheres(Cl, np.random.default_rng(2).uniform(0, 1.2, (50, 4)))
    assert np.isnan(Rl).all() and np.isnan(Xl).all()


def test_trisector_points_and_contacts():
    C, r = _quadruples(5, 300); ts = Trisector(C[:, 0], C[:, 1], C[:, 2], r[:, 0], r[:, 1], r[:, 2]); X, R = tangent_spheres(C, r); i = np.arange(len(C))
    Rc, hc = ts.contacts(i, C[:, 3], r[:, 3], 1e-9); live = np.flatnonzero(ts.kind >= 0); found = 0
    for k in live:
        sol = [X[k, j] for j in range(2) if np.isfinite(R[k, j])]; m = np.flatnonzero(np.isfinite(Rc[k]) & (Rc[k] + r[k].min() >= -1e-9)); kk = np.full(len(m), k)
        x = ts.xyz(Rc[k, m], hc[k, m], kk); t, s = ts.param(Rc[k, m], hc[k, m], kk); R2, h2 = ts.point(t, s, kk); found += len(sol)
        assert np.abs(np.linalg.norm(x[:, None] - C[k][None], axis=2) - (Rc[k, m][:, None] + r[k][None])).max(initial=0) < 1e-8 and np.allclose(ts.xyz(R2, h2, kk), x, atol=1e-8)
        assert all(np.linalg.norm(x - y, axis=1).min(initial=np.inf) < 1e-6 for y in sol) and all(any(np.linalg.norm(p - y) < 1e-6 for y in sol) for p in x)
    assert len(live) > 250 and found > 200
