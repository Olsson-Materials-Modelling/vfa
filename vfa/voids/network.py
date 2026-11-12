"""The pore network of the clearance: additively weighted Voronoi nodes and edges, included and free spheres, percolation, pores and throats."""
import itertools
from math import comb
import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree
from ..structure.cell import SHIFTS, _valid, folded, images, volume, widths
from .clearance import _centres, _radii, clearance_field, exact_clearance, threads
from .spheres import Trisector, tangent_spheres


class _Offsets:
    """Union-find with lattice offsets to the root; union returns True when it closes a loop with a net translation."""

    def __init__(self, n):
        self.p = list(range(n)); self.size = [1] * n; self.off = [(0, 0, 0)] * n

    def root(self, a):
        path = []
        while self.p[a] != a: path.append(a); a = self.p[a]
        acc = (0, 0, 0)
        for b in reversed(path): o = self.off[b]; acc = (acc[0] + o[0], acc[1] + o[1], acc[2] + o[2]); self.p[b] = a; self.off[b] = acc
        return a, (self.off[path[0]] if path else (0, 0, 0))

    def union(self, u, v, duv):
        (ru, ou), (rv, ov) = self.root(u), self.root(v); d = (int(duv[0]), int(duv[1]), int(duv[2]))
        if ru == rv: return (ov[0] - ou[0], ov[1] - ou[1], ov[2] - ou[2]) != d
        if self.size[ru] < self.size[rv]: self.p[ru] = rv; self.off[ru] = (ov[0] - d[0] - ou[0], ov[1] - d[1] - ou[1], ov[2] - d[2] - ou[2]); self.size[rv] += self.size[ru]
        else: self.p[rv] = ru; self.off[rv] = (ou[0] + d[0] - ov[0], ou[1] + d[1] - ov[1], ou[2] + d[2] - ov[2]); self.size[ru] += self.size[rv]
        return False


class _Space:
    """The atom balls and their images within reach, with the exact clearance and the largest reach any query needed."""

    def __init__(self, pos, cell, rad0, reach, k, max_images, max_pairs, workers):
        N = len(pos); self.cell = cell; self.inv = np.linalg.inv(cell); self.w = widths(cell); self.reach = reach; self.k = k; self.max_pairs = max_pairs; self.workers = workers; rep, rad = images(pos, cell, rad0, reach)
        fr = rep @ self.inv; sel = np.flatnonzero(np.all((fr >= -reach / self.w) & (fr <= 1 + reach / self.w), axis=1))
        if len(sel) > max_images: raise RuntimeError(f'voronoi_network: {len(sel)} atom images within the reach {reach:.1f} A exceed max_images ({max_images})')
        self.P, self.r, self.atom = rep[sel], rad[sel], sel % N; self.tree = cKDTree(self.P); self.rmax = float(self.r.max()); self.need = 0.0; self.N = N

    def outside(self, x):
        f = x @ self.inv; return (np.maximum(np.maximum(-f, f - 1), 0) * self.w).max(1)

    def clear(self, x):
        if not len(x): return np.zeros(0)
        c = exact_clearance(self.tree, self.r, x, self.k, workers=self.workers); self.need = max(self.need, float((self.outside(x) + np.maximum(c, 0)).max()) + self.rmax); return c

    def blocks(self, x, rho):
        if not len(x): return []
        n = self.tree.query_ball_point(x, rho, return_length=True, workers=self.workers); c = np.cumsum(n); out = []; s = 0
        if n.max() > self.max_pairs: raise RuntimeError(f'voronoi_network: {int(n.max())} atom images within {float(rho[np.argmax(n)]):.1f} A of one point exceed max_pairs ({self.max_pairs})')
        while s < len(n): e = int(np.searchsorted(c, (c[s - 1] if s else 0) + self.max_pairs, 'right')); out.append(slice(s, e)); s = e
        return out

    def balls(self, x, rho):
        lists = self.tree.query_ball_point(x, rho, return_sorted=True, workers=self.workers); own = np.repeat(np.arange(len(x)), [len(b) for b in lists]); return own, np.fromiter(itertools.chain.from_iterable(lists), np.int64, len(own))

    def canonical(self, x):
        n = np.floor(x @ self.inv).astype(np.int64); return x - n @ self.cell, n


def _largest_clearance(pos, cell, rad0, workers=-1, points=1 << 22):
    C, n = clearance_field(pos, cell, rad0, max(0.5, float(np.prod(np.linalg.norm(cell, axis=1)) / points) ** (1 / 3)), workers=workers)
    return float(C.max()) + _half_diagonal(cell / np.asarray(n, float)[:, None])


def _half_diagonal(E):
    return 0.5 * float(np.linalg.norm(np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1]]) @ E, axis=1).max())


def _seeds(sp, tol):
    y = sp.P - sp.cell.sum(0) / 2; sc = float(np.abs(y).max()); lift = ((y * y).sum(1) - sp.r ** 2) / sc ** 2 + 1e-9 * np.random.default_rng(0).random(sp.N)[sp.atom]
    try: hull = ConvexHull(np.c_[y / sc, lift])
    except QhullError as e: raise RuntimeError(f'voronoi_network: the regular triangulation of {len(sp.P)} atom images within the reach {sp.reach:.1f} A failed: {e}'.split('\n')[0]) from e
    T = hull.simplices[hull.equations[:, 3] < 0]; T = T[sp.outside(sp.P[T].mean(1)) <= 0]; out = []
    for _ in range(3):
        X, R = tangent_spheres(sp.P[T], sp.r[T]); fin = np.isfinite(R); fin[fin] = sp.outside(X[fin]) + R[fin] + sp.rmax <= sp.reach; fX = np.full(R.shape, -np.inf); fX[fin] = sp.clear(X[fin])
        good = fin & (fX >= R - tol); out.append(X[good]); bad = np.flatnonzero(~good.any(1) & fin.any(1))
        if not bad.size: break
        xb = X[bad, np.argmax(np.where(fin[bad], fX[bad], -np.inf), 1)]; _, e = sp.tree.query(xb, k=8, workers=sp.workers); d = np.linalg.norm(sp.P[e] - xb[:, None], axis=2) - sp.r[e]
        d[np.any(e[:, :, None] == T[bad][:, None, :], axis=2)] = np.inf; e = e[np.arange(len(bad)), np.argmin(d, 1)]; T = np.concatenate([np.c_[np.delete(T[bad], a, axis=1), e] for a in range(4)])
    return np.concatenate(out)


def _facets(sp, x, R, tolK):
    rho = np.maximum(R, 0) + sp.rmax + tolK; sp.need = max(sp.need, float((sp.outside(x) + rho).max())); parts = []
    for s in sp.blocks(x, rho):
        own, b = sp.balls(x[s], rho[s]); own = own + s.start; t = np.linalg.norm(sp.P[b] - x[own], axis=1) - sp.r[b] - R[own] <= tolK; parts.append((own[t], b[t]))
    own, b = (np.concatenate(q) for q in zip(*parts)); cnt = np.bincount(own, minlength=len(x)); start = np.r_[0, np.cumsum(cnt)]
    four = np.flatnonzero(cnt == 4); K4 = b[start[four][:, None] + np.arange(4)]; F4 = np.array([(0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 3, 1), (1, 2, 3, 0)])
    u = sp.P[K4] - x[four][:, None]; u = u / np.linalg.norm(u, axis=2)[:, :, None]; flat = np.abs(np.linalg.det(u[:, 1:] - u[:, :1])) < 1e-9; bad = cnt < 4; bad[four[flat]] = True; four, K4 = four[~flat], K4[~flat]
    tri = [K4[:, F4[:, :3]].reshape(-1, 3)]; oth = [K4[:, F4[:, 3]].reshape(-1, 1)]; src = [np.repeat(four, 4)]; m = 1
    for v in np.flatnonzero(cnt > 4):
        K = b[start[v]:start[v + 1]]; u = sp.P[K] - x[v]; u = u / np.linalg.norm(u, axis=1)[:, None]
        try: F = np.sort(ConvexHull(u).simplices, axis=1) if np.linalg.svd(u - u.mean(0), compute_uv=False)[2] >= 1e-9 else None
        except QhullError: F = None
        if F is None: bad[v] = True; continue
        O = np.array([np.setdiff1d(K, K[f]) for f in F]); tri.append(K[F]); oth.append(O); src.append(np.full(len(F), v)); m = max(m, O.shape[1])
    O = np.concatenate([np.pad(o, ((0, 0), (0, m - o.shape[1])), constant_values=-1) for o in oth])
    return np.concatenate(tri), O, np.concatenate(src), bad


def _extremes(ts, i, t0, d, sig, s, R0, R1):
    k = ts.kind[i]; t1 = t0 + sig * d
    with np.errstate(invalid='ignore'):
        c0 = np.where(k == 0, t0 * t1 < 0, (k == 2) & (((sig * (0 - t0)) % (2 * np.pi)) < d)); c1 = (k == 2) & (((sig * (np.pi - t0)) % (2 * np.pi)) < d)
    Rz, _ = ts.point(np.zeros(len(i)), s, i); Rpi = ts.Rm[i] - ts.a[i]; Rz = np.where(c0, Rz, np.nan); Rpi = np.where(c1, Rpi, np.nan)
    hi = np.nanmax(np.c_[Rz, Rpi, R0, R1], axis=1); lo = np.nanmin(np.c_[Rz, Rpi, R0, R1], axis=1)
    tlo = np.where(lo == Rz, 0.0, np.where(lo == Rpi, np.pi, np.nan)); tlo[(lo == R0) | (lo == R1)] = np.nan; return lo, hi, tlo


def _trace(sp, xv, Rv, tri, O, tol, bound):
    E = len(tri); ts = Trisector(sp.P[tri[:, 0]], sp.P[tri[:, 1]], sp.P[tri[:, 2]], sp.r[tri[:, 0]], sp.r[tri[:, 1]], sp.r[tri[:, 2]]); hv = ((xv - ts.c) * ts.n).sum(1); tv, sv = ts.param(Rv, hv)
    step = np.where(ts.kind == 2, 1e-6, 1e-6 * (1 + np.abs(tv))); oc = np.where(O >= 0, O, 0); score = []
    for sg in (1.0, -1.0):
        R, h = ts.point(tv + sg * step, sv); x = ts.xyz(R, h); score.append(np.where(O >= 0, np.linalg.norm(sp.P[oc] - x[:, None], axis=2) - sp.r[oc] - R[:, None], np.inf).min(1))
    sig = np.where(score[0] >= score[1], 1.0, -1.0); avail = np.maximum(sp.reach - sp.outside(ts.c), 0); rho0 = 2 * (np.maximum(Rv, 0) + 0.5) + ts.ri + sp.rmax; cap = 2 * bound + ts.ri + sp.rmax + 1e-9
    rho = np.minimum(np.minimum(rho0, avail), cap); dist = np.full(E, np.nan); Rw = np.full(E, np.nan); hw = np.full(E, np.nan); bw = np.full(E, -1, np.int64); queue = np.flatnonzero((ts.kind >= 0) & (np.maximum(score[0], score[1]) > -tol))

    def advance(e):
        own, b = sp.balls(ts.c[e], rho[e]); i = e[own]; keep = np.all(b[:, None] != tri[i], axis=1); own, b, i = own[keep], b[keep], i[keep]
        Rc, hc = ts.contacts(i, sp.P[b], sp.r[b], tol); tc, sc = ts.param(Rc.ravel(), hc.ravel(), np.repeat(i, 4)); tc, sc = tc.reshape(-1, 4), sc.reshape(-1, 4); k = ts.kind[i][:, None]
        with np.errstate(invalid='ignore'):
            d = sig[i][:, None] * (tc - tv[i][:, None]); d = np.where(k == 2, d % (2 * np.pi), d); d = np.where(np.isfinite(Rc) & (d > np.where(k == 2, 1e-9, 1e-9 * (1 + np.abs(tv[i]))[:, None])) & ((k == 2) | (sc == sv[i][:, None])), d, np.inf)
        j = np.argmin(d, 1); dm = d[np.arange(len(d)), j]; o = np.lexsort((dm, own)); o = o[np.r_[True, own[o][1:] != own[o][:-1]]] if o.size else o; pick = -np.ones(e.size, np.int64); pick[own[o]] = o
        found = (pick >= 0); found[found] = np.isfinite(dm[pick[found]]); pk = np.maximum(pick, 0); loop = ~found & (ts.kind[e] == 2)
        Rn = np.where(found, Rc[pk, j[pk]] if len(j) else np.nan, np.nan); hn = np.where(found, hc[pk, j[pk]] if len(j) else np.nan, np.nan); bn = np.where(found, b[pk] if len(b) else -1, -1); dd = np.where(found, dm[pk] if len(dm) else np.inf, 2 * np.pi)
        _, hi, _ = _extremes(ts, e, tv[e], dd, sig[e], sv[e], Rv[e], np.where(found, Rn, Rv[e])); req = 2 * hi + ts.ri[e] + sp.rmax + 1e-9
        fine = (found | loop) & (req <= rho[e]); dist[e[fine]] = np.where(loop[fine], np.inf, dd[fine]); Rw[e[fine]] = Rn[fine]; hw[e[fine]] = hn[fine]; bw[e[fine]] = bn[fine]; live = ~fine & (rho[e] < cap[e])
        short = live & (rho[e] >= avail[e] - 1e-12); sp.need = max([sp.need] + list(sp.outside(ts.c[e[short]]) + np.minimum(np.maximum(np.where(found[short], req[short], 2 * rho[e[short]]), rho0[e[short]]), cap[e[short]])))
        grow = live & ~short; rho[e[grow]] = np.minimum(np.minimum(np.where(found[grow], 1.25 * req[grow], 2 * rho[e[grow]]), avail[e[grow]]), cap[e[grow]]); return e[grow]
    while queue.size: queue = np.concatenate([advance(queue[s]) for s in sp.blocks(ts.c[queue], rho[queue])])
    done = np.isfinite(dist); idx = np.flatnonzero(done); xw, R1 = _snap(sp, tri[idx], bw[idx], ts.xyz(Rw[idx], hw[idx], idx), Rw[idx], 1e3 * tol); ok = sp.clear(xw) >= R1 - tol
    _, _, tlo = _extremes(ts, idx, tv[idx], dist[idx], sig[idx], sv[idx], Rv[idx], Rw[idx]); Rp, hp = ts.point(tv[idx] + sig[idx] * dist[idx] / 2, sv[idx], idx); mid = ts.xyz(Rp, hp, idx); ok &= sp.clear(mid) >= Rp - tol
    neck = np.minimum(Rv[idx], R1); m = np.isfinite(tlo)
    if m.any(): Rq, hq = ts.point(tlo[m], sv[idx][m], idx[m]); neck[m] = np.minimum(neck[m], sp.clear(ts.xyz(Rq, hq, idx[m])))
    loops = int(np.isinf(dist).sum()); return idx[ok], xw[ok], neck[ok], mid[ok], E - loops - int(ok.sum())


def _snap(sp, tri, ball, x, R, tol):
    has = np.flatnonzero(ball >= 0); x, R = x.copy(), R.copy()
    if not has.size: return x, R
    Xs, Rs = tangent_spheres(sp.P[np.sort(np.c_[tri[has], ball[has]], axis=1)], sp.r[np.sort(np.c_[tri[has], ball[has]], axis=1)]); d = np.linalg.norm(Xs - x[has][:, None], axis=2); d = np.where(np.isfinite(d), d, np.inf)
    j = np.argmin(d, 1); a = np.arange(len(has)); near = d[a, j] <= tol; x[has[near]] = Xs[a, j][near]; R[has[near]] = Rs[a, j][near]; return x, R


def _subsets(m, k):
    return np.array(list(itertools.combinations(range(m), k)), np.int64).reshape(-1, k)


def _candidates(sp, tol, spacing=0.4, block=1 << 15):
    n = np.maximum(np.ceil(np.linalg.norm(sp.cell, axis=1) / spacing).astype(np.int64), 4); d2 = 2 * _half_diagonal(sp.cell / n[:, None]); m = int(np.prod(n)); sets = {}; top = -np.inf
    for s in range(0, m, block):
        g = _centres(np.arange(s, min(s + block, m)), n, sp.cell); f = sp.clear(g); lim = f + d2 + tol; top = max(top, float(f.max()) + d2 / 2); sp.need = max(sp.need, float((lim + sp.rmax).max()))
        for blk in sp.blocks(g, lim + sp.rmax):
            own, b = sp.balls(g[blk], lim[blk] + sp.rmax); v = sp.P[b] - g[blk][own]; D = np.linalg.norm(v, axis=1); gap = D - sp.r[b] - f[blk][own]; o = np.lexsort((gap, own)); first = o[np.r_[True, own[o][1:] != own[o][:-1]]]
            near = np.empty(blk.stop - blk.start, np.int64); near[own[first]] = first; k0 = near[own]
            with np.errstate(divide='ignore', invalid='ignore'): u = np.where(D[:, None] > 0, v / D[:, None], 0); slope = np.minimum(2.0, np.linalg.norm(u - u[k0], axis=1) + d2 / D + d2 / D[k0])
            keep = gap <= d2 / 2 * np.where(np.isfinite(slope), slope, 2.0) + tol; own, b = own[keep], b[keep]; cnt = np.bincount(own, minlength=blk.stop - blk.start); start = np.r_[0, np.cumsum(cnt)[:-1]]
            for c in np.unique(cnt[cnt >= 3]): sets.setdefault(int(c), []).append(np.unique(b[start[cnt == c][:, None] + np.arange(c)], axis=0))
    return {c: np.unique(np.concatenate(v), axis=0) for c, v in sorted(sets.items())}, top


def _chunks(sets, k, size=1 << 18):
    for c, K in sets.items():
        if c < k: continue
        if comb(c, k) <= size:
            S = _subsets(c, k); rows = max(1, size // len(S))
            for i in range(0, len(K), rows): yield np.unique(K[i:i + rows][:, S].reshape(-1, k), axis=0)
            continue
        it = itertools.combinations(range(c), k)
        while len(S := np.array(list(itertools.islice(it, size)), np.int64).reshape(-1, k)):
            for row in K: yield row[S]


def _enumerated(sp, tol, tolV, max_sets):
    sets, top = _candidates(sp, tol); work = sum(len(K) * (comb(c, 4) + comb(c, 3)) for c, K in sets.items())
    if work > max_sets: raise RuntimeError(f'voronoi_network: verifying the network needs {work} sets of four or three atoms, more than max_sets ({max_sets}); raise max_sets or pass verify=False')
    out = []
    for Q in _chunks(sets, 4):
        X, R = tangent_spheres(sp.P[Q], sp.r[Q]); X, R = X.reshape(-1, 3), R.ravel(); ok = np.isfinite(R); ok[ok] = sp.outside(X[ok]) <= tolV; X, R = X[ok], R[ok]; X = sp.canonical(X[sp.clear(X) >= R - tol])[0]
        _, i = np.unique(np.round(X / 1e-9).astype(np.int64), axis=0, return_index=True); out.append(X[np.sort(i)])
    X = np.concatenate(out) if out else np.zeros((0, 3)); _, i = np.unique(np.round(X / 1e-9).astype(np.int64), axis=0, return_index=True); X = X[np.sort(i)]
    X = X[~_facets(sp, X, sp.clear(X), tol / 10)[3]] if len(X) else X; return _distinct(X, sp, tolV), _isolated(sp, sets, top, tol, tolV)


def _isolated(sp, sets, top, tol, tolV):
    centres = []
    for T in _chunks(sets, 3):
        P, r = sp.P[T], sp.r[T]; ts = Trisector(P[:, 0], P[:, 1], P[:, 2], r[:, 0], r[:, 1], r[:, 2]); low = -r.min(1) - tol; e = np.flatnonzero((ts.kind == 2) & (ts.Rm - ts.a >= low) & (ts.Rm + ts.a <= top + tol))
        R0 = ts.Rm[e] + ts.a[e]; on = sp.clear(sp.canonical(ts.xyz(R0, np.zeros(len(e)), e))[0]) >= R0 - tol; e, R0 = e[on], R0[on]; m = ts.xyz(ts.Rm[e], np.zeros(len(e)), e); half = np.maximum(ts.a[e] * np.linalg.norm(ts.q[e], axis=1), ts.b[e])
        near = sp.outside(m) <= half + tol; e, R0, m, half = e[near], R0[near], m[near], half[near]; free = np.ones(len(e), bool); rho = half + np.maximum(R0, 0) + sp.rmax + tol; sp.need = max([sp.need] + list(sp.outside(m) + rho))
        for blk in sp.blocks(m, rho):
            own, b = sp.balls(m[blk], rho[blk]); i = own + blk.start; keep = np.all(b[:, None] != T[e[i]], axis=1); Rc, _ = ts.contacts(e[i[keep]], sp.P[b[keep]], sp.r[b[keep]], tol); free[i[keep][np.isfinite(Rc).any(1)]] = False
        centres.append(ts.xyz(R0[free], np.zeros(int(free.sum())), e[free]))
        dj, dk = P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]; jj = (dj * dj).sum(1); line = np.flatnonzero((np.cross(dj, dk) ** 2).sum(1) <= 1e-12 * jj * (dk * dk).sum(1))
        if not line.size: continue
        u = dj[line] / np.sqrt(jj[line])[:, None]; sj, sk = np.sqrt(jj[line]), (dk[line] * u).sum(1); ri, rj, rk = r[line].T; det = sj * (rk - ri) - sk * (rj - ri); good = np.abs(det) > 1e-12 * (1 + np.abs(sj) + np.abs(sk))
        with np.errstate(divide='ignore', invalid='ignore'):
            bj, bk = (sj * sj - rj * rj + ri * ri) / 2, (sk * sk - rk * rk + ri * ri) / 2; ax = (bj * (rk - ri) - bk * (rj - ri)) / det; Rl = (sj * bk - sk * bj) / det; rho2 = (Rl + ri) ** 2 - ax ** 2
        z = np.where(np.abs(u[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]]); w = z - (z * u).sum(1)[:, None] * u; w = w / np.linalg.norm(w, axis=1)[:, None]; c0 = P[line, 0] + ax[:, None] * u
        ok = np.flatnonzero(good & (rho2 > tol * tol) & (Rl >= low[line]) & (Rl <= top + tol)); rad = np.sqrt(rho2[ok]); ok = ok[sp.clear(sp.canonical(c0[ok] + rad[:, None] * w[ok])[0]) >= Rl[ok] - tol]; rad = np.sqrt(rho2[ok]); ok, rad = ok[sp.outside(c0[ok]) <= rad + tol], rad[sp.outside(c0[ok]) <= rad + tol]
        free = np.ones(len(ok), bool); rho = rad + np.maximum(Rl[ok], 0) + sp.rmax + tol; sp.need = max([sp.need] + list(sp.outside(c0[ok]) + rho))
        for blk in sp.blocks(c0[ok], rho):
            own, b = sp.balls(c0[ok][blk], rho[blk]); i = ok[own + blk.start]; j = own + blk.start; keep = np.all(b[:, None] != T[line[i]], axis=1); i, j, b = i[keep], j[keep], b[keep]; v = sp.P[b] - c0[i]; s = (v * u[i]).sum(1); q = np.linalg.norm(v - s[:, None] * u[i], axis=1); t = (Rl[i] + sp.r[b]) ** 2
            hit = (Rl[i] + sp.r[b] >= -tol) & (t >= s * s + (np.sqrt(rho2[i]) - q) ** 2 - tol) & (t <= s * s + (np.sqrt(rho2[i]) + q) ** 2 + tol); free[j[hit]] = False
        centres.append(c0[ok][free] + rad[free][:, None] * w[ok][free])
    return len(_distinct(sp.canonical(np.concatenate(centres))[0], sp, tolV)) if centres else 0


def _match(X, sp, x, tol):
    best = -np.ones(len(x), np.int64); shift = np.zeros((len(x), 3), np.int64)
    if not len(X) or not len(x): return best, shift
    tree = cKDTree(X)
    for s in SHIFTS:
        d, j = tree.query(x + s @ sp.cell, distance_upper_bound=tol, workers=sp.workers); hit = (best < 0) & np.isfinite(d); best[hit] = j[hit]; shift[hit] = s
    return best, shift


def _distinct(x, sp, tol):
    if len(x) < 2: return x
    tree = cKDTree(x); dup = np.zeros(len(x), bool)
    for s in SHIFTS:
        lists = tree.query_ball_point(x + s @ sp.cell, tol, workers=sp.workers); own = np.repeat(np.arange(len(x)), [len(l) for l in lists]); j = np.fromiter((b for l in lists for b in l), np.int64, len(own)); dup[own[j < own]] = True
    return x[~dup]


def _skeleton(sp, seeds, tol, tolV, cap, bound, block=2048):
    X = np.zeros((0, 3)); F = np.zeros(0); queue = sp.canonical(seeds)[0]; bad = 0; failed = 0; parts = []
    while len(queue):
        j, _ = _match(X, sp, queue, tolV); uniq = _distinct(queue[j < 0], sp, tolV)
        if not len(uniq): break
        f = sp.clear(uniq); tri, O, src, nb = _facets(sp, uniq, f, tol / 10)
        if not len(X): keep = ~nb; tri, O, src = tri[keep[src]], O[keep[src]], np.cumsum(keep)[src[keep[src]]] - 1; uniq, f = uniq[keep], f[keep]
        else: bad += int(nb.sum())
        base = len(X); X = np.r_[X, uniq]; F = np.r_[F, f]; src = src + base
        if len(X) > cap: raise RuntimeError(f'voronoi_network: more than {cap} vertices traced')
        out = []
        for c in range(0, len(src), block):
            e, xw, neck, mid, nf = _trace(sp, X[src[c:c + block]], F[src[c:c + block]], tri[c:c + block], O[c:c + block], tol, bound); failed += nf; e = e + c; parts.append((src[e], xw, neck, mid - X[src[e]])); out.append(xw)
        queue = sp.canonical(np.concatenate(out))[0] if out else np.zeros((0, 3))
    return X, F, parts, bad, failed


def voronoi_network(positions, cell, radii, k=24, tol_A=1e-7, max_images=1 << 18, max_pairs=1 << 19, workers=-1, verify=True, max_sets=None):
    """The additively weighted Voronoi network of the atom balls: nodes with their clearance, edges with neck clearance and lattice offset, and its completeness."""
    pos, cell = _valid(positions, cell, 'voronoi_network'); rad0 = _radii(radii, len(pos), 'voronoi_network')
    workers = threads(workers, 'voronoi_network')
    if not isinstance(verify, bool): raise ValueError(f'voronoi_network: verify must be True or False, not {verify!r}')
    if isinstance(k, bool) or not float(k).is_integer() or k < 4 or not (np.isfinite(tol_A) and tol_A > 0): raise ValueError(f'voronoi_network: k must be an integer of at least 4 and tol_A positive and finite, not {k!r} and {tol_A!r}')
    if max_sets is not None and (isinstance(max_sets, bool) or not float(max_sets).is_integer() or max_sets < 1): raise ValueError(f'voronoi_network: max_sets must be None or a positive integer, not {max_sets!r}')
    if isinstance(max_pairs, bool) or not float(max_pairs).is_integer() or max_pairs < 1: raise ValueError(f'voronoi_network: max_pairs must be a positive integer, not {max_pairs!r}')
    max_pairs = int(max_pairs); pos = folded(pos, cell, 'voronoi_network')[0]; n_at = len(pos); tolV = 1e1 * tol_A; rmax = float(rad0.max()); reach = 3.0 * (volume(cell) / n_at) ** (1 / 3) + 2 * rmax; bound = np.inf
    for _ in range(8):
        sp = _Space(pos, cell, rad0, reach, k, max_images, max_pairs, workers)
        try: seeds, why = _seeds(sp, tol_A), 'voronoi_network: no vertex certified among the tetrahedra of the regular triangulation'
        except RuntimeError as e: seeds, why = np.zeros((0, 3)), str(e)
        if not len(seeds):
            if not np.isfinite(bound): bound = _largest_clearance(pos, cell, rad0, workers) + tolV
            if reach >= 3 * (bound + rmax) + tolV: raise RuntimeError(why)
            reach = min(2 * reach, 3 * (bound + rmax) + tolV); continue
        guess = 3 * (float(sp.clear(seeds).max()) + sp.rmax) + 1.0
        if guess > reach: sp = _Space(pos, cell, rad0, guess, k, max_images, max_pairs, workers); reach = guess
        X, F, parts, bad, failed = _skeleton(sp, seeds, tol_A, tolV, 60 * n_at, bound); V, iso, add = None, None, np.zeros((0, 3))
        if verify and sp.need <= reach:
            V, iso = _enumerated(sp, tol_A, tolV, (1 << 18) * n_at if max_sets is None else int(max_sets)); add = V[_match(X, sp, V, tolV)[0] < 0]
            if len(add): X, F, parts, bad, failed = _skeleton(sp, np.r_[seeds, add], tol_A, tolV, 60 * n_at, bound)
        if sp.need <= reach: break
        if not np.isfinite(bound): bound = _largest_clearance(pos, cell, rad0, workers) + tolV
        reach = min(max(sp.need * 1.05, reach * 1.25), max(3 * (bound + rmax) + tolV, reach))
    else: raise RuntimeError(f'voronoi_network: the image reach did not converge ({reach:.1f} A)')
    u = np.concatenate([q[0] for q in parts]); xw = np.concatenate([q[1] for q in parts]); neck = np.concatenate([q[2] for q in parts]); rel = np.concatenate([q[3] for q in parts])
    xc, nw = sp.canonical(xw); w, s = _match(X, sp, xc, tolV); hit = w >= 0; failed += int((~hit).sum()); u, w, d, neck, rel, xw = u[hit], w[hit], nw[hit] - s[hit], neck[hit], rel[hit], xw[hit]
    flip = (u > w) | ((u == w) & ((d[:, 0] < 0) | ((d[:, 0] == 0) & ((d[:, 1] < 0) | ((d[:, 1] == 0) & (d[:, 2] < 0))))))
    a, b = np.where(flip, w, u), np.where(flip, u, w); rel = np.where(flip[:, None], rel - (xw - X[u]), rel); d = np.where(flip[:, None], -d, d)
    _, first = np.unique(np.c_[a, b, d, np.round(rel / 1e-5).astype(np.int64)], axis=0, return_index=True); e = np.sort(first); cu, cv, duv = a[e], b[e], d[e]; neck = np.minimum(neck[e], np.minimum(F[cu], F[cv]))
    complete = bool(len(X) == len(V) and (_match(X, sp, V, tolV)[0] >= 0).all() and (_match(V, sp, X, tolV)[0] >= 0).all()) if verify else None
    return dict(n_nodes=len(X), canon_clear=F, canon_xyz=X, edges=(cu, cv, duv), neck=neck, n_uncertified_nodes=bad, n_uncertified_edges=failed, complete=complete,
                n_nodes_enumerated=len(V) if verify else None, n_nodes_added=len(add) if verify else None, n_isolated_edges=iso)


def _exact(net):
    return net['n_uncertified_nodes'] == 0 and net['n_uncertified_edges'] == 0 and net['complete'] is True


def free_sphere(net):
    """Radius (A) of the largest free sphere, NaN when no loop of the traced edges percolates."""
    cu, cv, duv = net['edges']; dsu = _Offsets(net['n_nodes'])
    for e in np.argsort(-net['neck'], kind='stable'):
        if net['neck'][e] < 0: break
        if dsu.union(int(cu[e]), int(cv[e]), duv[e]): return float(net['neck'][e])
    return float('nan')


def descriptors(net):
    """Diameters Di, Df and Dif (A), cavity-to-neck ratio, percolation (None when undecided), node and edge counts, completeness and exactness."""
    ri = float(net['canon_clear'].max()); rf = free_sphere(net); nn, ne = int(net['n_uncertified_nodes']), int(net['n_uncertified_edges']); exact = _exact(net)
    go = True if np.isfinite(rf) else (False if exact else None); on = _percolating_nodes(net, rf) if go else np.zeros(net['n_nodes'], bool)
    return dict(Di_A=2 * ri, Df_A=2 * rf, Dif_A=2 * float(net['canon_clear'][on].max()) if on.any() else np.nan, cavity_to_neck=ri / rf if rf > 0 else np.nan, percolates=go, n_nodes=net['n_nodes'], n_edges=len(net['neck']),
                exact=exact, n_uncertified_nodes=nn, n_uncertified_edges=ne, complete=net['complete'], n_nodes_enumerated=net['n_nodes_enumerated'], n_nodes_added=net['n_nodes_added'], n_isolated_edges=net['n_isolated_edges'])


def _percolating_nodes(net, probe):
    cu, cv, duv = net['edges']; dsu = _Offsets(net['n_nodes']); roots = set()
    for e in np.flatnonzero(net['neck'] >= probe):
        if dsu.union(int(cu[e]), int(cv[e]), duv[e]): roots.add(dsu.root(int(cu[e]))[0])
    roots = {dsu.root(r)[0] for r in roots}; return np.array([dsu.root(i)[0] in roots for i in range(net['n_nodes'])], bool)


def percolating_edges(net, probe=None):
    """Mask of edges on a percolating component at the probe radius (default: the free sphere), and the probe used."""
    if probe is None:
        probe = free_sphere(net)
        if not np.isfinite(probe): return np.zeros(len(net['neck']), bool), probe
    if not np.isfinite(probe) or probe < 0: raise ValueError(f'percolating_edges: the probe radius must be finite and non-negative, not {probe}')
    cu, cv, _ = net['edges']; node = _percolating_nodes(net, probe); keep = net['neck'] >= probe
    return keep & node[cu] & node[cv], probe


def pore_network(net, probe, merge_frac=0.92):
    """Pores, throats and percolating pores at a probe radius; decided False when the network is not exact and no pore percolates."""
    if not np.isfinite(probe) or probe < 0: raise ValueError(f'pore_network: the probe radius must be finite and non-negative, not {probe}')
    if not 0 <= merge_frac <= 1: raise ValueError(f'pore_network: merge_frac must lie in [0, 1], not {merge_frac}')
    cu, cv, duv = net['edges']; neck, node_r, xyz = net['neck'], net['canon_clear'], net['canon_xyz']; n = net['n_nodes']
    live_node = node_r >= probe; live = (neck >= probe) & live_node[cu] & live_node[cv]; dsu = _Offsets(n)
    for e in np.flatnonzero(live & (neck >= merge_frac * np.minimum(node_r[cu], node_r[cv]))): dsu.union(int(cu[e]), int(cv[e]), duv[e])
    lab = np.array([dsu.root(i)[0] for i in range(n)], np.int64); keep = np.flatnonzero(live_node)
    if keep.size == 0: return dict(pore_xyz=np.zeros((0, 3)), pore_r=np.zeros(0), pore_percolating=np.zeros(0, bool), throat_pairs=np.zeros((0, 2), np.int64), throat_r=np.zeros(0), probe=probe, merge_frac=merge_frac, decided=_exact(net))
    _, first, inv = np.unique(lab[keep], return_index=True, return_inverse=True); inv = np.argsort(np.argsort(first, kind='stable'), kind='stable')[inv]; pore_r = np.full(len(first), -np.inf); pore_xyz = np.zeros((len(first), 3))
    for i, node in enumerate(keep):
        if node_r[node] > pore_r[inv[i]]: pore_r[inv[i]] = node_r[node]; pore_xyz[inv[i]] = xyz[node]
    idx = -np.ones(n, np.int64); idx[keep] = inv; pa, pb = idx[cu], idx[cv]; real = live & (pa >= 0) & (pb >= 0) & (pa != pb); throat = {}
    for e in np.flatnonzero(real):
        key = (min(int(pa[e]), int(pb[e])), max(int(pa[e]), int(pb[e])))
        if neck[e] > throat.get(key, -np.inf): throat[key] = float(neck[e])
    node_perc = _percolating_nodes(net, probe); pore_perc = np.zeros(len(first), bool)
    for i, node in enumerate(keep):
        if node_perc[node]: pore_perc[inv[i]] = True
    tk = np.array(sorted(throat), np.int64).reshape(-1, 2)
    return dict(pore_xyz=pore_xyz, pore_r=pore_r, pore_percolating=pore_perc, throat_pairs=tk, throat_r=np.array([throat[tuple(k)] for k in tk]) if len(tk) else np.zeros(0), probe=probe, merge_frac=merge_frac, decided=bool(_exact(net) or node_perc.any()))
