"""Spheres touching atom balls from outside in closed form: tangent spheres of four balls, the trisector curve of three."""
import numpy as np


def tangent_spheres(C, r, rank_tol=1e-10):
    """Centres (n, 2, 3) and radii (n, 2) of the spheres touching each quadruple of balls from outside, NaN where absent."""
    C = np.asarray(C, float); r = np.asarray(r, float); d = C[:, 1:] - C[:, :1]; al = (d * d).sum(2) - r[:, 1:] ** 2 + r[:, :1] ** 2
    A = np.concatenate([2 * d, 2 * (r[:, 1:] - r[:, :1])[:, :, None]], 2); U, s, Vt = np.linalg.svd(A); full = s[:, 2] > rank_tol * s[:, 0]
    inv = np.where(s > rank_tol * s[:, :1], 1 / np.where(s > 0, s, 1), 0); zp = np.einsum('nji,nj->ni', Vt[:, :3], inv * np.einsum('nji,nj->ni', U, al)); zn = Vt[:, 3]
    yp, Rp, yn, Rn = zp[:, :3], zp[:, 3] + r[:, 0], zn[:, :3], zn[:, 3]
    a = (yn * yn).sum(1) - Rn ** 2; b = (yp * yn).sum(1) - Rp * Rn; c = (yp * yp).sum(1) - Rp ** 2; disc = b * b - a * c
    sq = np.sqrt(np.maximum(disc, 0)); q = -(b + np.where(b >= 0, 1, -1) * sq); lin = np.abs(a) <= 1e-12 * np.maximum((yn * yn).sum(1), 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        t1 = np.where(lin, -c / (2 * b), q / a); t2 = np.where(lin, np.nan, c / q)
    t = np.stack([t1, t2], 1); t[(disc < 0) & ~lin] = np.nan
    X = C[:, :1] + yp[:, None] + t[:, :, None] * yn[:, None]; R = Rp[:, None] + t * Rn[:, None] - r[:, :1]
    X[~full] = np.nan; R[~full] = np.nan; X[~full, 0] = C[~full, 0] + yp[~full]; R[~full, 0] = np.linalg.norm(yp[~full], axis=1) - r[~full, 0]
    off = ~full & (np.abs(np.linalg.norm(X[:, :1] - C, axis=2) - R[:, :1] - r).max(1) > 1e-9 * (1 + np.abs(d).max((1, 2)))); X[off] = np.nan; R[off] = np.nan
    bad = ~np.isfinite(R) | (R + r.min(1)[:, None] < -1e-9) | ~np.all(np.isfinite(X), axis=2); X[bad] = np.nan; R[bad] = np.nan
    return X, R


class Trisector:
    """The curve of sphere centres touching three balls from outside: a branch (kind 0), component (1) or ellipse (2)."""

    def __init__(self, ci, cj, ck, ri, rj, rk):
        dj, dk = cj - ci, ck - ci; n = np.cross(dj, dk); jj, jk, kk = (dj * dj).sum(1), (dj * dk).sum(1), (dk * dk).sum(1); det = jj * kk - jk * jk; line = det <= 1e-12 * jj * kk; det = np.where(line, 1, det)
        n = n / np.where(line, 1, np.linalg.norm(n, axis=1))[:, None]; al = np.stack([jj - rj ** 2 + ri ** 2, kk - rk ** 2 + ri ** 2], 1) / 2; be = -np.stack([rj - ri, rk - ri], 1)
        Gi = np.stack([np.stack([kk, -jk], 1), np.stack([-jk, jj], 1)], 1) / det[:, None, None]; ca, cb = np.einsum('nij,nj->ni', Gi, al), np.einsum('nij,nj->ni', Gi, be)
        self.c, self.ri, self.n = ci, ri, n; self.p0 = ca[:, :1] * dj + ca[:, 1:] * dk; self.q = cb[:, :1] * dj + cb[:, 1:] * dk
        self.A = 1 - (self.q * self.q).sum(1); self.B = ri - (self.p0 * self.q).sum(1); self.C0 = ri ** 2 - (self.p0 * self.p0).sum(1); D = self.B ** 2 - self.A * self.C0
        self.kind = np.where(line, -1, np.where(self.A < -1e-9, np.where(D > 0, 2, -1), np.where((self.A > 1e-9) & (D < 0), 1, 0)))
        with np.errstate(divide='ignore', invalid='ignore'):
            self.Rm = -self.B / self.A; Hm = -D / self.A; self.a = np.sqrt(np.abs(Hm / self.A)); self.b = np.sqrt(np.abs(Hm))

    def xyz(self, R, h, i=slice(None)):
        """Points of the curves i at tangent radius R and height h."""
        return self.c[i] + self.p0[i] + R[:, None] * self.q[i] + h[:, None] * self.n[i]

    def point(self, t, s, i=slice(None)):
        """Tangent radius and height at parameter t on branch or component s of the curves i."""
        k = self.kind[i]; R = np.full(len(t), np.nan); h = np.full(len(t), np.nan); A, B, C0 = self.A[i], self.B[i], self.C0[i]
        with np.errstate(divide='ignore', invalid='ignore'):
            m = k == 0; c = C0[m] - t[m] ** 2; R[m] = -c / (B[m] + s[m] * np.sqrt(np.maximum(B[m] ** 2 - A[m] * c, 0))); h[m] = t[m]
            m = k == 1; R[m] = t[m]; h[m] = s[m] * np.sqrt(np.maximum(A[m] * t[m] ** 2 + 2 * B[m] * t[m] + C0[m], 0))
            m = k == 2; R[m] = self.Rm[i][m] + self.a[i][m] * np.cos(t[m]); h[m] = self.b[i][m] * np.sin(t[m])
        return R, h

    def param(self, R, h, i=slice(None)):
        """Parameter and branch or component of points (R, h) on the curves i."""
        k = self.kind[i]; t = np.full(len(R), np.nan); s = np.ones(len(R)); A, B, C0 = self.A[i], self.B[i], self.C0[i]
        with np.errstate(divide='ignore', invalid='ignore'):
            m = k == 0; c = C0[m] - h[m] ** 2; sq = np.sqrt(np.maximum(B[m] ** 2 - A[m] * c, 0)); t[m] = h[m]
            s[m] = np.where(np.abs(-c / (B[m] + sq) - R[m]) <= np.abs(-c / (B[m] - sq) - R[m]), 1.0, -1.0)
            m = k == 1; t[m] = R[m]; s[m] = np.sign(h[m])
            m = k == 2; t[m] = np.arctan2(h[m] / self.b[i][m], (R[m] - self.Rm[i][m]) / self.a[i][m])
        return t, s

    def contacts(self, i, ce, re, tol):
        """Up to four (R, h) at which the ball (ce, re) touches the sphere on curve i, NaN where absent."""
        de = ce - self.c[i]; z = (de * self.n[i]).sum(1); p0, q, A, B, C0, ri = self.p0[i], self.q[i], self.A[i], self.B[i], self.C0[i], self.ri[i]
        l0 = -2 * (de * p0).sum(1) + (de * de).sum(1) + ri ** 2 - re ** 2; l1 = -2 * (de * q).sum(1) + 2 * (ri - re); L = 1 + np.linalg.norm(de, axis=1)
        a2 = l1 * l1 - 4 * z * z * A; b2 = l0 * l1 - 4 * z * z * B; c2 = l0 * l0 - 4 * z * z * C0; disc = b2 * b2 - a2 * c2; flat = np.abs(z) <= 1e-10 * L
        with np.errstate(divide='ignore', invalid='ignore'):
            sq = np.sqrt(np.maximum(disc, 0)); qq = -(b2 + np.where(b2 >= 0, 1, -1) * sq); quad = np.abs(a2) > 1e-14 * np.maximum(np.abs(b2), 1)
            R1 = np.where(quad, qq / a2, -c2 / (2 * b2)); R2 = np.where(quad, c2 / qq, np.nan); Rf = -l0 / l1; Hf = A * Rf * Rf + 2 * B * Rf + C0; hf = np.sqrt(np.maximum(Hf, 0))
            Rs = np.stack([R1, R2, Rf, Rf], 1); hs = np.stack([(l0 + l1 * R1) / (2 * z), (l0 + l1 * R2) / (2 * z), hf, -hf], 1)
        ok = np.stack([~flat & (disc >= 0), ~flat & (disc >= 0) & quad, flat & (np.abs(l1) > 1e-10 * L) & (Hf >= 0), flat & (np.abs(l1) > 1e-10 * L) & (Hf >= 0)], 1)
        ok &= np.isfinite(Rs) & np.isfinite(hs) & (Rs + re[:, None] >= -tol) & (Rs + ri[:, None] >= -tol); return np.where(ok, Rs, np.nan), np.where(ok, hs, np.nan)
