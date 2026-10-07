"""The Voronoi network of the atom centres as the pore topology: largest included and free spheres, percolation, pores and throats."""
import numpy as np
from scipy.spatial import Voronoi, cKDTree
from ..structure.cell import images, volume
from .clearance import exact_clearance


class _Offsets:
    """Union-find carrying the lattice offset from each node to its root; a union that closes a loop with a net translation is a percolating path."""

    def __init__(self, n):
        self.p = np.arange(n); self.off = np.zeros((n, 3), np.int64)

    def root(self, a):
        acc = np.zeros(3, np.int64)
        while self.p[a] != a: acc = acc + self.off[a]; a = self.p[a]
        return a, acc

    def union(self, u, v, duv):
        ru, ou = self.root(u); rv, ov = self.root(v)
        if ru == rv: return not np.array_equal(ov - ou, duv)
        self.p[rv] = ru; self.off[rv] = ou + duv - ov; return False


def voronoi_network(positions, cell, radii, edge_samples=33, k=24, margin=0.3):
    """The Voronoi network of the atom surfaces: nodes (one per image class) with their clearance, edges with the neck clearance and the lattice offset they cross."""
    cell = np.asarray(cell, float); inv = np.linalg.inv(cell); rep, rad = images(positions, cell, radii); tree = cKDTree(rep); vor = Voronoi(rep)
    clear = lambda pts: exact_clearance(tree, rad, pts, k)
    V = vor.vertices; f = V @ inv; usable = np.all((f >= -margin) & (f < 1 + margin), axis=1); keep = np.flatnonzero(usable)
    if keep.size < 4: return None
    remap = -np.ones(len(V), np.int64); remap[keep] = np.arange(keep.size); fk = f[keep]; off = np.floor(fk).astype(np.int64); wrapped = (fk - off) @ cell
    groups = cKDTree(wrapped).query_ball_tree(cKDTree(wrapped), r=1e-6); canon = -np.ones(keep.size, np.int64); nc = 0
    for a, g in enumerate(groups):
        if canon[a] < 0: canon[g] = nc; nc += 1
    node_clear = clear(V[keep]); canon_clear = np.zeros(nc); np.maximum.at(canon_clear, canon, node_clear)
    pairs = set()
    for ridge in vor.ridge_vertices:
        if -1 in ridge or len(ridge) < 2: continue
        for a, b in zip(ridge, ridge[1:] + ridge[:1]):
            if usable[a] and usable[b] and remap[a] != remap[b]: pairs.add((min(remap[a], remap[b]), max(remap[a], remap[b])))
    if not pairs: return None
    pairs = np.array(sorted(pairs), np.int64); A, B = V[keep][pairs[:, 0]], V[keep][pairs[:, 1]]; t = np.linspace(0, 1, edge_samples)
    neck = clear((A[:, None] * (1 - t)[None, :, None] + B[:, None] * t[None, :, None]).reshape(-1, 3)).reshape(len(pairs), edge_samples).min(1)
    inside = np.all((fk >= 0) & (fk < 1), axis=1); canon_xyz = np.zeros((nc, 3)); seen = np.zeros(nc, bool)
    for a in list(np.flatnonzero(inside)) + list(range(len(canon))):
        if not seen[canon[a]]: canon_xyz[canon[a]] = V[keep][a]; seen[canon[a]] = True
    return dict(n_nodes=nc, canon=canon, canon_clear=canon_clear, canon_xyz=canon_xyz, edges=(canon[pairs[:, 0]], canon[pairs[:, 1]], off[pairs[:, 1]] - off[pairs[:, 0]]), neck=neck,
                xyz=V[keep], pairs=pairs, node_clear=node_clear, cell=cell, volume=volume(cell))


def free_sphere(net):
    """Radius of the largest percolating sphere: edges added widest first until a loop with a net lattice translation closes."""
    cu, cv, duv = net['edges']; dsu = _Offsets(net['n_nodes'])
    for e in np.argsort(-net['neck']):
        if dsu.union(int(cu[e]), int(cv[e]), duv[e]): return float(net['neck'][e])
    return float('nan')


def descriptors(net):
    """Diameters (A) of the largest included sphere, the largest free sphere and the largest included sphere on the percolating path, and their ratio."""
    ri = float(net['canon_clear'].max()); rf = free_sphere(net); reach = net['canon_clear'][net['canon_clear'] >= rf] if np.isfinite(rf) else np.zeros(0)
    return dict(Di_A=2 * ri, Df_A=2 * rf, Dif_A=2 * float(reach.max()) if reach.size else np.nan, cavity_to_neck=ri / rf if rf > 0 else np.nan, n_nodes=net['n_nodes'], n_edges=len(net['neck']))


def _percolating_nodes(net, probe):
    cu, cv, duv = net['edges']; dsu = _Offsets(net['n_nodes']); roots = set()
    for e in np.flatnonzero(net['neck'] >= probe):
        if dsu.union(int(cu[e]), int(cv[e]), duv[e]): roots.add(int(dsu.root(int(cu[e]))[0]))
    every = np.array([dsu.root(i)[0] for i in range(net['n_nodes'])]); roots = {int(dsu.root(r)[0]) for r in roots}
    return np.isin(every, list(roots)) if roots else np.zeros(net['n_nodes'], bool)


def percolating_edges(net, probe=None):
    """Edges carrying a percolating component at the probe radius (default: the free-sphere radius)."""
    probe = free_sphere(net) if probe is None else probe; cu, cv, _ = net['edges']; node = _percolating_nodes(net, probe); keep = net['neck'] >= probe
    return keep & node[cu] & node[cv], probe


def pore_network(net, probe, merge_frac=0.92):
    """Pores and throats at a probe radius: nodes merge when the neck between them is at least merge_frac of the smaller clearance."""
    cu, cv, duv = net['edges']; neck, node_r, xyz = net['neck'], net['canon_clear'], net['canon_xyz']; n = net['n_nodes']
    live_node = node_r >= probe; live = (neck >= probe) & live_node[cu] & live_node[cv]; parent = np.arange(n)

    def find(a):
        while parent[a] != a: parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for e in np.flatnonzero(live & (neck >= merge_frac * np.minimum(node_r[cu], node_r[cv]))):
        ra, rb = find(int(cu[e])), find(int(cv[e]))
        if ra != rb: parent[rb] = ra
    lab = np.array([find(i) for i in range(n)]); keep = np.flatnonzero(live_node)
    if keep.size == 0: return dict(pore_xyz=np.zeros((0, 3)), pore_r=np.zeros(0), pore_percolating=np.zeros(0, bool), throat_pairs=np.zeros((0, 2), np.int64), throat_r=np.zeros(0), probe=probe, merge_frac=merge_frac)
    uniq, inv = np.unique(lab[keep], return_inverse=True); pore_r = np.zeros(len(uniq)); pore_xyz = np.zeros((len(uniq), 3))
    for i, node in enumerate(keep):
        if node_r[node] > pore_r[inv[i]]: pore_r[inv[i]] = node_r[node]; pore_xyz[inv[i]] = xyz[node]
    idx = -np.ones(n, np.int64); idx[keep] = inv; pa, pb = idx[cu], idx[cv]; real = live & (pa >= 0) & (pb >= 0) & (pa != pb); throat = {}
    for e in np.flatnonzero(real):
        key = (min(int(pa[e]), int(pb[e])), max(int(pa[e]), int(pb[e])))
        if neck[e] > throat.get(key, -np.inf): throat[key] = float(neck[e])
    node_perc = _percolating_nodes(net, probe); pore_perc = np.zeros(len(uniq), bool)
    for i, node in enumerate(keep):
        if node_perc[node]: pore_perc[inv[i]] = True
    tk = np.array(sorted(throat), np.int64).reshape(-1, 2)
    return dict(pore_xyz=pore_xyz, pore_r=pore_r, pore_percolating=pore_perc, throat_pairs=tk, throat_r=np.array([throat[tuple(k)] for k in tk]) if len(tk) else np.zeros(0), probe=probe, merge_frac=merge_frac)
