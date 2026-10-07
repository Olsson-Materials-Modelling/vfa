"""Rings of a periodic bond graph under three criteria (primitive = Franzblau shortest-path, King, Guttman), enumerated on the covering graph so that helical closed walks are never rings; sizes count atoms, or network-former units on a bipartite graph."""
from collections import deque
import numpy as np
from .cell import minimum_image


def _adjacency(n, edges, offsets):
    adj = [[] for _ in range(n)]
    for (i, j), o in zip(edges, offsets): adj[i].append((j, tuple(int(x) for x in o))); adj[j].append((i, tuple(-int(x) for x in o)))
    return adj


def _step(node, nbr, o):
    return (nbr, (node[1][0] + o[0], node[1][1] + o[1], node[1][2] + o[2]))


def _distances(adj, source, cutoff):
    dist = {(source, (0, 0, 0)): 0}; q = deque([(source, (0, 0, 0))])
    while q:
        node = q.popleft(); d = dist[node]
        if d >= cutoff: continue
        for nbr, o in adj[node[0]]:
            nxt = _step(node, nbr, o)
            if nxt not in dist: dist[nxt] = d + 1; q.append(nxt)
    return dist


def canonical(members):
    """Rotation- and reversal-invariant identity of a cycle."""
    seq = list(members); n = len(seq); return min(tuple(d[r:] + d[:r]) for d in (seq, seq[::-1]) for r in range(n))


def _primitive(dmaps, path):
    L = len(path)
    for a in range(L):
        da = dmaps[path[a][0]]; base = path[a][1]
        for b in range(a + 1, L):
            rel = tuple(path[b][1][k] - base[k] for k in range(3)); true = da.get((path[b][0], rel))
            if true is not None and true < min(b - a, L - (b - a)): return False
    return True


def primitive_rings(n_atoms, edges, offsets, max_length, bipartite=True):
    """Primitive (Franzblau shortest-path) rings, no shortcut between any two ring atoms, up to max_length atoms, each once, as atom-index tuples."""
    adj = _adjacency(n_atoms, edges, offsets); dmaps = [_distances(adj, a, max_length // 2 + 1) for a in range(n_atoms)]; found = {}
    for root in range(n_atoms):
        stack = [((root, (0, 0, 0)), [(root, (0, 0, 0))])]
        while stack:
            node, path = stack.pop()
            if len(path) > max_length: continue
            for nbr, o in adj[node[0]]:
                nxt = _step(node, nbr, o)
                if nxt == (root, (0, 0, 0)):
                    if len(path) >= 3 and (not bipartite or len(path) % 2 == 0) and _primitive(dmaps, path): found.setdefault(canonical([p[0] for p in path]), tuple(p[0] for p in path))
                    continue
                if nbr < root or nxt in path: continue
                back = dmaps[root].get(nxt)
                if back is None or len(path) + back > max_length: continue
                if len(path) < max_length: stack.append((nxt, path + [nxt]))
    return sorted(found.values(), key=lambda r: (len(r), canonical(r)))


def _shortest_paths(adj, source, target, cutoff, blocked_node=None, blocked_edge=None):
    """All shortest paths from source to target in the covering graph, avoiding one node or one edge; empty if none within the cutoff."""
    ok = lambda a, b: (blocked_node is None or (a != blocked_node and b != blocked_node)) and (blocked_edge is None or {a, b} != blocked_edge)
    dist = {source: 0}; q = deque([source]); pred = {source: []}
    while q:
        node = q.popleft(); d = dist[node]
        if node == target or d >= cutoff: continue
        for nbr, o in adj[node[0]]:
            nxt = _step(node, nbr, o)
            if not ok(node, nxt): continue
            if nxt not in dist: dist[nxt] = d + 1; pred[nxt] = [node]; q.append(nxt)
            elif dist[nxt] == d + 1: pred[nxt].append(node)
    if target not in dist: return []
    paths, stack = [], [(target, [target])]
    while stack:
        node, p = stack.pop()
        if node == source: paths.append(p[::-1]); continue
        for pr in pred[node]: stack.append((pr, p + [pr]))
    return paths


def king_rings(n_atoms, edges, offsets, max_length, bipartite=True):
    """King rings: per atom and pair of its neighbours, the shortest paths between the neighbours avoiding the atom; each once."""
    adj = _adjacency(n_atoms, edges, offsets); found = {}
    for c in range(n_atoms):
        base = (c, (0, 0, 0)); nb = [_step(base, j, o) for j, o in adj[c]]
        for x in range(len(nb)):
            for y in range(x + 1, len(nb)):
                for p in _shortest_paths(adj, nb[x], nb[y], max_length - 2, blocked_node=base):
                    if len(p) + 1 >= 3 and (not bipartite or (len(p) + 1) % 2 == 0): cyc = [c] + [n[0] for n in p]; found.setdefault(canonical(cyc), tuple(cyc))
    return sorted(found.values(), key=lambda r: (len(r), canonical(r)))


def guttman_rings(n_atoms, edges, offsets, max_length, bipartite=True):
    """Guttman rings: per bond, the shortest paths between its atoms avoiding the bond; each once."""
    adj = _adjacency(n_atoms, edges, offsets); found = {}
    for (i, j), o in zip(edges, offsets):
        a = (int(i), (0, 0, 0)); b = (int(j), tuple(int(x) for x in o))
        for p in _shortest_paths(adj, a, b, max_length - 1, blocked_edge={a, b}):
            if len(p) >= 3 and (not bipartite or len(p) % 2 == 0): found.setdefault(canonical([n[0] for n in p]), tuple(n[0] for n in p))
    return sorted(found.values(), key=lambda r: (len(r), canonical(r)))


CRITERIA = {'primitive': primitive_rings, 'king': king_rings, 'guttman': guttman_rings}


def rings(n_atoms, edges, offsets, max_length, criterion='primitive', bipartite=True):
    """Rings under one criterion up to max_length atoms."""
    return CRITERIA[criterion](n_atoms, edges, offsets, max_length, bipartite)


def rings_converged(n_atoms, edges, offsets, criterion='primitive', bipartite=True, start=16, step=4, hard_cap=48):
    """Rings with the cap raised by step until two consecutive caps agree; returns the rings and the confirming cap."""
    prev = None; cap = start
    while cap <= hard_cap:
        r = rings(n_atoms, edges, offsets, cap, criterion, bipartite)
        if prev is not None and len(r) == len(prev): return r, cap
        prev, cap = r, cap + step
    raise RuntimeError(f'{criterion} rings did not converge below {hard_cap} atoms')


def ring_sizes(found, bipartite=True):
    """Ring size: network-former units on a bipartite graph, atoms otherwise."""
    return np.array([len(r) // 2 if bipartite else len(r) for r in found], int)


def ring_extents(found, positions, cell):
    """Largest distance between two atoms of each ring, unwrapped along the cycle: a ring whose extent reaches a cell length runs through the periodic images and is a finite-size artefact, not a ring of the material."""
    pos = np.asarray(positions, float); out = np.empty(len(found))
    for k, ring in enumerate(found):
        p = [pos[ring[0]]]
        for a in ring[1:]: p.append(p[-1] + minimum_image(pos[a] - p[-1], cell))
        p = np.array(p); out[k] = np.max(np.linalg.norm(p[:, None] - p[None], axis=2))
    return out

