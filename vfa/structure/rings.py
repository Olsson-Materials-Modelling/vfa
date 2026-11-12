"""Rings of a periodic bond graph under three criteria (primitive: Franzblau's shortest-path rings; King; Guttman), enumerated on the covering graph."""
from collections import deque, Counter
import numpy as np
from .cell import minimum_image


def _adjacency(n, edges, offsets):
    adj = [[] for _ in range(n)]
    for (i, j), o in zip(edges, offsets): i, j, o = int(i), int(j), tuple(int(x) for x in o); adj[i].append((j, o)); adj[j].append((i, (-o[0], -o[1], -o[2])))
    return adj


def _two_colour(n_atoms, edges, name):
    nbrs = [[] for _ in range(n_atoms)]; col = [-1] * n_atoms
    for i, j in edges: nbrs[int(i)].append(int(j)); nbrs[int(j)].append(int(i))
    for s in range(n_atoms):
        if col[s] >= 0: continue
        col[s] = 0; q = deque([s])
        while q:
            a = q.popleft()
            for b in nbrs[a]:
                if col[b] < 0: col[b] = 1 - col[a]; q.append(b)
                elif col[b] == col[a]: raise ValueError(f'{name}: the bond graph is not bipartite: atoms {a} and {b} are bonded and take the same colour, an odd cycle that may close through the periodic images; pass bipartite=False to count ring sizes in atoms')


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


def _node_key(nodes):
    n = len(nodes); m = min(x[0] for x in nodes); best = None
    for seq in (nodes, nodes[::-1]):
        for r in range(n):
            if seq[r][0] != m: continue
            o = seq[r][1]; k = tuple((seq[(r + s) % n][0], (seq[(r + s) % n][1][0] - o[0], seq[(r + s) % n][1][1] - o[1], seq[(r + s) % n][1][2] - o[2])) for s in range(n))
            if best is None or k < best: best = k
    return best


def canonical(members):
    """Identity of a ring invariant under rotation, reversal and, with image offsets, a lattice translation."""
    seq = list(members); n = len(seq)
    if n and not np.isscalar(seq[0]): return _node_key([(int(a), tuple(int(x) for x in t)) for a, t in seq])
    return min(tuple(d[r:] + d[:r]) for d in (seq, seq[::-1]) for r in range(n))


def _ordered(found, images):
    rows = sorted(found.items(), key=lambda kv: (len(kv[0]), canonical([x[0] for x in kv[1]]), kv[0]))
    return [tuple(v) if images else tuple(x[0] for x in v) for _, v in rows]


def _primitive(dmaps, path):
    L = len(path)
    for a in range(L):
        da = dmaps[path[a][0]]; base = path[a][1]
        for b in range(a + 1, L):
            rel = tuple(path[b][1][k] - base[k] for k in range(3)); true = da.get((path[b][0], rel))
            if true is not None and true < min(b - a, L - (b - a)): return False
    return True


def primitive_rings(n_atoms, edges, offsets, max_length, bipartite=True, images=False):
    """Primitive rings (Franzblau's shortest-path rings) up to max_length atoms, each once."""
    if bipartite: _two_colour(n_atoms, edges, 'primitive_rings')
    adj = _adjacency(n_atoms, edges, offsets); dmaps = [_distances(adj, a, max_length // 2 + 1) for a in range(n_atoms)]; found = {}
    for root in range(n_atoms):
        stack = [((root, (0, 0, 0)), [(root, (0, 0, 0))])]
        while stack:
            node, path = stack.pop()
            if len(path) > max_length: continue
            for nbr, o in adj[node[0]]:
                nxt = _step(node, nbr, o)
                if nxt == (root, (0, 0, 0)):
                    if len(path) >= 3 and _primitive(dmaps, path): found.setdefault(_node_key(path), path)
                    continue
                if nbr < root or nxt in path: continue
                back = dmaps[root].get(nxt)
                if back is None or len(path) + back > max_length: continue
                if len(path) < max_length: stack.append((nxt, path + [nxt]))
    return _ordered(found, images)


def _shortest_paths(adj, source, target, cutoff, blocked_node=None, blocked_edge=None):
    """All shortest paths from source to target in the covering graph, avoiding one node or one edge."""
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


def king_rings(n_atoms, edges, offsets, max_length, bipartite=True, images=False):
    """King rings up to max_length atoms: per atom and pair of its neighbours, the shortest paths avoiding the atom."""
    if bipartite: _two_colour(n_atoms, edges, 'king_rings')
    adj = _adjacency(n_atoms, edges, offsets); found = {}
    for c in range(n_atoms):
        base = (c, (0, 0, 0)); nb = [_step(base, j, o) for j, o in adj[c]]
        for x in range(len(nb)):
            for y in range(x + 1, len(nb)):
                for p in _shortest_paths(adj, nb[x], nb[y], max_length - 2, blocked_node=base):
                    if len(p) + 1 >= 3: cyc = [base] + p; found.setdefault(_node_key(cyc), cyc)
    return _ordered(found, images)


def guttman_rings(n_atoms, edges, offsets, max_length, bipartite=True, images=False):
    """Guttman rings up to max_length atoms: per bond, the shortest paths between its atoms avoiding the bond."""
    if bipartite: _two_colour(n_atoms, edges, 'guttman_rings')
    adj = _adjacency(n_atoms, edges, offsets); found = {}
    for (i, j), o in zip(edges, offsets):
        a = (int(i), (0, 0, 0)); b = (int(j), tuple(int(x) for x in o))
        for p in _shortest_paths(adj, a, b, max_length - 1, blocked_edge={a, b}):
            if len(p) >= 3: found.setdefault(_node_key(p), p)
    return _ordered(found, images)


CRITERIA = {'primitive': primitive_rings, 'king': king_rings, 'guttman': guttman_rings}


def rings(n_atoms, edges, offsets, max_length, criterion='primitive', bipartite=True, images=False):
    """Rings under one criterion up to max_length atoms, as atom-index tuples, or as (atom, image offset) pairs when images."""
    if criterion not in CRITERIA: raise ValueError(f'rings: unknown criterion {criterion!r}; one of {sorted(CRITERIA)}')
    return CRITERIA[criterion](n_atoms, edges, offsets, max_length, bipartite, images)


def rings_converged(n_atoms, edges, offsets, criterion='primitive', bipartite=True, start=16, step=4, hard_cap=48, margin=2.0, images=False):
    """Rings and the cap at which their size histogram is confirmed, the cap raised by step from start."""
    if step < 1 or margin < 1 or start > hard_cap: raise ValueError(f'rings_converged: needs step >= 1 (got {step}), margin >= 1 (got {margin}) and start <= hard_cap (got {start}, {hard_cap})')
    if bipartite: _two_colour(n_atoms, edges, 'rings_converged')
    prev = None; cap = start; largest = 0
    while cap <= hard_cap:
        r = rings(n_atoms, edges, offsets, cap, criterion, bipartite, images); h = dict(sorted(Counter(len(x) for x in r).items())); largest = max(h, default=0)
        if h and h == prev and cap >= margin * largest: return r, cap
        if margin * largest > hard_cap: break
        prev, cap = h, cap + step
    if not largest: raise RuntimeError(f'rings_converged: no {criterion} ring up to hard_cap {hard_cap} atoms')
    raise RuntimeError(f'rings_converged: {criterion} rings reach {largest} atoms and are not confirmed up to margin {margin:g} x {largest} atoms within hard_cap {hard_cap}')


def ring_sizes(found, bipartite=True):
    """Ring size: network-former units (half the atoms) on a bipartite graph, atoms otherwise."""
    n = np.array([len(r) for r in found], int)
    if bipartite and (n % 2).any(): raise ValueError(f'ring_sizes: a ring of {int(n[n % 2 == 1][0])} atoms cannot occur on a bipartite graph; pass bipartite=False')
    return n // 2 if bipartite else n


def ring_extents(found, positions, cell):
    """Largest distance between two atoms of each ring, placed by its image offsets when it carries them."""
    pos = np.asarray(positions, float); cell = np.asarray(cell, float); out = np.empty(len(found))
    for k, ring in enumerate(found):
        if len(ring) and not np.isscalar(ring[0]): p = np.array([pos[int(a)] + np.asarray(t, float) @ cell for a, t in ring])
        else:
            p = [pos[ring[0]]]
            for a in ring[1:]: p.append(p[-1] + minimum_image(pos[a] - p[-1], cell))
            p = np.array(p)
        out[k] = np.max(np.linalg.norm(p[:, None] - p[None], axis=2))
    return out
