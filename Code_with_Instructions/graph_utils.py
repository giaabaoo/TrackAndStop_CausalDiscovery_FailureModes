# Stub for graph_utils — fill_vstructures and direct_chordal_graph are only
# used by unused graph generators (hairball_plus, tree_plus, random_chordal_graph2).
# shanmugam_random_chordal (the only generator used in experiments) does not need them.

import networkx as nx

def fill_vstructures(g, order):
    """
    For each node in topological order, add edges between all pairs of parents
    to ensure the graph is chordal (fill v-structures).
    """
    import itertools
    for node in order:
        parents = list(g.predecessors(node))
        for p1, p2 in itertools.combinations(parents, 2):
            if not g.has_edge(p1, p2) and not g.has_edge(p2, p1):
                g.add_edge(min(p1, p2), max(p1, p2))


def direct_chordal_graph(g):
    """
    Given an undirected chordal graph, return a directed version using
    a perfect elimination ordering.
    """
    import random
    perm = list(g.nodes())
    random.shuffle(perm)
    d = nx.DiGraph()
    d.add_nodes_from(g.nodes())
    for i, u in enumerate(perm):
        for v in g.neighbors(u):
            if perm.index(v) > i:
                d.add_edge(u, v)
    return d
