"""D-separation labels for graph_scm datasets.

Builds a networkx DAG from a sampled ``RandomDataset`` graph and answers, for each
observed feature X_j, whether X_j is d-separated from the target Y given all remaining
observed features.

Two representations are supported:

``feature_as_child=True`` (default, correct)
    Every observed feature becomes its own graph node, a child of the latent node it was
    extracted from. This is the faithful representation because a feature is a lossy view
    of its node: it reads one column slice of a node matrix that also carries latent
    dimensions no feature ever reads (``node_n_latent_features``), and children receive
    the whole parent matrix. Conditioning on a feature therefore does not determine its
    node. It also handles, for free: the ``neighbor_id`` / ``softmax_id`` categorical
    modes (where children see the pre-discretization value), several features sharing one
    node, and the same issues for Y.

``feature_as_child=False`` (naive)
    Features are identified with their nodes. Kept only for comparison -- it is wrong
    whenever a node carries latent dimensions, and it is degenerate when two features
    share a node (the query node then lies inside its own conditioning set, which
    networkx rejects outright).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import networkx as nx


def build_ci_graph(
    dag: List[List[int]],
    node_feature_specs: List[Dict[str, object]],
    feature_as_child: bool = True,
) -> Tuple[nx.DiGraph, Dict[str, object], Dict[str, str]]:
    """Build the DAG over latent nodes and observed features.

    Parameters
    ----------
    dag
        ``dag[i]`` is the list of parent node indices of node ``i`` (as returned by
        ``RandomDAG.sample`` and stored on ``Dataset``).
    node_feature_specs
        ``node_feature_specs[i]`` maps feature name -> FeatureSpec for features extracted
        at node ``i``.
    feature_as_child
        See module docstring.

    Returns
    -------
    (G, feat_nodes, groups)
        ``feat_nodes`` maps feature name -> its node id in ``G``; ``groups`` maps feature
        name -> its group ("x" or "y").
    """
    G = nx.DiGraph()
    for i in range(len(dag)):
        G.add_node(("z", i))
    for child, parents in enumerate(dag):
        for parent in parents:
            G.add_edge(("z", parent), ("z", child))

    feat_nodes: Dict[str, object] = {}
    groups: Dict[str, str] = {}
    for node_idx, specs in enumerate(node_feature_specs):
        for name, spec in specs.items():
            groups[name] = spec.group
            if feature_as_child:
                fid = ("f", name)
                G.add_node(fid)
                G.add_edge(("z", node_idx), fid)
                feat_nodes[name] = fid
            else:
                feat_nodes[name] = ("z", node_idx)
    return G, feat_nodes, groups


def ci_mask(
    G: nx.DiGraph,
    feat_nodes: Dict[str, object],
    x_names: List[str],
    y_name: str,
    condition: bool = True,
) -> Dict[str, bool]:
    """For each x feature, is it d-separated from y given the other x features?

    ``condition=False`` conditions on the empty set instead, giving *marginal*
    independence -- used to measure how much the conditioning set actually does.

    Collisions (two features sharing a graph node, only possible when
    ``feature_as_child=False``) would make the query and conditioning sets overlap, which
    networkx rejects. They are resolved here rather than raised:
    a feature colliding with the target is called dependent (the node determines y), and a
    feature colliding with another conditioning feature is called independent (it is
    treated as redundant given its co-located sibling).
    """
    y_node = feat_nodes[y_name]
    out: Dict[str, bool] = {}
    for j in x_names:
        x_node = feat_nodes[j]
        if x_node == y_node:
            out[j] = False  # same node as the target -> dependent
            continue
        z = {feat_nodes[k] for k in x_names if k != j} if condition else set()
        if x_node in z:
            out[j] = True  # redundant given a co-located sibling
            continue
        z.discard(y_node)
        out[j] = nx.is_d_separator(G, {x_node}, {y_node}, z)
    return out
