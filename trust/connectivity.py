"""
The code in this file is an implementation of the BarterCast Algorithm
used as a reputation mechanism in P2P networks.
"""
import numpy as np
import networkx as nx
from networkx.algorithms.flow import edmonds_karp, shortest_augmenting_path


def build_auxiliary(G: nx.Graph, weighted: bool = True):
    directed = G.is_directed()

    mapping = {}
    H = nx.DiGraph()

    for i, node in enumerate(G):
        mapping[node] = i
        H.add_node(f"{i}A", id=node)
        H.add_node(f"{i}B", id=node)
        H.add_edge(f"{i}A", f"{i}B", capacity=1)

    edges = []
    for (source, target, data) in G.edges(data=True):
        w = data['weight'] if weighted else 1
        H.add_edge(f"{mapping[source]}B", f"{mapping[target]}A", capacity=w)
        if not directed:
            H.add_edge(f"{mapping[target]}B", f"{mapping[source]}A", capacity=w)

    # Store mapping as graph attribute
    H.graph["mapping"] = mapping
    return H


class LocalConnectivity:

    def __init__(self, graph: nx.Graph, weighted: bool = True) -> None:

        self.graph = graph

        self.aux = build_auxiliary(self.graph, weighted)

    def compute(self, seed_node: int, target_node: int) -> float:
        cutoff = None
        flow_func = edmonds_karp
        mapping = self.aux.graph.get("mapping", None)
        if mapping is None:
            raise nx.NetworkXError("Invalid auxiliary digraph.")

        kwargs = dict(flow_func=flow_func, residual=None, capacity='capacity')
        if flow_func is shortest_augmenting_path:
            kwargs["cutoff"] = cutoff
            kwargs["two_phase"] = True
        elif flow_func is edmonds_karp:
            kwargs["cutoff"] = cutoff

        return nx.maximum_flow_value(self.aux, f"{mapping[seed_node]}B", f"{mapping[target_node]}A", **kwargs)
