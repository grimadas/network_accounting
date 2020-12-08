"""
The code in this file is an implementation of the BarterCast Algorithm
used as a reputation mechanism in P2P networks.
"""
import networkx as nx
import numpy as np

import igraph as ig

from typing import Union


class BarterCast:

    def __init__(self, graph: Union[nx.Graph, ig.Graph], use_igraph: bool = False) -> None:
        self.graph = graph

        self.use_igraph = use_igraph

    def compute(self, seed_node: int, target_node: int) -> float:
        """Compute bartercast score of target node from perspective of seed_node"""
        if seed_node == target_node:
            return 1.0
        if self.use_igraph:
            maxflow_seed_target = self.graph.maxflow_value(seed_node, target_node, capacity='weight')
            maxflow_target_seed = self.graph.maxflow_value(target_node, seed_node, capacity='weight')
        else:
            maxflow_seed_target = nx.maximum_flow(self.graph, seed_node, target_node, capacity='weight')[0]
            maxflow_target_seed = nx.maximum_flow(self.graph, target_node, seed_node, capacity='weight')[0]
        values = float(np.arctan(maxflow_seed_target - maxflow_target_seed)) / float(0.5 * np.pi)
        return values


class MaxFlow:

    def __init__(self, graph: nx.DiGraph, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.graph = graph

    def compute(self, seed_node: int, target_node: int) -> float:
        """Compute bartercast score of target node from perspective of seed_node"""
        if seed_node == target_node:
            return 1.0
        maxflow_seed_target = nx.maximum_flow_value(self.graph, seed_node, target_node, capacity='weight')
        return maxflow_seed_target * self.alpha


class RawBarterCast:

    def __init__(self, graph: nx.DiGraph, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.graph = graph

    def compute(self, seed_node: int, target_node: int) -> float:
        """Compute bartercast score of target node from perspective of seed_node"""
        if seed_node == target_node:
            return 1.0
        maxflow_seed_target = nx.maximum_flow(self.graph, seed_node, target_node, capacity='weight')[0]
        maxflow_target_seed = nx.maximum_flow(self.graph, target_node, seed_node, capacity='weight')[0]
        values = float(maxflow_seed_target - maxflow_target_seed)
        return values


class BoundedBarterCast:

    def __init__(self, graph: Union[nx.DiGraph, ig.Graph], alpha: float = 1.0, use_igraph: bool = False) -> None:
        self.alpha = alpha
        self.graph = graph

        self.use_igraph = use_igraph

        self.scores = {}

    def net_contrib(self, node: int) -> float:

        if self.use_igraph:
            out_deg = self.graph.strength(node, mode='OUT', weights='weight')
            in_deg = self.graph.strength(node, mode='IN', weights='weight')
        else:
            out_deg = self.graph.out_degree(node, weight='weight')
            in_deg = self.graph.in_degree(node, weight='weight')

        return min(self.alpha * (out_deg + 1) - in_deg, 1000)

    def _calc(self, seed_node: int, target_node: int) -> None:
        if self.use_igraph:
            val = self.graph.maxflow_value(seed_node, target_node, capacity='weight')
        else:
            val = nx.maximum_flow_value(self.graph, seed_node, target_node, capacity='weight')
        self.scores[seed_node][target_node] = val

    def calc(self, seed_node: int, target_node: int) -> float:
        if seed_node not in self.scores:
            self.scores[seed_node] = {}
            self._calc(seed_node, target_node)
        if target_node not in self.scores[seed_node]:
            self._calc(seed_node, target_node)
        return self.scores[seed_node][target_node]

    def compute(self, seed_node: int, target_node: int) -> float:
        """Compute bartercast score of target node from perspective of seed_node"""
        if seed_node == target_node:
            return 1.0
        p1 = 0.0
        coef = self.net_contrib(seed_node)
        if coef > 0:
            p1 = coef * self.calc(seed_node, target_node)

        p2 = 0.0
        coef = self.net_contrib(target_node)
        if coef > 0:
            p2 = coef * self.calc(target_node, seed_node)
        values = float(p1 - p2)
        return values


class PenaltyCast:

    def __init__(self, graph: nx.DiGraph, alpha: float = 2.0) -> None:
        self.graph = graph

        self.scores = {}
        self.path_counts = {}

        self.aux_scores = {}

        self.penalites = {}
        self.alpha = alpha

        self.auxes = {}

    def _calc(self, seed_node: int, target_node: int) -> None:
        val = nx.maximum_flow(self.graph, seed_node, target_node, capacity='weight')
        self.scores[seed_node][target_node] = val[0]

        raw_count = {k: sum(v.values()) for k, v in val[1].items()}
        total_sum = sum(raw_count.values()) - raw_count[seed_node]
        norm_count = {k: v / total_sum for k, v in raw_count.items() if v > 0 and k != seed_node}
        self.path_counts[seed_node][target_node] = norm_count

    def calc(self, seed_node: int, target_node: int) -> float:
        if seed_node not in self.scores:
            self.scores[seed_node] = {}
            self.path_counts[seed_node] = {}
            self._calc(seed_node, target_node)
        if target_node not in self.scores[seed_node]:
            self._calc(seed_node, target_node)
        return self.scores[seed_node][target_node]

    # build aux graph
    def aux_graph(self, seed_node: int) -> float:
        if seed_node in self.auxes:
            return self.auxes[seed_node]

        penalties = {}
        for k in self.graph.pred[seed_node]:
            self.calc(seed_node, target_node=k)
            w = self.graph[k][seed_node]['weight']

            for i, v in self.path_counts[seed_node][k].items():
                if i not in penalties:
                    penalties[i] = 0
                penalties[i] += v * w / self.alpha

        self.auxes[seed_node] = self.graph.copy()
        for k, v in penalties.items():
            if k in self.auxes[seed_node][seed_node]:
                self.auxes[seed_node][seed_node][k]['weight'] -= v
        return self.auxes[seed_node]

    def _aux_calc(self, seed_node: int, target_node: int) -> None:
        val = nx.maximum_flow_value(self.auxes[seed_node], seed_node, target_node, capacity='weight')
        self.aux_scores[seed_node][target_node] = val

    def recalc_penalites(self, seed_node: int, neigh_node: int) -> float:
        self._calc(seed_node, target_node=neigh_node)
        w = self.graph[neigh_node][seed_node]['weight']

        self.aux_graph(seed_node)
        for i, v in self.path_counts[seed_node][neigh_node].items():
            if i in self.auxes[seed_node][seed_node]:
                self.auxes[seed_node][seed_node][i]['weight'] -= v * w / self.alpha

    def aux_calc(self, seed_node: int, target_node: int) -> float:
        self.aux_graph(seed_node)
        if seed_node not in self.aux_scores:
            self.aux_scores[seed_node] = {}
            self._aux_calc(seed_node, target_node)
        if target_node not in self.aux_scores[seed_node]:
            self._aux_calc(seed_node, target_node)
        return self.aux_scores[seed_node][target_node]

    def compute(self, seed_node: int, target_node: int) -> float:
        """Compute bartercast score of target node from perspective of seed_node"""
        if seed_node == target_node:
            return 1.0
        maxflow_target_seed = self.calc(target_node, seed_node)
        maxflow_seed_target = self.aux_calc(seed_node, target_node)
        values = maxflow_seed_target - maxflow_target_seed
        return values
