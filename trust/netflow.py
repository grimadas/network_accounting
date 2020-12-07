#!/usr/bin/env python3

import networkx as nx


class Netflow:
    """
    This class implements the Netflow algorithm.
    """

    def __init__(self, graph: nx.Graph, seed_node: int = None, alpha: float = 2) -> None:
        self.graph = graph
        self.alpha = alpha
        self.seed_node = seed_node

        self._compute_scores()

    def _prepare(self) -> None:
        self._graph = self.graph.copy()

        for neighbour in self._graph.out_edges([self.seed_node], 'weight', 0):
            cap = self._graph.adj[self.seed_node][neighbour[1]]['weight']
            self._graph.adj[self.seed_node][neighbour[1]]['weight'] = float(cap) / float(self.alpha)

    def _initial_step(self) -> None:
        """
        In the intial step, all capactities are computed
        """

        for node in self._graph.nodes():
            self._compute_capacity(node)
        return self._graph

    def _compute_capacity(self, node: int) -> None:
        if node == self.seed_node:
            return
        contribution = nx.maximum_flow_value(self._graph, self.seed_node, node, 'weight')
        consumption = nx.maximum_flow_value(self._graph, node, self.seed_node, 'weight')

        self._graph.add_node(node, weight=max(0, contribution - consumption))
        self._graph.add_node(node, bartercast=contribution - consumption)

    def _netflow_step(self):

        compute_score = lambda node: nx.maximum_flow_value(self._graph, self.seed_node, node,
                                                           'weight') if node != self.seed_node else 0

        scores = {node: compute_score(node) for node in self._graph.nodes()}
        nx.set_node_attributes(self._graph, scores, 'score')

    def _compute_scores(self):
        self._prepare()
        self._initial_step()
        self._netflow_step()

    def compute(self, seed_node: int, target_node: int) -> float:
        if seed_node != self.seed_node:
            self.seed_node = seed_node
            self._compute_scores()
        scores = nx.get_node_attributes(self._graph, 'score')
        return scores[target_node]
