#!/usr/bin/env python3

import networkx as nx


class PersonalizedPageRank:
    """
    This class implements the personalized pagerank
    """

    def __init__(self, graph: nx.Graph, seed_node: int = 0, seed_weight: float = 1.0, **kwargs) -> None:
        self.graph = graph
        self.seed_node = seed_node
        self.seed_weight = seed_weight

        self.params = kwargs
        self._recompute_pagerank()

    def _recompute_pagerank(self) -> float:
        self.rank = nx.pagerank(self.graph, personalization={self.seed_node: self.seed_weight}, **self.params)

    def compute(self, seed_node: int, target_node: int) -> float:
        """Compute personal pagerank from seed_node to target_node"""
        if self.seed_node != seed_node:
            self.seed_node = seed_node
            self._recompute_pagerank()
        return self.rank[target_node]
