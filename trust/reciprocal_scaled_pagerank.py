"""
The code in this file is an implementation of the Personalised PageRank algorithm, whereby we scaled the random
walks by the net contribution of the seed node. Additionally we compute the scaled personalised PagRanks of
the seed node starting at the target node. The reputation scores produced by this reputation mechanism are given
by the difference of the two.
"""
from collections import defaultdict
from math import ceil

import networkx as nx

from .random_walks import BiasStrategies, RandomWalks


class ReciprocalScaledPageRank:
    """
    This class implements the personalized pagerank scaled by the net contribution of the seed node and deducts
    the personalized PageRank of the source node from the seed node, again scaled by its net contribution.
    """

    def __init__(self, graph: nx.DiGraph,
                 base_number_of_walks: int = 10,
                 reset_probability: float = 0.1,
                 alpha: float = 1.0
                 ) -> None:
        self.graph = graph
        self.number_random_walks = base_number_of_walks
        self.reset_probability = reset_probability
        self.alpha = alpha

        self.random_walks = RandomWalks(self.graph, alpha, base_number_of_walks)

    def net_contrib(self, node: int) -> float:
        return min(self.graph.out_degree(node, weight='weight'), 1000)

    def compute(self, seed_node: int, target_node: int) -> float:
        if not self.random_walks.has_node(seed_node):
            self.random_walks.run(seed_node,
                                  int(self.number_random_walks * self.net_contrib(seed_node)),
                                  self.reset_probability)

        if not self.random_walks.has_node(target_node):
            self.random_walks.run(target_node,
                                  int(self.number_random_walks * self.net_contrib(target_node)),
                                  self.reset_probability)

        pr1 = self.random_walks.get_total_hits(seed_node, target_node)
        pr2 = self.random_walks.get_total_hits(target_node, seed_node)
        return pr1 / self.number_random_walks - pr2 / self.number_random_walks


class WBPPR(ReciprocalScaledPageRank):

    def net_contrib(self, node: int) -> float:
        return min(self.alpha * (self.graph.out_degree(node, weight='weight') + 1)
                   - self.graph.in_degree(node, weight='weight'), 1000)


class SBPPR:

    def __init__(self, graph: nx.DiGraph,
                 base_number_random_walks: int = 10,
                 reset_probability: float = 0.1,
                 alpha: float = 1.0,
                 self_manage_penalties: bool = False) -> None:

        self.graph = graph
        self.number_random_walks = base_number_random_walks
        self.reset_probability = reset_probability

        self.penalties = {}
        self.self_manage_penalties = self_manage_penalties
        self.alpha = alpha

        self.random_walks = RandomWalks(self.graph, alpha, self.number_random_walks)

    def net_contrib(self, node: int) -> float:
        return min(self.alpha * (self.graph.out_degree(node, weight='weight') + 1), 1000)

    def calculate_penalty(self, s, t, value):
        encounters = defaultdict(int)
        penalties = defaultdict(int)
        for k in self.random_walks.random_walks[s]:
            indexes = [i for i, x in enumerate(k) if x == t]
            if indexes:
                start_val = 1
                for v in indexes:
                    for p in range(start_val, v + 1):
                        encounters[k[p]] += 1
        # calculate fractions
        for k, v in encounters.items():
            penalties[k] += ceil(value * v / encounters[t])
        return penalties

    def add_penalties(self, seed_node, penalties):
        self.penalties[seed_node] = penalties

    def get_penalties(self, seed_node):
        if seed_node not in self.penalties:
            self.penalties[seed_node] = defaultdict(int)
        for e, _, w in self.graph.in_edges(seed_node, data=True):
            pen = self.calculate_penalty(seed_node, e, w['weight'])
            for k, v in pen.items():
                self.penalties[seed_node][k] += v
        return self.penalties[seed_node]

    def compute(self, seed_node: int, target_node: int) -> float:
        if not self.random_walks.has_node(seed_node):
            self.random_walks.run(seed_node,
                                  int(self.number_random_walks * self.net_contrib(seed_node)),
                                  self.reset_probability,
                                  bias_strategy=BiasStrategies.EDGE_WEIGHT_BOUNDED,
                                  penalties=self.penalties.get(seed_node, None),
                                  update_weight=False
                                  )

            if self.self_manage_penalties and seed_node not in self.penalties:
                self.get_penalties(seed_node)
                self.random_walks.run(seed_node,
                                      int(self.number_random_walks * self.net_contrib(seed_node)),
                                      self.reset_probability,
                                      bias_strategy=BiasStrategies.EDGE_WEIGHT_BOUNDED,
                                      penalties=self.penalties.get(seed_node, None),
                                      update_weight=False
                                      )

        if not self.random_walks.has_node(target_node):
            self.random_walks.run(target_node,
                                  int(self.number_random_walks * self.net_contrib(target_node)),
                                  self.reset_probability,
                                  bias_strategy=BiasStrategies.EDGE_WEIGHT_BOUNDED,
                                  penalties=None,
                                  update_weight=False
                                  )

        penalty = 0.0 if not self.penalties.get(seed_node) else self.penalties[seed_node].get(target_node, 0)
        pr1 = self.random_walks.get_total_hits(seed_node, target_node) - penalty / self.alpha
        pr2 = self.random_walks.get_total_hits(target_node, seed_node)
        return pr1 - pr2
