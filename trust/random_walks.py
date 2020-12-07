import networkx as nx
import numpy as np

from collections import Counter, defaultdict
import random
from typing import Callable, Optional, Dict, List

from enum import Enum


class BiasStrategies(Enum):
    ALPHA_DIFF = 1
    EDGE_WEIGHT = 2
    EDGE_WEIGHT_BOUNDED = 3


class RandomWalks:

    def __init__(self, graph: nx.DiGraph, alpha: float = 1.0, base_number_random_walks: int = 1000) -> None:
        self.alpha = alpha
        self.graph = graph

        self.random_walks = {}
        self.counters = {}
        self.num_edge_walks = defaultdict(int)
        self.hits = {}
        self.number_of_walks = {}
        self.base_number_of_random_walks = base_number_random_walks

        self.penalties = defaultdict(int)

    def bias(self, u: int, v: int, cur_weight: float = None, revert: bool = False) -> float:
        cur_weight = float('inf') if not cur_weight else cur_weight
        if revert:
            u, v = v, u
        return min(max(self.alpha * self.graph.get_edge_data(u, v, default={'weight': 0})['weight']
                       - self.graph.get_edge_data(v, u, default={'weight': 0})['weight'], 0), cur_weight)

    def weight(self, current_node: int, neigh: int, cur_weight: float = None, revert: bool = False) -> float:
        cur_weight = float('inf') if not cur_weight else cur_weight
        if revert:
            current_node, neigh = neigh, current_node
        return min(self.graph.get_edge_data(current_node, neigh, default={'weight': 0})['weight'], cur_weight)

    def edge_weight_bounded(self,
                            current_node: int,
                            neigh: int,
                            cur_weight: float = None,
                            revert: bool = False) -> float:
        val = self.base_number_of_random_walks * self.weight(current_node, neigh, cur_weight, revert)
        return 0.0 if self.num_edge_walks[(current_node, neigh)] \
                      + self.base_number_of_random_walks * int(
            self.penalties.get(neigh, 0) / self.alpha) >= val else val

    def neigh(self, x: int, back_random_walk: bool = False) -> List:
        try:
            return list(self.graph.predecessors(x)) if back_random_walk else list(self.graph.successors(x))
        except nx.NetworkXError as _:
            return []

    def _compute_random_walks(self,
                              seed_node: int,
                              weight_func: Callable[[int, int, Optional[float], Optional[bool]], float],
                              reset_probability: float,
                              back_random_walk: bool = False,
                              update_weight: bool = False
                              ) -> List:
        random_walk = [seed_node]
        cur_weight = float('inf')
        c = random.uniform(0, 1)

        while c > reset_probability and len(self.neigh(random_walk[-1], back_random_walk)) > 0:
            current_node = random_walk[-1]
            current_neighbors = self.neigh(current_node, back_random_walk)
            current_edge_weights = np.array(
                [weight_func(current_node, neighbor, cur_weight, back_random_walk) for neighbor in current_neighbors])
            cumulated_edge_weights = np.cumsum(current_edge_weights)

            if cumulated_edge_weights[-1] == 0:
                break

            random_id = list(
                cumulated_edge_weights < (random.uniform(0, 1) * cumulated_edge_weights[-1])).index(
                False)
            next_node = current_neighbors[random_id]
            if update_weight:
                cur_weight = min(current_edge_weights[random_id], cur_weight)
            self.num_edge_walks[(current_node, next_node)] += 1
            random_walk.append(next_node)

            c = random.uniform(0, 1)
        if seed_node not in self.random_walks:
            self.random_walks[seed_node] = []

        self.random_walks[seed_node].append(random_walk)
        return random_walk

    def run_one_walk(self,
                     seed_node: int,
                     reset_probability: float = 0.33,
                     back_random_walk: bool = False,
                     bias_strategy: BiasStrategies = BiasStrategies.EDGE_WEIGHT,
                     update_weight: bool = False,
                     penalties: Dict[int, int] = None) -> List:
        w_func = None
        if bias_strategy == BiasStrategies.ALPHA_DIFF:
            w_func = self.bias
        elif bias_strategy == BiasStrategies.EDGE_WEIGHT:
            w_func = self.weight
        elif bias_strategy == BiasStrategies.EDGE_WEIGHT_BOUNDED:
            w_func = self.edge_weight_bounded

        if penalties:
            self.penalties = penalties

        return self._compute_random_walks(seed_node, w_func,
                                          reset_probability, back_random_walk,
                                          update_weight)

    def run(self,
            seed_node: int,
            num_random_walks: int = 5000,
            reset_probability: float = 0.33,
            back_random_walk: bool = False,
            bias_strategy: BiasStrategies = BiasStrategies.EDGE_WEIGHT,
            update_weight: bool = False,
            penalties: Dict[int, int] = None
            ) -> None:

        self.num_edge_walks = defaultdict(int)
        self.random_walks[seed_node] = []
        for n in range(num_random_walks):
            self.run_one_walk(seed_node, reset_probability,
                              back_random_walk, bias_strategy,
                              update_weight, penalties)

        self.number_of_walks[seed_node] = num_random_walks
        self.counters[seed_node] = Counter(x for xs in self.random_walks[seed_node] for x in xs)
        self.hits[seed_node] = Counter(x for xs in self.random_walks[seed_node] for x in set(xs))

    def has_node(self, node: int) -> bool:
        return node in self.random_walks

    def get_total_hits(self, seed_node: int, target_node: int) -> float:
        if seed_node == target_node:
            return self.number_of_walks[seed_node]
        return self.counters[seed_node].get(target_node, 0)

    def get_total_sum(self, seed_node: int) -> float:
        return sum(self.counters[seed_node].values())

    def get_number_of_hits(self, seed_node: int, target_node: int) -> float:
        # number of hits
        if seed_node == target_node:
            return self.number_of_walks[seed_node]
        return self.hits[seed_node].get(target_node, 0)
