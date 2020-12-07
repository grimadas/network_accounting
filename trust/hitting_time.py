"""
The code in this file is an implementation of the Personalised Hitting Time Algorithm
used as a reputation mechanism in P2P networks.
"""
import networkx as nx

from .random_walks import RandomWalks, BiasStrategies


class PersonalizedHittingTime:
    """This class implements a Monte Carlo implementation of the hitting time algorithm
    by running random walks in a networkx graph"""

    def __init__(self, graph: nx.Graph, seed_node: int = None, number_random_walks: int = 10000,
                 reset_probability: float = 0.1) -> None:
        self.graph = graph
        self.number_random_walks = number_random_walks
        self.reset_probability = reset_probability
        self.seed_node = seed_node

        self.random_walks = RandomWalks(self.graph)
        self.random_walks.run(seed_node, int(self.number_random_walks), self.reset_probability)

    def compute(self, seed_node: int, target_node: int) -> float:
        if not self.random_walks.has_node(seed_node):
            self.random_walks.run(seed_node,
                                  int(self.number_random_walks),
                                  self.reset_probability)

        return self.random_walks.get_number_of_hits(seed_node, target_node) / self.number_random_walks


class BiasedPHT:
    """This class implements a Monte Carlo implementation of the hitting time algorithm
    by running random walks in a networkx graph"""

    def __init__(self, graph: nx.Graph, seed_node: int = None, number_random_walks: int = 10000,
                 reset_probability: float = 0.1, alpha: float = 1.0) -> None:
        self.graph = graph
        self.number_random_walks = number_random_walks
        self.reset_probability = reset_probability
        self.seed_node = seed_node

        self.random_walks = RandomWalks(self.graph, alpha)
        self.random_walks.run(seed_node, int(self.number_random_walks), self.reset_probability)

    def compute(self, seed_node: int, target_node: int) -> float:
        if not self.random_walks.has_node(seed_node):
            self.random_walks.run(seed_node,
                                  int(self.number_random_walks),
                                  self.reset_probability,
                                  bias_strategy=BiasStrategies.ALPHA_DIFF
                                  )

        return self.random_walks.get_number_of_hits(seed_node, target_node) / self.number_random_walks
