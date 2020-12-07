import networkx as nx

from .random_walks import RandomWalks, BiasStrategies


class TrustRank:
    """
    This class implements the personalized pagerank scaled by the net contribution of the seed node and deducts
    the personalized PageRank of the source node from the seed node, again scaled by its net contribution.
    """

    def __init__(self, graph: nx.Graph,
                 number_random_walks: int = 10000,
                 reset_probability: float = 0.33,
                 alpha: float = 1.0,
                 use_bias: bool = False,
                 update_weight: bool = False,
                 ) -> None:
        self.graph = graph
        self.number_random_walks = number_random_walks
        self.reset_probability = reset_probability
        self.alpha = alpha
        self.use_bias = use_bias
        self.update_weight = update_weight

        self.random_walks = RandomWalks(self.graph, alpha)
        self.reverse_walks = RandomWalks(self.graph, alpha)

    def compute(self, seed_node: int, target_node: int) -> float:

        if not self.random_walks.has_node(seed_node):
            bias_strategy = BiasStrategies.ALPHA_DIFF if self.use_bias else BiasStrategies.EDGE_WEIGHT
            self.random_walks.run(seed_node, self.number_random_walks,
                                  self.reset_probability, bias_strategy=bias_strategy, update_weight=self.update_weight)
            self.reverse_walks.run(seed_node, self.number_random_walks, self.reset_probability,
                                   back_random_walk=True,  bias_strategy=bias_strategy, update_weight=self.update_weight)

        # Process random walks with weighted PHT: number of hits of a target node
        pr1 = self.random_walks.get_number_of_hits(seed_node,
                                                   target_node) / self.number_random_walks
        pr2 = self.reverse_walks.get_number_of_hits(seed_node, target_node) / self.number_random_walks
        return pr1 - pr2
