from collections import defaultdict

from ledger0.common import TRUST_STORE, WORK_GRAPH
from ledger0.graph_storage import WorkGraphStorage
from p2psimpy import BaseRunner, Storage

import networkx as nx

from trust import BoundedBarterCast, WBPPR


class ReputationStorage:

    def __init__(self, my_peer_id: str, trust_function=BoundedBarterCast, **kwargs) -> None:
        self.trust = trust_function
        self.params = kwargs
        self.seed_node = my_peer_id

        self._trust_val = None
        self.local_reputation = defaultdict(float)

    def update_graph(self, G: nx.DiGraph) -> None:
        self._trust_val = self.trust(G, **self.params)

    def get_reputation(self, target_node: str) -> float:
        return self._trust_val.compute(self.seed_node, target_node)


class ReputationService(BaseRunner):

    def __init__(self,
                 peer,
                 init_delay=2000,
                 delta_delay=2000,
                 trust_function=BoundedBarterCast,
                 **kwargs
                 ) -> None:
        """
        init_timeout: milliseconds to wait before starting the message production.
        msg_rate: number of messages per second
        """
        super().__init__(peer)

        # Work graph storage
        # Init the work graph
        self.peer.add_storage(WORK_GRAPH,
                              WorkGraphStorage())
        self.work_graph = self.peer.get_storage(WORK_GRAPH)

        self.init_timeout = init_delay
        self.delta = delta_delay

        self.reputation = ReputationStorage(self.peer.peer_id,
                                            trust_function=trust_function,
                                            **kwargs)

        self.peer.add_storage(TRUST_STORE, Storage())

    def update_reputation(self) -> None:
        # Choose min value from the edge weight
        self.reputation.update_graph(self.work_graph.min_work_graph)
        # Write to the peer store
        reps = {}
        for n in self.work_graph.min_work_graph.nodes():
            reps[n] = self.reputation.get_reputation(n)
        self.peer.store(TRUST_STORE, self.env.now, reps)

    def run(self) -> None:
        # Wait the initial timeout
        yield self.env.timeout(self.init_timeout)
        while True:
            self.update_reputation()
            yield self.env.timeout(self.delta)
