from collections import defaultdict

import networkx as nx

from typing import Optional, Dict


class WorkGraphStorage:

    def __init__(self):
        self.G = nx.DiGraph()

        self.min_work_graph = nx.DiGraph()

        self.fresh_coef = defaultdict(lambda: 1)

    def add(self,
            from_id: str,
            to_id: str,
            **kwargs) -> None:
        self.G.add_edge(from_id, to_id, **kwargs)
        # Update the min work graph
        weight = min(self.get(from_id, to_id).get('total_received', 0),
                     self.get(from_id, to_id).get('total_sent', 0))
        self.min_work_graph.add_edge(from_id, to_id, weight=weight)

    def get(self,
            from_id: str,
            to_id: str) -> Optional[Dict]:
        return self.G.get_edge_data(from_id, to_id)
