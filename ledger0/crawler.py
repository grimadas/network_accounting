from collections import defaultdict
from random import shuffle

from ledger0.common import WORK_GRAPH
from ledger0.graph_storage import WorkGraphStorage
from ledger0.ledger import LedgerTx
from p2psimpy import BaseMessage, BaseRunner, BaseHandler
from trust import RandomWalks


class CrawlRequest(BaseMessage):
    pass


class CrawlerService(BaseRunner, BaseHandler):

    def __init__(self, peer, init_timeout=2000, crawl_interval=500):
        """
        init_timeout: milliseconds to wait before starting the message production.
        crawl_interval: do crawl step once per interval
        """
        super().__init__(peer)

        # calculate tx_interval
        self.init_timeout = init_timeout

        self.tx_interval = crawl_interval
        self.counter = defaultdict(int)

        # Let's add a storage layer to store messages
        self.strg_name = 'tx_val'
        self.online = True
        self.back = False

        # Get work graph
        self.peer.add_storage(WORK_GRAPH,
                              WorkGraphStorage())
        self.work_graph = self.peer.get_storage(WORK_GRAPH)

    def crawl_step(self):
        """Make a crawl step given the local knowledge of the graph """

        # 0. Get work graph from the storage
        work_graph = self.work_graph.min_work_graph

        # 1. Take a random walk in the graph
        if not self.online:
            return None
        r = RandomWalks(work_graph)
        walk = r.run_one_walk(str(self.peer.peer_id),
                              reset_probability=0.13,
                              back_random_walk=self.back)
        self.back = not self.back

        # 2. Send crawl request to the peer
        if len(walk) > 1:
            crawl_peer = walk[1]
            topic = walk[-1]

            # Form a message
            partner = self.peer.get_connected_peer(crawl_peer)
            if partner:
                msg = CrawlRequest(self.peer, topic)
                self.peer.send(partner, msg)

    def run(self) -> None:
        # Wait the initial timeout
        yield self.env.timeout(self.init_timeout)
        while True:
            self.crawl_step()
            yield self.env.timeout(self.tx_interval)

    def handle_message(self, msg: CrawlRequest) -> None:
        # Send back the edge info regarding the peer
        crawl_topic = msg.data
        #
        edges = list(self.work_graph.G.edges(crawl_topic, data=True))
        shuffle(edges)
        # Send random 5 edges?
        for e in edges[:5]:
            msg_data = e[2].copy()
            msg_data['id'] = e[0] + '_' + e[1]
            new_msg = LedgerTx(self.peer, msg_data)
            self.peer.send(msg.sender, new_msg)

    @property
    def messages(self):
        return CrawlRequest,
