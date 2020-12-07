from p2psimpy import GossipService, GossipMessage, Peer, MessageProducer

# 1. Gossip Handler: exchange the data through gossip network. Simulate the Bitcoin gossip protocol
# 2. DHT use-case, or IPFs. Build a search index and process the workload.

"""
Wire Protocol used in Ethereum: 
1. Establish connection and send status message. 
2. Three high-level tasks: Chain Sync, Block propagate, Tx exchange. They are disjoint
3. Single message is limited by 16 MiB -> Reaction to that. Disconnection   
"""

"""
Possible gossip protocols: 
1. Simple push based gossip (Blind)
2. Push based gossip with rumor-spreading (remove the transactions when it is not hot)
"""

""" Additional service reacts """


class AccountableGossip(GossipService):

    def __init__(self, peer: Peer, fanout=3, exclude_peers: set = None, exclude_types: set = None):
        super().__init__(peer, fanout, exclude_peers, exclude_types)
        peer.msg_type_to_watch.add(GossipMessage)


class AccountableMessageProducer(MessageProducer):

    def __init__(self, peer, init_timeout=1000,
                 msg_rate=5, init_fanout=5,
                 init_ttl=4, pre_task=None, post_task=None):
        super().__init__(peer, init_timeout, msg_rate, init_fanout, init_ttl, pre_task, post_task)
        peer.msg_type_to_watch.add(GossipMessage)
