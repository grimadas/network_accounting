from typing import Dict, Tuple

from ledger0.common import WORK_GRAPH
from ledger0.graph_storage import WorkGraphStorage
from p2psimpy import Peer
from p2psimpy.messages import BaseMessage
from p2psimpy.services.base import BaseHandler, BaseRunner

"""
The ledger that will finalize and update the peer counters. 
The are two run strategies:
1. Run it periodically every delta time
2. Run finalization every k packets.   
"""


class LedgerTx(BaseMessage):
    """Pairwise Transaction: an atom of the Ledger """
    pass


class LedgerNetterWorker(BaseRunner, BaseHandler):

    def __init__(self, peer: Peer,
                 init_timeout=1000,
                 msg_timeout=1000,
                 threshold_val=1000,
                 eps_diff=0.03) -> None:
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

        # calculate tx_interval
        self.init_timeout = init_timeout
        self.tx_interval = msg_timeout

        # Min threshold to consider
        self.threshold_val = threshold_val  # at least one mb packet in a transaction

        self.eps_diff = eps_diff  # diff in percent

    def get_last_counters(self, s_id: str, r_id: str) -> Tuple[float, int, float, int]:
        last_msg = self.work_graph.get(s_id, r_id)
        last_total_sent = 0.0 if not last_msg else last_msg.get('total_sent', 0.0)
        last_s_counter = 0 if not last_msg else last_msg.get('s_counter', 0)
        last_total_received = 0.0 if not last_msg else last_msg.get('total_received', 0.0)
        last_r_counter = 0.0 if not last_msg else last_msg.get('r_counter', 0)
        return last_total_sent, last_s_counter, last_total_received, last_r_counter

    def process_local_counters(self) -> None:
        for r_id, val in self.peer.counters.send_counters.items():
            # Get last finalized counter
            own_id = str(self.peer.peer_id)
            l_sent, s_counter, l_received, r_counter = self.get_last_counters(own_id, r_id)
            if val >= self.threshold_val + l_sent:
                peer_object = self.peer.get_connected_peer(r_id)
                if not peer_object:
                    self.logger.error('Peer is not connected!')
                    break
                interaction_id = own_id + '_' + r_id

                msg_data = {'id': interaction_id,
                            'total_sent': val,
                            's_counter': s_counter + 1,
                            'sign1': True}

                msg = LedgerTx(self.peer, msg_data)
                self.peer.send(peer_object, msg)

                self.store_tx_data(own_id, r_id, msg_data)

    def run(self):
        # Wait the initial timeout
        yield self.env.timeout(self.init_timeout)
        while True:
            self.process_local_counters()
            yield self.env.timeout(self.tx_interval)

    @property
    def messages(self):
        return LedgerTx,

    def verify_tx_up_to_date(self,
                             msg_val: float,
                             msg_counter: int,
                             last_val: float,
                             last_counter: int,
                             last_msg_data: Dict,
                             msg_sender: Peer) -> bool:
        # Verify the message:
        # 1. Check if you have newer version
        # 2. Check that counter are growing only
        if last_counter > msg_counter or last_val > msg_val:
            # I have newer transaction. Resend it to the peer
            new_msg = LedgerTx(self.peer, last_msg_data)
            self.peer.send(msg_sender, new_msg)
            return False
        return True

    def verify_tx(self, msg: LedgerTx) -> bool:
        msg_id = msg.data['id']
        s_id, r_id = msg_id.split('_')

        s_val, s_counter, r_val, r_counter = self.get_last_counters(s_id, r_id)
        last_msg_data = self.work_graph.get(s_id, r_id)
        last_msg_data = {} if not last_msg_data else last_msg_data
        msg_sender = msg.sender

        if 'total_sent' in msg.data:
            # A message from the sender
            new_msg_data = {'id': msg_id,
                            'total_sent': last_msg_data.get('total_sent', 0),
                            's_counter': last_msg_data.get('s_counter', 0),
                            'sign1': last_msg_data.get('sign1', False)
                            }
            res = self.verify_tx_up_to_date(msg.data['total_sent'],
                                            msg.data['s_counter'],
                                            s_val,
                                            s_counter,
                                            new_msg_data,
                                            msg_sender)
        elif 'total_received' in msg.data:
            new_msg_data = {'id': msg_id,
                            'total_received': last_msg_data.get('total_received', 0),
                            'r_counter': last_msg_data.get('r_counter', 0),
                            'sign2': last_msg_data.get('sign2', False)
                            }
            res = self.verify_tx_up_to_date(msg.data['total_received'],
                                            msg.data['r_counter'],
                                            r_val,
                                            r_counter,
                                            new_msg_data,
                                            msg_sender)
        else:
            # Ignore the message
            res = False
        return res

    def store_tx_data(self, s_id: str, r_id: str, msg_data: Dict) -> None:
        self.peer.storage[WORK_GRAPH].add(s_id, r_id, **msg_data)

    def react_on_request(self, s_id: str, r_id: str) -> None:
        new_val = self.peer.counters.get_last_receive_counter(s_id)
        r_count = self.work_graph.get(s_id, r_id).get('r_counter', 0) + 1
        assert new_val >= self.work_graph.get(s_id, r_id).get('total_received', 0.0)
        peer_object = self.peer.get_connected_peer(s_id)
        if not peer_object:
            # There is not peer connected
            self.logger.error('Peer is not connected!', s_id)
            return

        response_msg = {'id': s_id+'_'+r_id,
                        'total_received': new_val,
                        'r_counter': r_count,
                        'sign2': True}

        msg = LedgerTx(self.peer, response_msg)
        self.peer.send(peer_object, msg)
        self.store_tx_data(s_id, r_id, response_msg)

    def handle_message(self, msg: LedgerTx) -> None:
        # Handle LedgerTx
        s_id, r_id = msg.data['id'].split('_')

        if self.verify_tx(msg):
            self.store_tx_data(s_id, r_id, msg.data)
            if str(self.peer.peer_id) == r_id:
                # React on the request from the edge update
                self.react_on_request(s_id, r_id)
