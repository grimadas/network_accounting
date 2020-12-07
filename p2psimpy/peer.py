from __future__ import annotations

from collections import defaultdict
import os
import random
from typing import Optional

from simpy import Store

from p2psimpy.logger import setup_logger
from p2psimpy.messages import Hello
from p2psimpy.network import Connection
from p2psimpy.services.base import BaseHandler, BaseRunner


class BandwidthCounters:

    def __init__(self) -> None:
        self.send_counters = defaultdict(float)
        self.receive_counters = defaultdict(float)

    def get_last_send_counter(self, receiver_id: str) -> int:
        return self.send_counters[receiver_id]

    def get_last_receive_counter(self, sender_id: str) -> int:
        return self.receive_counters[sender_id]

    def reset_send_counter(self, receiver_id: str) -> None:
        self.send_counters.pop(receiver_id, None)

    def reset_receive_counter(self, sender_id: str) -> None:
        self.receive_counters.pop(sender_id, None)

    def inc_send_counter(self, receiver_id: str, val: float) -> None:
        self.send_counters[receiver_id] += val

    def inc_receive_counter(self, send_id: str, val: float) -> None:
        self.receive_counters[send_id] += val


class Peer:

    def __init__(self, sim, peer_id: int, peer_type: str,
                 location: str, bandwidth_ul: float, bandwidth_dl: float, **kwargs):
        """
        Physical representation of a Peer
        :param sim: Simulation environment
        :param name:
        :param location:
        """
        self.sim = sim
        self.env = sim.env
        self.peer_type = peer_type
        self.peer_id = str(peer_id)
        self.name = str(peer_id) + ":" + str(peer_type)
        self.location = location
        self.bandwidth_ul = bandwidth_ul
        self.bandwidth_dl = bandwidth_dl
        self.__dict__.update(kwargs)

        peer_repr = repr(self)
        if sim.sim_dir:
            self.log_name = os.path.join(sim.sim_dir, peer_repr + ".log")
            self.logger = setup_logger(peer_repr, self.log_name)
        else:
            self.log_name = None
            self.logger = None

        # Message queue for the received messages
        self.msg_queue = Store(self.env)

        # Network connections
        self.online = True
        self.connections = dict()

        # Known peers
        self.disconnect_callbacks = []
        self.last_seen = dict()

        # Peer services
        self.handlers = {}  # Service.handle_message(self, msg) called on message
        self.mh_map = {}  # Message -> Handler map
        self.runners = {}  # Peer service runners
        self.mprt_map = {}  # Message -> Pre-Receive Trigger

        # Storage
        self.storage = {}
        self.counters: BandwidthCounters = BandwidthCounters()
        self.msg_type_to_watch = set()

        # Monitoring services 
        self.bytes_load = {}  # Overhead on bytes per sec
        self.msg_count_load = {}  # Msg per sec

        # Start peer as it is created
        self.env.process(self.run())

    def __repr__(self):
        return '%s_%s' % (self.__class__.__name__, self.name)

    def __lt__(self, other):
        return self.name < other.name

    def run(self):
        while True:
            # Receive message  
            msg = yield self.msg_queue.get()
            num_bytes = msg.size
            sender = msg.sender
            delay = num_bytes / self.bandwidth_dl
            yield self.env.timeout(delay)

            # Trigger pre-receive Triggers if any
            if msg.pre_task:
                val = yield self.env.process(msg.pre_task(msg, self))
            if not msg.pre_task or val:
                self.receive(msg)
            # Trigger post-receive if any

    def is_connected(self, other):
        return other in self.connections

    def bootstrap_connect(self, other):
        """
        Create ad-hoc connection and send Hello
        :param other: peer object
        """
        #
        cnx = Connection(self, other)
        cnx.send(Hello(self), connect=True)

    def connect(self, other):
        """
        Add peer to the connections and repeat the same with other peer
        :param other: peer object
        """
        if not self.is_connected(other):
            if self.logger:
                self.logger.info("%s: Connecting to %s", self.env.now, repr(other))
            self.connections[other] = Connection(self, other)
            # We create bilateral connection
            if not other.is_connected(self):
                other.connect(self)

    def disconnect(self, other):
        """
        Disconnect with previously connected peer
        :param other: peer object
        """
        if self.is_connected(other):
            if self.logger:
                self.logger.warning("%s: Breaking connection with %s, Last Seen (%s)",
                                    self.env.now, repr(other),
                                    str(self.last_seen[other.peer_id]))
            del self.connections[other]
            if other.is_connected(self):
                other.disconnect(self)
            for cb in self.disconnect_callbacks:
                cb(self, other)

    def receive(self, msg):
        """
        Receive message, will trigger handlers on the message
        :param msg:
        :return:
        """
        if self.online:
            msg_sender = msg.sender.peer_id

            # Monitor the overhead of the message size 
            now_sec = int(self.env.now / 1000)
            self.bytes_load[now_sec] = self.bytes_load.get(now_sec, 0) + msg.size
            self.msg_count_load[now_sec] = self.msg_count_load.get(now_sec, 0) + 1

            # Update peer connection data
            self.last_seen[msg_sender] = self.env.now

            if self.logger:
                self.logger.info("%s: Received message <%s> from %s", self.env.now, repr(msg), msg_sender)

            # Update the pairwise counter
            if any(isinstance(msg, t) for t in self.msg_type_to_watch):
                self.counters.inc_receive_counter(msg_sender, msg.size)

            # Find the services that should be triggered
            services = set()
            for msg_type in self.mh_map:
                if isinstance(msg, msg_type):
                    services.update(self.mh_map[msg_type])

            if not services:
                if self.logger:
                    self.logger.error("No handler for the message %s", msg)
                raise Exception("No handler for the message ", msg, repr(self))
            else:
                for service_id in services:
                    self.handlers[service_id].handle_message(msg)

    def send(self, receiver, msg):
        """
        Send to a receiver peer in a fire-and-forget fashion.
        If receiver is not connected will raise and exception
        """
        # fire and forget
        if self.online:
            if receiver not in self.connections:
                if self.logger:
                    self.logger.error("%s: Sending message to a not connected peer %s",
                                      self.env.now, repr(receiver))
                raise Exception("Not connected")
            if self.logger:
                self.logger.info("%s: Sending message <%s> to %s", self.env.now, repr(msg), receiver)
            self.connections[receiver].send(msg)

            # Update the pairwise counter
            if any(isinstance(msg, t) for t in self.msg_type_to_watch):
                self.counters.inc_send_counter(receiver.peer_id, msg.size)

    def _get_connections(self, exclude_bootstrap=True, except_set: set = None, except_type: set = None):
        if except_set is None:
            except_set = set()
        if except_type is None:
            except_type = set()
        if exclude_bootstrap:
            except_type.add('bootstrap')
        conn_set = set(self.connections.keys()) - except_set
        return (p for p in conn_set if p.peer_type not in except_type)

    def get_connected_peer(self, peer_id: str) -> Optional[Peer]:
        for p in self.connections.keys():
            if str(peer_id) == str(p.peer_id):
                return p
        return None

    def gossip(self, msg, f, exclude_bootstrap=True, except_peers: set = None, except_type: set = None):
        """
        Send to f neighbours selected randomly
        :param msg: Message
        :param f: the fanout parameter (number of peers to gossip to)
        :param exclude_bootstrap: Exclude bootstrap from gossip
        :param except_peers: connected peers to exclude from gossip
        :param except_type: exclude from gossip type of peers
        """
        if not self.online:
            return None
        gossip_set = list(self._get_connections(exclude_bootstrap, except_peers, except_type))
        selected = random.sample(gossip_set, min(f, len(gossip_set)))

        for other in selected:
            self.send(other, msg)

        return selected

    def broadcast(self, msg, exclude_bootstrap=True, except_set: set = None, except_type: set = None):
        """Send to all connected peers except given """
        for other in self._get_connections(exclude_bootstrap, except_set, except_type):
            self.send(other, msg)

    def add_service(self, service):
        """
        Add service to the peer
        """
        serv_name = type(service).__name__
        if isinstance(service, BaseHandler):
            self.handlers[serv_name] = service
            for m in service.messages:
                if m not in self.mh_map:
                    self.mh_map[m] = set()
                self.mh_map[m].add(serv_name)
        if isinstance(service, BaseRunner):
            self.runners[serv_name] = service

    def start_all_runners(self):
        for runner in self.runners.values():
            runner.start()

    def get_storage(self, storage_name):
        return self.storage.get(storage_name)

    def store(self, storage_name, msg_id, msg, rewrite: bool = False):
        if storage_name not in self.storage:
            if self.logger:
                self.logger.error("No storage %s found", storage_name)
            raise Exception("No storage {} found" % storage_name)
        self.storage[storage_name].add(msg_id, msg, rewrite)

    def add_storage(self, storage_name, storage):
        if storage_name not in self.storage:
            self.storage[storage_name] = storage
