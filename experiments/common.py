import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns   
from scipy.stats import rankdata

import collections
from random import choice, randint
from typing import List

from p2psimpy.config import Config, Dist
from p2psimpy.consts import MBit

# Configuration for experiments set: 
class Locations(Config):
    locations = ['Ohio', 'Ireland', 'Tokyo']
    latencies = {
        'Ohio': {'Ohio': Dist('invgamma', (5.54090, 0.333305, 0.987249)),
                 'Ireland': Dist('norm', (73.6995, 1.19583092197097127)),
                 'Tokyo': Dist('norm', (156.00904977375566, 0.09469886668079797))
                },
        'Ireland':{'Ireland': Dist('invgamma', (6.4360455224301525, 0.8312748033308526, 1.086191852963273)),
                   'Tokyo': Dist('norm', (131.0275, 0.25834811785650774))
                  },
        'Tokyo': {'Tokyo':  Dist('invgamma', (11.104508341331055, 0.3371934865734555, 2.0258998705983737))}
    }
    
# Peer physical properties 
class PeerConfig(Config):
    location = Dist('sample', Locations.locations)
    bandwidth_ul = Dist( 'norm', (50*MBit, 10*MBit))
    bandwidth_dl = Dist( 'norm', (50*MBit, 10*MBit))



def prepare_topology(num_peers=25, frac_selfish=1, num_clients=1):    
    # Create network topology
    G = nx.erdos_renyi_graph(num_peers, 0.4)   
    nx.relabel_nodes(G, {k: k+1 for k in G.nodes()} ,copy=False)
    
    # Connect the client node to a random peer
    client_edges = [(i, choice(list(G.nodes())))
                    for i in range(num_peers+1, num_clients+num_peers+1)]
    G.add_edges_from(client_edges)

    types_map = {k: 'peer' if k < num_peers+1 else 'client' for k in G.nodes()}
    # Add some selfish nodes
    for i in range(frac_selfish):
        s_j = randint(1, num_peers)
        types_map[s_j] = 'selfish'
    # Assign a peer type to the peers 
    nx.set_node_attributes(G, types_map , 'type')
    return G

# Visualization routines 

def visualize_peer_client_network(G):
    plt.figure(figsize=(10,10))

    # Draw client/ peer network 

    master_nodes = [n for (n,ty) in \
        nx.get_node_attributes(G,'type').items() if ty == 'peer']
    client_nodes = [n for (n,ty) in \
        nx.get_node_attributes(G,'type').items() if ty == 'client']
    selfish_nodes = [n for (n,ty) in \
        nx.get_node_attributes(G,'type').items() if ty == 'selfish']
    peer_nodes = master_nodes + selfish_nodes
    
    pos = nx.kamada_kawai_layout(G)

    nx.draw_networkx_nodes(G, pos, nodelist=master_nodes,
                           node_color='blue', node_shape='o',
                           node_size=500)
    nx.draw_networkx_nodes(G, pos, nodelist=selfish_nodes,
                           node_color='red', node_shape='o',
                           node_size=500, label=1)
    nx.draw_networkx_nodes(G, pos, nodelist=client_nodes,
                           node_color='green', node_shape='^',
                           node_size=100, label=1)
    
    nx.draw_networkx_labels(G, pos, labels={k:k for k in peer_nodes},
                            font_color='w')

    nx.draw_networkx_edges(G, pos,
                           edgelist=G.subgraph(peer_nodes).edges(),
                           width=1.5)
    nx.draw_networkx_edges(G, pos,
                           edgelist=G.edges(nbunch=client_nodes),
                           style='dotted')

def draw_distribution(dist: list) -> None:
    degreeCount = collections.Counter(dist)
    deg, cnt = zip(*degreeCount.items())

    print(np.var(dist), np.median(dist))
    fig, ax = plt.subplots()
    plt.bar(deg, cnt, color="b")

    plt.title("Degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree");
    
def draw_degree_dist(G: nx.Graph) -> None:
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
    draw_distribution(degree_sequence)   

def draw_rank_vals(x, r, title, log_val = 'symlog', ylim = (None, None), xlim=(None, None)) -> None:
    g = sns.lineplot(list(r.values()), list(x.values()))
    g.axhline(0, ls='--')

    g.set_xscale('log')
    g.set_yscale(log_val)
    g.set(ylim=ylim, xlim=xlim)
    g.set(ylabel='Contribution value', xlabel='Peer ranking', title=title);
    
    return g

def rank_vals(x_val):
    return dict(zip(x_val.keys(), rankdata([-i for i in x_val.values()], method='min')))  


# Generate work network topology
def generate_work_graph_power_law(N: int, **kwargs):
    alpha = kwargs.get('alpha', 0.4)
    beta = kwargs.get('beta', 0.5)
    gamma = kwargs.get('gamma', 0.1)
    delta_in = kwargs.get('delta_in', 0.2)
    delta_out = kwargs.get('delta_in', 0)
    seed = kwargs.get('seed', None)

    G =  nx.scale_free_graph(N*3, alpha=alpha, beta=beta, gamma=gamma, 
                             delta_in=delta_in, delta_out=delta_out, seed=seed)
    G1 = nx.DiGraph()
    for u,v,data in G.edges(data=True):
        if u != v:
            if u > N-1:
                u = u % N
            if v > N-1:
                v = v % N
            w = data['weight'] if 'weight' in data else 1.0
            if G1.has_edge(u,v):
                G1[u][v]['weight'] += w
            else:
                G1.add_edge(u, v, weight=w)
    return G1

def generate_work_graph_uniform(N: int, p: float = 0.1, rand_dist: Dist = 'uniform', rand_params: List[float] = [1, 14]):
    G1 = nx.gnp_random_graph(N, p, directed=True)
    d = Dist(rand_dist, rand_params)
    for (u, v) in G1.edges():
        G1.edges[u,v]['weight'] = d.get()
    return G1
 