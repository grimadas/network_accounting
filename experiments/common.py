import networkx as nx
import matplotlib.pyplot as plt

from random import choice, randint

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