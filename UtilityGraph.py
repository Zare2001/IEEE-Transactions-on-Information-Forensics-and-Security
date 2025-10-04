import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def calc_incidence_nested(G):
    A_ij = {i: {} for i in G.nodes()}
    for edge in G.edges:
        i, j = edge
        if i < j:
            A_ij[i][j] = 1
            A_ij[j][i] = -1
        else:
            A_ij[i][j] = -1
            A_ij[j][i] = 1  
    return A_ij


def build_random_graph(num_nodes_fix,required_probability=0.999,fix_num_nodes=False,r_fix=None):
    # 2D
    num_nodes = int(np.ceil(np.sqrt(1 / (1 - required_probability))))
    if fix_num_nodes:
        num_nodes=num_nodes_fix
    
    if r_fix is not None:
        r_c=r_fix
    else:
        r_c = np.sqrt(np.log(2*num_nodes) / num_nodes)
    

    pos = {i: (np.random.uniform(low=0, high=100), np.random.uniform(low=0, high=100)) for i in range(num_nodes)}

    G = nx.random_geometric_graph(n=num_nodes, radius=r_c * 100, pos=pos)

    A = nx.adjacency_matrix(G).toarray()
    return num_nodes, G, A, pos,r_c