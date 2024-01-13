import torch
import numpy as np
import random
import math
import scipy.sparse as sp
import networkx as nx
import os

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16: 8'
    # torch.use_deterministic_algorithms(True)

def neg_smaple(g, smaple_rate):
    g_adj = nx.adjacency_matrix(g)
    g_adj = sparse_to_tuple(g_adj)
    idx, vals, shape = g_adj[0], g_adj[1], g_adj[2]
    num_edge = g.number_of_edges()
    neg_edge_num = num_edge * smaple_rate
    num_node = g.number_of_nodes()

    idx_from = []
    idx_to = []
    edge_id = []
    for i in range(num_edge):
        edge_id.append(num_node * idx[i][0] + idx[i][1])
        idx_from.append(idx[i][0])
        idx_to.append(idx[i][1])

    from_id = np.random.choice(idx_from, size=neg_edge_num * 4, replace=True)
    to_id = np.random.choice(idx_to, size=neg_edge_num * 4, replace=True)
    neg_edge = np.stack([from_id, to_id])
    neg_edge_ids = neg_edge[0] * num_node + neg_edge[1]

    out_ids = set()
    edge_id = set(edge_id)
    num_sampled = 0
    sampled_indices = []
    for i in range(neg_edge_num * 4):
        n_eid = neg_edge_ids[i]
        if n_eid in out_ids or neg_edge[0, i] == neg_edge[1, i] or n_eid in edge_id:
            continue
        out_ids.add(n_eid)
        sampled_indices.append(i)
        num_sampled += 1
        if num_sampled >= neg_edge_num:
            break
    neg_edge = neg_edge[:, sampled_indices]
    neg_edge = torch.tensor(neg_edge, dtype=torch.long)

    edge_label = torch.ones(num_edge, dtype=torch.int)
    node_indices = torch.LongTensor(2, num_edge)
    edge_flag = 0
    for edge in g.edges():
        node_indices[0, edge_flag] = edge[0]
        node_indices[1, edge_flag] = edge[1]
        edge_flag += 1
    neg_edge_label = torch.zeros(num_sampled, dtype=torch.int)

    return node_indices, neg_edge, edge_label, neg_edge_label

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def preprocess_adj(adj):
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)) # D
    d_inv_sqrt = np.power(rowsum, -0.5).flatten() # D^-0.5
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt) # D^-0.5
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo() # D^-0.5AD^-0.5