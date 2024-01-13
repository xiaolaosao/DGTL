import os
import networkx as nx
import torch
import torch.optim as optim

from utils import *
from scipy import sparse
from loss_model import ReconLoss
from DGTL_model import DGTL_model

def train_DGTLGCN():
    #load graph
    graphs = list()
    fnames = sorted(os.listdir(DATA_DIR))
    gg = list()
    use_flag = 0
    for curr_sort in fnames:
        if use_flag in train_data:
            g = nx.read_gpickle(DATA_DIR + "/" + curr_sort)
            gg.append(g)
        use_flag += 1
    for curr_file in gg:
        graphs.append(curr_file)
    A = list()
    feature = list()
    for graph in graphs:
        adj = nx.adjacency_matrix(graph)
        A.append(adj.astype(float))
        feature.append(sparse.csr_matrix(np.identity(graph.number_of_nodes())))

    # model
    num_features_nonzero = None
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DGTL_model(num_node, emb_dim, gcn_hidden, num_features_nonzero, len(train_data) - 1, Gmodel, dropout, GAT_heads)
    model_opt = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weightdecay)
    model.to(device)
    loss_new = ReconLoss(device)

    #load edge indices and label
    indices_list = list()
    edge_label_list = list()
    fnames_indices = sorted(os.listdir(indices_DIR))
    fnames_edge_label = sorted(os.listdir(edge_label_DIR))
    for i in range(100):
        if i in test_data:
            indices_list.append(torch.load(indices_DIR + "/" + fnames_indices[i]).long().to(device))
            edge_label_list.append(torch.load(edge_label_DIR + "/" + fnames_edge_label[i]).long().to(device))

    # merge graph
    sum_graph = nx.DiGraph()
    for i, t in enumerate(train_data):
        if i == 0:
            sum_graph.add_nodes_from(graphs[i].nodes())
        sum_graph.add_edges_from(graphs[i].edges())
    #1 : 1 smapling
    sum_pos_index, sum_neg_index, _, _ = neg_smaple(sum_graph, 1)

    adj_new = list()
    for i in range(len(train_data)):
        adj_new.append(
            torch.add(torch.tensor(nx.adjacency_matrix(graphs[i]).todense()), torch.eye(num_node)).to(torch.float32))

    for i in range(len(adj_new)):
        adj_new[i] = adj_new[i].unsqueeze(2).to(device)
        if i == 0:
            adj_new_set = adj_new[i]
        else:
            adj_new_set = torch.cat((adj_new_set, adj_new[i]), 2)

    feature_list = list()
    for i in range(len(train_data)):
        feature_sparse_to_tuple = sparse_to_tuple(feature[i])

        k = torch.from_numpy(feature_sparse_to_tuple[0]).long().to(device)
        v = torch.from_numpy(feature_sparse_to_tuple[1]).to(device)
        feature_sparse_to_tuple = torch.sparse.FloatTensor(k.t(), v, feature_sparse_to_tuple[2]).to(device)
        feature_sparse_to_tuple = feature_sparse_to_tuple.to_dense().to(torch.float32)
        feature_list.append(feature_sparse_to_tuple)

    for i in range(len(feature_list)):
        feature_list[i] = feature_list[i].unsqueeze(2).to(device)
        if i == 0:
            feature_set = feature_list[i]
        else:
            feature_set = torch.cat((feature_set, feature_list[i]), 2)

    time_vector = torch.zeros(sum_pos_index.size(1), len(train_data))
    for i in range(sum_pos_index.size(1)):
        for j in range(len(train_data)):
            if A[j][sum_pos_index[0][i], sum_pos_index[1][i]] == 1:
                time_vector[i][j] = 1
    time_vector = time_vector.to(device)

    decayW = 1
    decay_length = len(train_data) - 1
    decay_vec = torch.ones(1, decay_length)
    for i in range(decay_length):
        decay_vec[0][i] = math.exp(-decayW * (decay_length - i - 1))
    for i in range(time_vector.size(0) - 1):
        if i == 0:
            decay_tensor = torch.cat((decay_vec, decay_vec), 0)
        else:
            decay_tensor = torch.cat((decay_tensor, decay_vec), 0)
    decay_tensor = decay_tensor.to(device)

    # indices
    sum_pos_index_dict = dict()
    pos_time_vector_index_list = list()
    neg_time_vector_index_list = list()
    for i in range(sum_pos_index.size(1)):
        sum_pos_index_dict[(sum_pos_index[0][i].item(), sum_pos_index[1][i].item())] = i
    for i, t in enumerate(test_data):
        pos_time_vector_index = list()
        neg_time_vector_index = list()
        half_index = int(indices_list[i][0].shape[0] / 2)
        pos_index = torch.cat([indices_list[i][0][:half_index], indices_list[i][1][:half_index]]).view(2, -1)
        neg_index = torch.cat([indices_list[i][0][half_index:], indices_list[i][1][half_index:]]).view(2, -1)
        for j in range(pos_index.size(1)):
            try:
                pos_time_vector_index.append(sum_pos_index_dict[(pos_index[0][j].item(), pos_index[1][j].item())])
            except:
                pos_time_vector_index.append(-1)
        pos_time_vector_index_list.append(pos_time_vector_index)
        for j in range(neg_index.size(1)):
            try:
                neg_time_vector_index.append(sum_pos_index_dict[(neg_index[0][j].item(), neg_index[1][j].item())])
            except:
                neg_time_vector_index.append(-1)
        neg_time_vector_index_list.append(neg_time_vector_index)


    last_pos_index, last_neg_index, _, _ = neg_smaple(graphs[-1], 1)
    last_time_vector_pos_index = list()
    for j in range(last_pos_index.size(1)):
        try:
            last_time_vector_pos_index.append(
                sum_pos_index_dict[(last_pos_index[0][j].item(), last_pos_index[1][j].item())])
        except:
            last_time_vector_pos_index.append(time_vector.size(0))
    last_time_vector_neg_index = list()
    for j in range(last_neg_index.size(1)):
        try:
            last_time_vector_neg_index.append(
                sum_pos_index_dict[(last_neg_index[0][j].item(), last_neg_index[1][j].item())])
        except:
            last_time_vector_neg_index.append(time_vector.size(0))


    sum_graph_train= nx.DiGraph()
    sum_graph_test = nx.DiGraph()
    for i, t in enumerate(train_data[:-1]):
        if i == 0:
            sum_graph_train.add_nodes_from(graphs[i].nodes())
        sum_graph_train.add_edges_from(graphs[i].edges())
    sum_graph_train_pos_index, _, _, _ = neg_smaple(sum_graph_train, 1)
    for i, t in enumerate(train_data[1:]):
        if i == 0:
            sum_graph_test.add_nodes_from(graphs[i + 1].nodes())
        sum_graph_test.add_edges_from(graphs[i + 1].edges())
    sum_graph_test_pos_index, _, _, _ = neg_smaple(sum_graph_test, 1)

    # train
    gama = 1 - eta
    best_map = 0
    best_auc = 0
    edge_index_train = sum_graph_train_pos_index.to(device)
    edge_index_test = sum_graph_test_pos_index.to(device)
    for e in range(epoch):
        model.train()
        prob, time_vector_train = model((feature_set[:, :, :-1], adj_new_set[:, :, :-1]), time_vector[: , :-1] * decay_tensor, edge_index_train)

        zero_vector = torch.zeros(1, emb_dim).float().to(device)
        time_vector_train = torch.cat((time_vector_train, zero_vector), 0)
        last_pos_time_vector = time_vector_train[last_time_vector_pos_index]
        last_neg_time_vector = time_vector_train[last_time_vector_neg_index]

        loss_cross = loss_new(prob * eta, last_pos_index, last_neg_index, time_vector_flag,
                              last_pos_time_vector * gama, last_neg_time_vector * gama)
        re_adj = torch.sigmoid(torch.matmul(prob, prob.t()))
        loss = torch.sum(torch.sigmoid(torch.sqrt(torch.matmul(re_adj, re_adj.t())).diag()))
        loss_cross = loss_cross + p * loss
        model_opt.zero_grad()
        loss_cross.backward()
        model_opt.step()


        model.eval()
        auc_list = []
        ap_list = []
        for i, t in enumerate(test_data):
            prob_test, time_vector_test = model((feature_set[:, :, 1:], adj_new_set[:, :, 1:]), time_vector[: , 1:] * decay_tensor, edge_index_test)
            zero_vector = torch.zeros(1, emb_dim).float().to(device)
            time_vector_test = torch.cat((time_vector_test, zero_vector), 0)

            half_index = int(indices_list[i][0].shape[0] / 2)
            pos_index = torch.cat([indices_list[i][0][:half_index], indices_list[i][1][:half_index]]).view(2, -1)
            neg_index = torch.cat([indices_list[i][0][half_index:], indices_list[i][1][half_index:]]).view(2, -1)
            for k in range(len(pos_time_vector_index_list[i])):
                if pos_time_vector_index_list[i][k] == -1:
                    pos_time_vector_index_list[i][k] = time_vector.size(0)
            pos_time_vector_test = time_vector_test[pos_time_vector_index_list[i]]
            for k in range(len(neg_time_vector_index_list[i])):
                if neg_time_vector_index_list[i][k] == -1:
                    neg_time_vector_index_list[i][k] = time_vector.size(0)
            neg_time_vector_test = time_vector_test[neg_time_vector_index_list[i]]

            auc, ap = loss_new.predict(prob_test * eta, pos_index, neg_index, time_vector_flag, pos_time_vector_test * gama, neg_time_vector_test * gama)
            auc_list.append(auc)
            ap_list.append(ap)
        map = np.mean(ap_list)
        mauc = np.mean(auc_list)

        if map > best_map:
            best_map = map
            best_auc = mauc
            stop = 0
        if mauc > best_auc:
            best_map = map
            best_auc = mauc
            stop = 0
        if stop > 50:
            print('stop')
            print('best_map={:.4f} best_auc={:.4f}'.format(best_map, best_auc))
            break
        print('test{}:  map={:.4f} auc={:.4f}'.format(e + 1, map, mauc))
        stop += 1



if __name__ == '__main__':
##############################Enron####################################
    setup_seed(42)
    train_data = [x for x in range(0, 8)]
    test_data = [x for x in range(8, 11)]

    DATA_DIR = './data/enron10/enron10_graphs/'
    indices_DIR = './data/enron10/enron10_indices/indices'
    edge_label_DIR = './data/enron10/enron10_indices/edges_label'

    Gmodel = 'GCN'
    GAT_heads = [2, 1]
    learning_rate = 0.002
    feat_dim = 184
    num_node = 184
    emb_dim = 16
    epoch = 1000
    gcn_hidden = [64, 16]
    dropout = [0.2, 0]
    time_vector_flag = 1
    weightdecay = 0
    eta = 0.5
    p = 0.01
#########################################################################
###############################UCI15#####################################
    # setup_seed(42)
    # train_data = [x for x in range(5, 11)]
    # test_data = [x for x in range(11, 13)]
    #
    # DATA_DIR = './data/uci15/uci_graphs/'
    # indices_DIR = './data/uci15/uci_indices_1_1/indices'
    # edge_label_DIR = './data/uci15/uci_indices_1_1/edge_label'
    #
    # Gmodel = 'GCN'
    # GAT_heads = [4, 1]
    # learning_rate = 0.005
    # feat_dim = 1899
    # num_node = 1899
    # emb_dim = 16
    # epoch = 2000
    # gcn_hidden = [512, 16]
    # dropout = [0.1, 0]
    # time_vector_flag = 1
    # weightdecay = 0
    # eta = 0.4
    # p = 0.1
#########################################################################

#############################BCA50_add###################################
    # setup_seed(42)
    # train_data = [x for x in range(0, 30)]
    # test_data = [x for x in range(49, 50)]
    #
    # DATA_DIR = './data/BC_Alpha50/BC_Alpha_graphs_add/'
    # indices_DIR = './data/BC_Alpha50/BC_Alpha_add_indices_1_1/indices'
    # edge_label_DIR = './data/BC_Alpha50/BC_Alpha_add_indices_1_1/edges_label'
    #
    # Gmodel = 'GCN'
    # learning_rate = 0.001
    # feat_dim = 3783
    # num_node = 3783
    # emb_dim = 32
    # epoch = 2000
    # gcn_hidden = [512, 32]
    # GAT_heads = [2, 1]
    # dropout = [0.2, 0]
    # time_vector_flag = 1
    # weightdecay = 0
    # eta = 0.6
    # p = 0.1
#########################################################################

    train_DGTLGCN()
