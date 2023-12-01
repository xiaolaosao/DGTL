import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import SAGEConv, GATConv


class DGTL_model(nn.Module):
    def __init__(self, input_dim, output_dim, hidden, num_features_nonzero, train_len, Gmodel, dropout, GATHead):
        super(DGTL_model, self).__init__()
        assert hidden[-1] == output_dim
        self.num_layer = len(hidden)
        self.hidden = [input_dim] + hidden
        self.sft = nn.Softmax(dim=1)
        self.dropout = dropout
        self.train_len = train_len
        self.layers = nn.Sequential()
        self.Gmodel = Gmodel
        if self.Gmodel == 'GCN':
            for i in range(self.num_layer):
                self.layers.add_module('Conv' + str(i),
                                       GraphConvolution(self.hidden[i], self.hidden[i + 1], num_features_nonzero,
                                                        activation=F.leaky_relu, dropout=self.dropout[i],
                                                        is_sparse_inputs=False))
        elif self.Gmodel =='SAGE':
            for i in range(self.num_layer):
                self.layers.add_module('SAGE' + str(i),
                                       SAGEConv(self.hidden[i], self.hidden[i + 1]))
        elif self.Gmodel == 'GAT':
            for i in range(self.num_layer):
                if i == 0:
                    self.layers.add_module('GAT' + str(i),
                                           GATConv(self.hidden[i], self.hidden[i + 1], heads=GATHead[i]))
                else:
                    self.layers.add_module('GAT' + str(i),
                                           GATConv(self.hidden[i] * GATHead[i - 1], self.hidden[i + 1], heads=GATHead[i]))
        self.time_liner = nn.Linear(train_len, output_dim)
        self.relu = nn.ReLU()
        self.time_liner_adj = nn.Linear(train_len, 1)
        self.w_adj_weight = nn.Parameter(torch.ones(train_len, 1))

    def forward(self, inputs, time_vector, edge_index=None):
        x, support = inputs
        support = torch.matmul(support, self.w_adj_weight)
        support = support.squeeze(dim=2)
        rowsum = support.sum(1)
        x = torch.matmul(x, self.w_adj_weight)/self.train_len
        x = x.squeeze(dim=2)

        if self.Gmodel == 'GCN':
            d_inv_sqrt = torch.pow(rowsum, -0.5)
            d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
            d_inv_sqrt = torch.diag(d_inv_sqrt)
            support = torch.matmul(torch.matmul(d_inv_sqrt, support), d_inv_sqrt)
            out = self.layers((x, support))
        elif self.Gmodel == 'SAGE' or self.Gmodel == 'GAT':
            for i in range(self.num_layer):
                if i == 0:
                    out = self.layers[i](x, edge_index)
                else:
                    out = self.layers[i](out, edge_index)
                out = F.relu(out)
                # out = F.leaky_relu(out)
                out = F.dropout(out, p=self.dropout[i], training=self.training)

        time_vector = self.time_liner(time_vector)
        time_vector = self.relu(time_vector)
        if self.Gmodel == 'GCN':
            prob = self.sft(out[0])
        elif self.Gmodel == 'SAGE' or self.Gmodel == 'GAT':
            prob = self.sft(out)

        return prob, time_vector


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, num_features_nonzero,
                 is_sparse_inputs=False,
                 bias=False,
                 activation=F.leaky_relu,
                 dropout=0.0,
                 featureless=False):
        super(GraphConvolution, self).__init__()

        self.bias = bias
        self.activation = activation
        self.is_sparse_inputs = is_sparse_inputs
        self.featureless = featureless
        self.dropout = dropout
        self.num_features_nonzero = num_features_nonzero
        self.weight = glorot_init(input_dim, output_dim)
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, inputs):
        x, support = inputs
        # convolve
        if not self.featureless:  # if it has features x
            if self.is_sparse_inputs:
                xw = torch.sparse.mm(x, self.weight)
            else:
                xw = torch.mm(x, self.weight)
        else:
            xw = self.weight
        out = torch.mm(support, xw)
        if self.bias is not None:
            out += self.bias
        out = self.activation(out)
        out = F.dropout(out, p=self.dropout, training=self.training)

        return out, support

def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return nn.Parameter(initial)