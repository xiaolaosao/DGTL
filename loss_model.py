import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score

EPS = 1e-15

class ReconLoss(nn.Module):
    def __init__(self, device):
        super(ReconLoss, self).__init__()
        self.device = device

    def decoder(self, z, edge_index, time_vector_flag, time_vector, sigmoid=True):
        if time_vector_flag == 1:
            s_edge_index = torch.cat((z[edge_index[0]], time_vector), 1)
            t_edge_index = torch.cat((z[edge_index[1]], time_vector), 1)
        elif time_vector_flag == 2:
            s_edge_index = z[edge_index[0]] + time_vector
            t_edge_index = z[edge_index[1]] + time_vector
        else:
            s_edge_index = z[edge_index[0]]
            t_edge_index = z[edge_index[1]]
        value = (s_edge_index * t_edge_index).sum(dim=1)
        return torch.sigmoid(value)


    def forward(self, z, pos_edge_index, neg_edge_index, time_vector_flag, pos_time_vector, neg_time_vector):
        decoder = self.decoder
        pos_loss = -torch.log(
            decoder(z, pos_edge_index, time_vector_flag, pos_time_vector) + EPS).mean()
        neg_loss = -torch.log(1 - decoder(z, neg_edge_index, time_vector_flag, neg_time_vector) + EPS).mean()

        return pos_loss + neg_loss


    def predict(self, z, pos_edge_index, neg_edge_index, time_vector_flag, pos_time_vector, neg_time_vector):
        decoder = self.decoder
        pos_y = z.new_ones(pos_edge_index.size(1)).to(self.device)
        neg_y = z.new_zeros(neg_edge_index.size(1)).to(self.device)
        y = torch.cat([pos_y, neg_y], dim=0)
        pos_pred = decoder(z, pos_edge_index, time_vector_flag, pos_time_vector)
        neg_pred = decoder(z, neg_edge_index, time_vector_flag, neg_time_vector)
        pred = torch.cat([pos_pred, neg_pred], dim=0)
        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        return roc_auc_score(y, pred), average_precision_score(y, pred)