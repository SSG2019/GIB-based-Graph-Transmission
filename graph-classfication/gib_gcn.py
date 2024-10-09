import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj
import numpy as np


class GIBGCN(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden):
        super(GIBGCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))

        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

        self.cluster1 = Linear(hidden, hidden)
        self.cluster2 = Linear(hidden, 2)
        self.mse_loss = nn.MSELoss()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.cluster1.reset_parameters()
        self.cluster2.reset_parameters()

    def assignment(self, x):

        return self.cluster2(torch.tanh(self.cluster1(x)))

    def aggregate(self, assignment, x, batch, edge_index):

        max_id = torch.max(batch)
        if torch.cuda.is_available():
            EYE = torch.ones(2).cuda()
        else:
            EYE = torch.ones(2)

        all_adj = to_dense_adj(edge_index)[0]

        all_pos_penalty = 0
        all_graph_embedding = []
        all_pos_embedding = []

        st = 0
        end = 0

        for i in range(int(max_id + 1)):

            j = 0

            while batch[st + j] == i and st + j <= len(batch) - 2:
                j += 1

            end = st + j

            if end == len(batch) - 1:
                end += 1

            one_batch_x = x[st:end]
            one_batch_assignment = assignment[st:end]

            group_features = torch.mm(torch.t(one_batch_assignment), one_batch_x)

            pos_embedding = group_features[0].unsqueeze(dim=0)

            Adj = all_adj[st:end, st:end]
            new_adj = torch.mm(torch.t(one_batch_assignment), Adj)
            new_adj = torch.mm(new_adj, one_batch_assignment)

            normalize_new_adj = F.normalize(new_adj, p=1, dim=1, eps=0.00001)

            norm_diag = torch.diag(normalize_new_adj)
            pos_penalty = self.mse_loss(norm_diag, EYE)

            graph_embedding = torch.mean(x, dim=0, keepdim=True)

            all_pos_embedding.append(pos_embedding)
            all_graph_embedding.append(graph_embedding)

            all_pos_penalty = all_pos_penalty + pos_penalty

            st = end

        all_pos_embedding = torch.cat(tuple(all_pos_embedding), dim=0)
        all_graph_embedding = torch.cat(tuple(all_graph_embedding), dim=0)
        all_pos_penalty = all_pos_penalty / (max_id + 1)

        return all_pos_embedding, all_graph_embedding, all_pos_penalty

    def forward(self, data, snr):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)

        assignment = torch.nn.functional.softmax(self.assignment(x), dim=1)

        all_pos_embedding, all_graph_embedding, all_pos_penalty = self.aggregate(assignment, x, batch, edge_index)

        z = all_pos_embedding
        z_square = torch.mul(z, z)
        power = torch.mean(z_square).sqrt()
        if power > 1:
            z = torch.div(z, power)
        all_pos_embedding = z

        z = all_graph_embedding
        z_square = torch.mul(z, z)
        power = torch.mean(z_square).sqrt()
        if power > 1:
            z = torch.div(z, power)
        all_graph_embedding = z

        if snr:
            snr = 10 ** (snr / 10)
            noise_std = 1 / np.sqrt(2 * snr)
            all_pos_embedding = all_pos_embedding + torch.normal(0, noise_std, size=all_pos_embedding.shape).cuda()

        x = F.relu(self.lin1(all_pos_embedding))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1), all_pos_embedding, all_graph_embedding, all_pos_penalty

    def __repr__(self):
        return self.__class__.__name__


class Discriminator(torch.nn.Module):
    def __init__(self, hidden_size):
        super(Discriminator, self).__init__()

        self.input_size = 2 * hidden_size
        self.hidden_size = hidden_size
        self.lin1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.lin2 = torch.nn.Linear(self.hidden_size, 1)
        self.relu = torch.nn.ReLU()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, embeddings, positive):
        cat_embeddings = torch.cat((embeddings, positive), dim=-1)

        pre = self.relu(self.lin1(cat_embeddings))
        pre = self.relu(self.lin2(pre))

        return pre
