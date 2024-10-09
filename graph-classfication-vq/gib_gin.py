import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GINConv
from torch_geometric.utils import to_dense_adj

from vq_ema import VectorQuantizerEMA


class GIBGIN(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden):
        super(GIBGIN, self).__init__()
        self.conv1 = GINConv(
            Sequential(
                Linear(dataset.num_features, hidden),
                ReLU(),
                Linear(hidden, hidden),
                ReLU(),
                BN(hidden),
            ), train_eps=False)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden, hidden),
                        ReLU(),
                        Linear(hidden, hidden),
                        ReLU(),
                        BN(hidden),
                    ), train_eps=False))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

        self.cluster1 = Linear(hidden, hidden)
        self.cluster2 = Linear(hidden, 2)
        self.mse_loss = nn.MSELoss()

        self.num_embeddings = 128
        self.vq_vae = VectorQuantizerEMA(self.num_embeddings, embedding_dim=hidden, decay=0.2)

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
            normalize_new_adj = F.normalize(new_adj, p=1, dim=1)
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

    def trans_matrix(self, diagonal_value, size):
        matrix = np.zeros((size, size))
        np.fill_diagonal(matrix, diagonal_value)

        other_value = (1 - diagonal_value) / (size - 1)
        matrix[np.where(matrix == 0)] = other_value
        matrix = matrix.astype(np.float32)
        tensor_matrix = torch.from_numpy(matrix)
        return tensor_matrix

    def forward(self, data, correct_p, commitment_cost):
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

        encodings, codebook = self.vq_vae(all_pos_embedding)

        trans_matrix = self.trans_matrix(correct_p, self.num_embeddings).to(encodings.device)
        transmitted_encodings = torch.zeros_like(encodings)
        for i in range(len(encodings)):
            p = random.random()
            if p <= trans_matrix[i][i]:
                transmitted_encodings[i] = encodings[i]
            else:
                integer = int((p - trans_matrix[i][i]) // ((1 - correct_p) / self.num_embeddings))
                transmitted_encodings[i] = (encodings[(i + 1 + integer) % len(encodings)])

        encodings = transmitted_encodings

        quantized = torch.matmul(encodings, codebook)

        e_latent_loss = F.mse_loss(quantized.detach(), all_pos_embedding)
        q_latent_loss = F.mse_loss(quantized, all_pos_embedding.detach())
        vq_loss = q_latent_loss + commitment_cost * e_latent_loss

        quantized = all_pos_embedding + (quantized - all_pos_embedding).detach()

        all_pos_embedding = quantized
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
