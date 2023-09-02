import random
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.inits import reset
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from loss import SupConLoss
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


EPS = 1e-15

class SugbCon(torch.nn.Module):

    def __init__(self, hidden_channels, encoder, pool, scorer, beta, degree_inv):
        super(SugbCon, self).__init__()
        self.SupConLoss = SupConLoss()
        self.encoder = encoder
        self.hidden_channels = hidden_channels
        self.pool = pool
        self.scorer = scorer
        self.marginloss = nn.MarginRankingLoss(0.5)
        self.sigmoid = nn.Sigmoid()
        # self.prompt = nn.parameter.Parameter(torch.rand(4))
        self.reset_parameters()
        
    def reset_parameters(self):
        reset(self.scorer)
        reset(self.encoder)
        reset(self.pool)

    def forward(self, x, edge_index, batch=None, index=None, edge_attr=None):
        r""" Return node and subgraph representations of each node before and after being shuffled """
        hidden = self.encoder(x, edge_index, edge_attr)
        if index is None:
            return hidden
        
        z = hidden[index]
        summary = self.pool(hidden, edge_index, batch)
        return z, summary


    def loss(self, hidden1, summary1, labels=None):
        features = torch.cat([hidden1.unsqueeze(1), summary1.unsqueeze(1)], dim=1)

        loss = self.SupConLoss(features, labels)
        return loss


    def test(self, train_z, train_y, test_z, test_y, solver='lbfgs',
             multi_class='auto', *args, **kwargs):
        r"""Evaluates latent space quality via a logistic regression downstream task."""
        clf = LogisticRegression(solver=solver, max_iter=500, multi_class=multi_class, *args,
                                 **kwargs).fit(train_z.detach().cpu().numpy(),
                                               train_y)
        test_acc = clf.score(test_z.detach().cpu().numpy(), test_y)
        return test_acc
    
    def clustering_test(self, test_z, test_y, n_way, rs=0):
        pred_y = KMeans(n_clusters=n_way, random_state=rs).fit(test_z.detach().cpu().numpy()).labels_
        nmi = normalized_mutual_info_score(test_y, pred_y)
        ari = adjusted_rand_score(test_y, pred_y)

        return nmi, ari
