import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loss import *
from dataset import *
from utils import *
from transformer import Transformer
from abmil import BatchedABMIL


class Feed(nn.Module):
    def __init__(self, gene_number, X_dim):
        super(Feed, self).__init__()
        self.fc6 = nn.Linear(X_dim, 1024)
        self.fc6_bn = nn.BatchNorm1d(1024)
        self.fc7 = nn.Linear(1024, 2048)
        self.fc7_bn = nn.BatchNorm1d(2048)
        self.fc8 = nn.Linear(2048, 2048)
        self.fc8_bn = nn.BatchNorm1d(2048)
        self.fc9 = nn.Linear(2048, gene_number)

    def forward(self, z, relu):
        h6 = F.relu(self.fc6_bn(self.fc6(z)))
        h7 = F.relu(self.fc7_bn(self.fc7(h6)))
        h8 = F.relu(self.fc8_bn(self.fc8(h7)))
        if relu:
            return F.relu(self.fc9(h8))
        else:
            return self.fc9(h8)


class HisHRST(nn.Module):
    def __init__(self, in_features, depth, heads, n_genes=1000, dropout=0.):
        super(HisHRST, self).__init__()
        self.x_embed = nn.Embedding(512, in_features)
        self.y_embed = nn.Embedding(512, in_features)
        self.trans = Transformer(dim=in_features, depth=depth, heads=heads, dim_head=64, mlp_dim=in_features,
                                 dropout=dropout)

        self.gene_head = nn.Sequential(
            nn.Linear(in_features, 2048),
            nn.ReLU(),
            nn.LayerNorm(2048),
            nn.Linear(2048, n_genes)
        )
        self.layer_norm = nn.LayerNorm(in_features)

    def forward(self, image, centers):

        centers_x = self.x_embed(centers[:, :, 0].long())
        centers_y = self.y_embed(centers[:, :, 1].long())

        x = image + centers_x + centers_y
        h = self.trans(x)
        x = self.gene_head(h)

        return h, F.relu(x)
