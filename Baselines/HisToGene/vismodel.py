import os
from argparse import ArgumentParser
from typing import Any, Optional

import pylab as p
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torchmetrics.functional import accuracy
from transformer import Vit
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import ViT_SKIN, ViT_HER2ST



class HisToGene(pl.LightningModule):
    def __init__(self, patch_size=112, n_layers=4, n_genes=1000, dim=1024, learning_rate=1e-4, dropout=0.1,
                 n_pos=128):
        super().__init__()
        self.learning_rate = learning_rate
        patch_dim = 3 * patch_size * patch_size
        self.patch_embedding = nn.Linear(patch_dim, dim)
        self.x_embed = nn.Embedding(n_pos, dim)
        self.y_embed = nn.Embedding(n_pos, dim)

        self.vit = Vit(dim=dim, depth=n_layers, heads=16, mlp_dim=2 * dim, dropout=dropout, emb_dropout=dropout)

        self.gene_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, n_genes)
        )

    def forward(self, patches, centers):
        patches = patches.flatten(2)
        patches = self.patch_embedding(patches)
        centers_x = self.x_embed(centers[:, :, 1])
        centers_y = self.y_embed(centers[:, :, 0])
        x = patches + centers_x + centers_y
        h = self.vit(x)
        x = self.gene_head(h)
        return x

    def training_step(self, batch, batch_idx):
        patch, center, exp = batch
        pred = self(patch, center)
        loss = F.mse_loss(pred.view_as(exp), exp)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        patch, center, exp = batch
        pred = self(patch, center)
        loss = F.mse_loss(pred.view_as(exp), exp)
        self.log('valid_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        patch, center, exp = batch
        pred = self(patch, center)
        loss = F.mse_loss(pred.view_as(exp), exp)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    @staticmethod
    def add_module_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# if __name__ == "__main__":
#     # a = torch.rand(1, 4000, 3 * 112 * 112)
#     # p = torch.ones(1, 4000, 2).long()
#     model = HisToGene(n_genes=785)
#     # print(count_parameters(model))
#     # x = model(a, p)
#     # print(x.shape)
#     # a = torch.rand(1, 3, 3, 3 * 112 * 112)
#     # model = FeatureExtractor()
#     # x = model(a)
#     # print(x.shape)
#     # model = Ours(gene_num=785)