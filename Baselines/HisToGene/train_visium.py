import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from utils import *
from predict import model_predict, sr_predict, model_predict_visium, get_R
from dataset import ViT_HER2ST, ViT_SKIN, ViT_Anndata
from vismodel import HisToGene
from scanpy import read_visium
from window_adata import *

data_dir = 'D:\dataset\DLPFC/'
section_list = ["151507", "151508", "151509", "151510", "151669", "151670", "151671", "151672", "151673", "151674",
                "151675", "151676"]
samples = {i: data_dir + i for i in section_list}
adata_dict = {name: read_visium(path, count_file=f'{name}_filtered_feature_bc_matrix.h5', library_id=name,
                                source_image_path=path + f'/spatial/{name}_full_image.tif') for name, path in
              samples.items()}

for k, v in adata_dict.items():
    v.var_names_make_unique()
sizes = [3000 for i in range(len(adata_dict))]

adata_sub_dict = window_adata(adata_dict, sizes)
gene_list = list(np.load('data/brain_cut_1000.npy'))
adata_path = './adata_pred_result/DLPFC/'

for fold in range(0, 12):
    # fold = fold
    test_sample = section_list[fold]
    test_sample_ori = section_list[fold]
    fold2name = dict(enumerate(section_list))

    model = HisToGene(n_layers=8, n_genes=84, learning_rate=1e-5)
    # trainer = pl.Trainer(
    #     accelerator='gpu',
    #     devices=1,
    #     max_epochs=100
    # )
    # train_set = list(
    #     set(list(adata_sub_dict.keys())) - set([i for i in list(adata_sub_dict.keys()) if test_sample in i]))
    # trainset = ViT_Anndata(adata_dict=adata_sub_dict, train_set=train_set, gene_list=gene_list,
    #                        train=True, flatten=True, adj=False, ori=False, prune='NA', neighs=4,
    #                        )
    # train_loader = DataLoader(trainset, batch_size=1, shuffle=True)
    #
    # torch.set_float32_matmul_precision('high')
    # trainer.fit(model, train_loader)
    #
    # torch.save(model.state_dict(),
    #            f"./model/DLPFC/{test_sample_ori}-HisToGene.ckpt")

    model.load_state_dict(
        torch.load(f"./model/DLPFC/{test_sample_ori}-HisToGene.ckpt"))
    test_sample_list = [i for i in list(adata_dict.keys()) if test_sample in i]
    test_set = list(set(list(adata_dict.keys())) - set(test_sample_list))
    test_dataset = ViT_Anndata(adata_dict=adata_dict, train_set=test_set, gene_list=gene_list,
                               train=False, flatten=True, adj=False, ori=False, prune='NA', neighs=4,
                               )
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0, shuffle=False)

    adata_pred, adata_truth = model_predict_visium(model, test_loader, 'cuda')
    adata_pred.write(adata_path + f"{test_sample_ori}_pred.h5ad")
    adata_truth.write(adata_path + f"{test_sample_ori}_truth.h5ad")

    print(adata_pred.shape, adata_truth.shape)
    R = get_R(adata_pred, adata_truth)[0]
    print('Pearson Correlation:', np.nanmean(R))
