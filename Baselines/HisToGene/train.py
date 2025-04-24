import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from utils import *
from predict import model_predict, sr_predict
from dataset import ViT_HER2ST, ViT_SKIN
from vismodel import HisToGene

dataset = "B1"
if dataset == "T1":
    section_list = ["151507", "151509", "151510", "151669", "151670", "151671", "151672", "151673", "151674", "151675",
                    "151676"]
    for section_id in section_list:
        start_time = time.time()
        # section_id = "151507"
        tag = f'T1_id_{section_id}'
        path1 = f'./T1_HistoGene/{section_id}/file_tmp'
        dataset_path = f'..\dataset\DLPFC/{section_id}'
        dataset = ViT_SKIN(dataset="T1", dataset_path=dataset_path, path1=path1, train=True, section_id=section_id)
        train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

        model = HisToGene(n_layers=8, n_genes=1000, learning_rate=1e-5)
        trainer = pl.Trainer(max_epochs=100, log_every_n_steps=1)
        trainer.fit(model, train_loader)
        trainer.save_checkpoint(f"model/last_train_{tag}.ckpt")

        end_time = time.time()

        elapsed_time = end_time - start_time

        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        time_str = f"{int(hours)}小时 {int(minutes)}分 {int(seconds)}秒"

        print(f"Total training time: {time_str}")

        with open(path1 + f"/training_time_{section_id}.txt", "w", encoding='utf-8') as file:
            file.write(f"Total training time: {time_str}\n")
elif dataset == 'T2':
    start_time = time.time()
    # section_id = "151507"
    tag = f'T2_Mouse_brain'
    path1 = f'./T2_HistoGene/Mouse_brain/file_tmp'
    dataset_path = f'..\dataset\T2'
    dataset = ViT_SKIN(dataset="T2", dataset_path=dataset_path, path1=path1, train=True, section_id=None)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = HisToGene(n_layers=8, n_genes=1000, learning_rate=1e-4)
    trainer = pl.Trainer(max_epochs=800, log_every_n_steps=1)
    trainer.fit(model, train_loader)
    trainer.save_checkpoint(f"model/last_train_{tag}.ckpt")

    end_time = time.time()

    elapsed_time = end_time - start_time

    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    time_str = f"{int(hours)}小时 {int(minutes)}分 {int(seconds)}秒"

    print(f"Total training time: {time_str}")

    with open(path1 + f"/training_time.txt", "w", encoding='utf-8') as file:
        file.write(f"Total training time: {time_str}\n")

elif dataset == 'B1':
    start_time = time.time()
    # section_id = "151507"
    tag = f'B1'
    path1 = f'./BC_ST/B1/file_tmp'
    os.makedirs(path1, exist_ok=True)
    # dataset = ViT_HER2ST(train=True, name=dataset)
    # train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    # model = HisToGene(n_layers=8, n_genes=785, learning_rate=1e-4)
    # trainer = pl.Trainer(max_epochs=100)
    # trainer.fit(model, train_loader)
    # trainer.save_checkpoint(f"model/last_train_{tag}.ckpt")
    #
    # end_time = time.time()
    #
    # elapsed_time = end_time - start_time
    #
    # hours, rem = divmod(elapsed_time, 3600)
    # minutes, seconds = divmod(rem, 60)
    # time_str = f"{int(hours)}小时 {int(minutes)}分 {int(seconds)}秒"
    #
    # print(f"Total training time: {time_str}")
    #
    # with open(path1 + f"/training_time.txt", "w", encoding='utf-8') as file:
    #     file.write(f"Total training time: {time_str}\n")

    model = HisToGene.load_from_checkpoint("model/last_train_" + tag + ".ckpt", n_layers=8,
                                           n_genes=785, learning_rate=1e-4)

    test_dataset = ViT_HER2ST(train=False, name=dataset)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    device = torch.device('cuda')
    adata_pred, adata_truth = model_predict(model, test_loader, device=device)

    use_gene = np.load('..\dataset\Her2st\data/her_hvg_cut_1000.npy', allow_pickle=True)
    adata_pred.var.index = use_gene
    adata_pred.obs = adata_truth.obs
    adata_pred.write_h5ad(path1 + '/recovered_data.h5ad')
    adata_truth.write_h5ad(path1 + '/raw_data.h5ad')
