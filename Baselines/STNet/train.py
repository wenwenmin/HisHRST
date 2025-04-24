import argparse
import torch
import random
import numpy as np
import os
from dataset import DATA_BRAIN
from torch.utils.data import DataLoader
from model import STModel
from tqdm import tqdm
import torch.nn.functional as F
import warnings
import anndata as ad
from utils import get_R


warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()

# model super parameters
parser.add_argument('--gene_num', type=int, default=84)  # 785, 685
parser.add_argument("--dropout", type=float, default=0.2)

# train parameters
parser.add_argument("--epoch", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--seed", type=int, default=64)  # 3407 64 62 38
parser.add_argument("--weight_decay", type=float, default=1e-3)

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
adata_path = './adata_pred_result/DLPFC/'

for i in range(3, 12):
    train_dataset = DATA_BRAIN(train=True, fold=i)
    train_dataLoader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = DATA_BRAIN(train=False, fold=i)
    test_dataLoader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print("len(test_dataset)", len(test_dataset))

    model = STModel(gene_num=args.gene_num, dropout=args.dropout)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    torch.set_float32_matmul_precision('high')

    for epoch in range(args.epoch):
        tqdm_train = tqdm(train_dataLoader, total=len(train_dataLoader))
        model.train()
        train_epoch_loss = []
        for patch, position, exp in tqdm_train:
            patch, position, exp = patch.to(device), position.to(device), exp.to(device)
            pred = model(patch)
            loss = F.mse_loss(pred, exp).to(device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_epoch_loss.append(loss.item())
            tqdm_train.set_postfix(train_loss=loss.item(), lr=args.lr, epoch=epoch + 1)


        avg_train_loss = sum(train_epoch_loss) / len(train_epoch_loss)

        # tqdm.write(
        #     f"Epoch {epoch + 1}/{args.epoch} => Avg Train Loss: {avg_train_loss:.4f}")

    torch.save(model.state_dict(), f"./model/DLPFC/{test_dataset.id2name[0]}.ckpt")

    model.load_state_dict(torch.load(f"./model/DLPFC/{test_dataset.id2name[0]}.ckpt"))
    model = model.to(device)
    model.eval()
    preds = None
    ct = None
    gt = None
    with torch.no_grad():
        for patch, loc, exp, center in tqdm(test_dataLoader):
            patch, loc, exp, center = patch.to(device), loc.to(device), exp.to(device), center.to(device)
            pred = model(patch)
            preds = pred.squeeze().cpu().numpy()
            ct = center.squeeze().cpu().numpy()
            gt = exp.squeeze().cpu().numpy()
        adata_pred = ad.AnnData(preds)
        adata_pred.obsm['spatial'] = ct
        adata_truth = ad.AnnData(gt)
        adata_truth.obsm['spatial'] = ct
    adata_pred.write(adata_path + f"{test_dataset.id2name[0]}_pred.h5ad")
    adata_truth.write(adata_path + f"{test_dataset.id2name[0]}_truth.h5ad")
    R = get_R(adata_pred, adata_truth)[0]
    print(R)
    print(np.nanmean(R))
