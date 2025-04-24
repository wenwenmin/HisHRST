import numpy as np
import torch
from dataset import DATA_BRAIN
from model import STModel
from torch.utils.data import DataLoader
from tqdm import tqdm
import anndata as ad
from utils import get_R
import warnings
warnings.filterwarnings('ignore')

test_dataset = DATA_BRAIN(train=False, fold=2)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = STModel(gene_num=84, dropout=0.2)
model.load_state_dict(torch.load(f"./model/DLPFC/{test_dataset.id2name[0]}.ckpt"))
model = model.to(device)
model.eval()
preds = None
ct = None
gt = None
with torch.no_grad():
    for patch, loc, exp, center in tqdm(test_loader):
        patch, loc, exp, center = patch.to(device), loc.to(device), exp.to(device), center.to(device)
        pred = model(patch)
        preds = pred.squeeze().cpu().numpy()
        ct = center.squeeze().cpu().numpy()
        gt = exp.squeeze().cpu().numpy()
    adata = ad.AnnData(preds)
    adata.obsm['spatial'] = ct
    adata_gt = ad.AnnData(gt)
    adata_gt.obsm['spatial'] = ct
R = get_R(adata, adata_gt)[0]
print(R)
print(np.nanmean(R))