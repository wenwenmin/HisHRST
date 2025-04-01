import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score
from skimage.metrics import structural_similarity as ssim
from model import *
from dataset import *
from config import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error
import warnings
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")
import hotspot
from matplotlib_venn import venn2, venn2_circles

section_list = ["151507", "151508", "151509", "151510", "151669", "151670", "151671", "151672", "151673", "151674",
                "151675", "151676"]

for section_id in section_list:

    save_model_path = f"../DLPFC/{section_id}"
    adata, _ = get_sectionData(section_id)
    adata_HisHRST = sc.read_h5ad(save_model_path + '/recovered_data.h5ad')
    print(adata_HisHRST, adata)

    pr_HisHRST = np.zeros(adata_HisHRST.shape[1])
    P_value = np.ones(adata_HisHRST.shape[1])
    mse_values = np.zeros(adata_HisHRST.shape[1])
    mae_values = np.zeros(adata_HisHRST.shape[1])
    used_gene = adata_HisHRST.var.index

    for it in tqdm(range(adata_HisHRST.shape[1])):
        F[it], P_value[it] = \
            pearsonr(adata_HisHRST[:, used_gene[it]].X.toarray().squeeze(),
                     adata[:, used_gene[it]].X.toarray().squeeze())
        mse_values[it] = mean_squared_error(adata_HisHRST[:, used_gene[it]].X.toarray().squeeze(),
                                            adata[:, used_gene[it]].X.toarray().squeeze())
        mae_values[it] = mean_absolute_error(adata_HisHRST[:, used_gene[it]].X.toarray().squeeze(),
                                             adata[:, used_gene[it]].X.toarray().squeeze())
      
    mask = ~np.isnan(pr_HisHRST)
    pr_HisHRST_n = pr_HisHRST[mask]
    used_gene_n = used_gene[mask]
    p_value = P_value[mask]
    print("section_id:", section_id, "PCC:", np.mean(pr_HisHRST_n))
    print("section_id:", section_id, "AVG MSE:", np.mean(mse_values))
    print("section_id:", section_id, "AVG MAE:", np.mean(mae_values))

