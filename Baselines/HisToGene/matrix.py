from skimage.metrics import structural_similarity as ssim
import scanpy as sc
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import normalize
from predict import *

section_id = "B1"
path1 = f'./BC_ST/{section_id}/file_tmp'
adata_histogene = sc.read_h5ad(path1 + '/recovered_data.h5ad')
adata = sc.read_h5ad(path1 + '/raw_data.h5ad')
fig_path = rf"C:\Users\DELL\Desktop\学术垃圾\HisHIST\recover\HisToGene/{section_id}/"
os.makedirs(fig_path, exist_ok=True)

pcc, p_value = get_R(adata_histogene, adata)
mse = mean_squared_error(adata_histogene.X, adata.X)
mae = mean_absolute_error(adata_histogene.X, adata.X)
X1 = adata_histogene.X
X2 = adata.X
X1 = normalize(X1, axis=1)
X2 = normalize(X2, axis=1)
ssim_value = ssim(X1, X2, data_range=X2.max() - X2.min())

print(np.mean(pcc), np.mean(mse), np.mean(mae), ssim_value)


# show_gene=["GNAS","FN1","SCD", "HLA-B", "HLA-DRA", "CD74", "FASN", "IGKC", "MYL12B"]
# sc.set_figure_params(dpi=80, figsize=(2.8, 3))
# sc.pl.embedding(adata_histogene, basis="coord", color=show_gene, s=300, show=False)
# ax = plt.gca()
# for spine in ax.spines.values():
#     spine.set_edgecolor('black')
#     spine.set_linewidth(1)
# plt.savefig(fig_path + "GeneRecover.pdf", format='pdf', bbox_inches="tight")

