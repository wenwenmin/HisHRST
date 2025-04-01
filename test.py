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

input_dir = r'D:\dataset\BreastCancer1'
adata = sc.read_visium(path=input_dir,
                       count_file='V1_Breast_Cancer_Block_A_Section_1_filtered_feature_bc_matrix.h5')
adata.var_names_make_unique()
metadata_file = r'D:\dataset\BreastCancer1\metadata.tsv'
metadata_df = pd.read_csv(metadata_file, sep='\t')
metadata_df = metadata_df.set_index('ID')
adata.obs['annot_type'] = adata.obs_names.map(metadata_df['annot_type'])
adata.obs['fine_annot_type'] = adata.obs_names.map(metadata_df['fine_annot_type'])
sc.pl.spatial(adata, color='fine_annot_type', title="Cell Type Distribution", size=1, show=False)

print(adata)
output_svg_path = r"C:\Users\DELL\Desktop\学术垃圾\HisToSGE\gsea_analy\venn_diagram4.pdf"
plt.savefig(output_svg_path, format='pdf')



# section_list = ["151507", "151508", "151509", "151510", "151669", "151670", "151671", "151672", "151673", "151674",
#                 "151675", "151676"]
# ## 508 669 670 671 672
# for section_id in section_list:
#     section_id = "IDC"
#     save_model_path = f"../IDC/{section_id}"
#     adata, _ = get_sectionData(section_id)
#     adata_HistoSGE = sc.read_h5ad(save_model_path + '/recovered_data.h5ad')
#     print(adata_HistoSGE, adata)
#
#     pr_stage = np.zeros(adata_HistoSGE.shape[1])
#     P_value = np.ones(adata_HistoSGE.shape[1])
#     mse_values = np.zeros(adata_HistoSGE.shape[1])
#     mae_values = np.zeros(adata_HistoSGE.shape[1])
#     used_gene = adata_HistoSGE.var.index
#
#     for it in tqdm(range(adata_HistoSGE.shape[1])):
#         pr_stage[it], P_value[it] = \
#             pearsonr(adata_HistoSGE[:, used_gene[it]].X.toarray().squeeze(),
#                      adata[:, used_gene[it]].X.toarray().squeeze())
#         mse_values[it] = mean_squared_error(adata_HistoSGE[:, used_gene[it]].X.toarray().squeeze(),
#                                             adata[:, used_gene[it]].X.toarray().squeeze())
#         mae_values[it] = mean_absolute_error(adata_HistoSGE[:, used_gene[it]].X.toarray().squeeze(),
#                                              adata[:, used_gene[it]].X.toarray().squeeze())
#
#     X1 = adata_HistoSGE.X
#     X2 = adata[:, used_gene].X.toarray()
#     X1 = normalize(X1, axis=1)
#     X2 = normalize(X2, axis=1)
#     ssim_value = ssim(X1, X2, data_range=X2.max() - X2.min())
#     print("section_id:", section_id, "ssim:", ssim_value)
#     mask = ~np.isnan(pr_stage)
#     pr_stage_n = pr_stage[mask]
#     used_gene_n = used_gene[mask]
#     p_value = P_value[mask]
#     print("section_id:", section_id, "PCC:", np.mean(pr_stage_n))
#     print("section_id:", section_id, "AVG MSE:", np.mean(mse_values))
#     print("section_id:", section_id, "AVG MAE:", np.mean(mae_values))
#
#     sorted_indices = np.argsort(pr_stage_n)[::-1][:5]
#     top_genes = [used_gene_n[idx] for idx in sorted_indices]
#     top_pcc_values = [pr_stage_n[idx] for idx in sorted_indices]
#     print(top_genes)
#     break

# #
# mask = ~np.isnan(pr_stage)
# pr_stage_n = pr_stage[mask]
# used_gene_n = used_gene[mask]
#
#
# sorted_indices = np.argsort(pr_stage_n)[::-1][:5]
# top_genes = [used_gene_n[idx] for idx in sorted_indices]
# top_pcc_values = [pr_stage_n[idx] for idx in sorted_indices]
# print(top_genes)
# print(adata_HistoSGE, adata_sample)
# with open(save_model_path + '/h_features.pkl', 'rb') as f:
#     h_features = pickle.load(f)
#
# # 打印特征形状以确认加载成功
# print(f'Loaded features shape: {h_features.shape}')
# adata_H = sc.AnnData(h_features)
# adata_H.obsm["spatial"] = adata.obsm["spatial"]
# adata_HistoSGE.obsm["spatial"] = adata.obsm["spatial"]
# 使用 KMeans 进行聚类
# ARI_HistoSGE = {}
# ARI_h = {}
#
# n_clusters = 7
#
# kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=456) #661
# kmeans_labels = kmeans.fit_predict(adata_HistoSGE.X) # h_features
#
# adata_HistoSGE.obs['kmeans'] = kmeans_labels.astype(str)
#
# ari = adjusted_rand_score(adata.obs['layer'].astype(str), adata_HistoSGE.obs['kmeans'].astype(str))
# # if ari < 0.18:
# sc.pl.spatial(adata_HistoSGE, color='kmeans', title='KMeans Clustering', spot_size=150, show=False)
# plt.savefig(rf'C:\Users\DELL\Desktop\学术垃圾\HisToSGE/{section_id}_{ari}.pdf', format='pdf')
#
# ARI_HistoSGE[i] = ari
# print(i, ari)
# break

#
#
# max_i = max(ARI_HistoSGE, key=ARI_HistoSGE.get)
# max_ari = ARI_HistoSGE[max_i]
# print(f"Max ARI_HistoSGE: {max_ari} at PCA components: {max_i}")
#
# max_i_h = max(ARI_h, key=ARI_h.get)
# max_ari_h = ARI_h[max_i_h]
# print(f"Max ARI_H: {max_ari_h} at PCA components: {max_i_h}")

# ARI = {}
# for i in range(100, 200):
#     sc.pp.pca(adata_HistoSGE, n_comps=84)
#     sc.tl.tsne(adata_HistoSGE)
#     # print(adata_HistoSGE)
#     adata_HistoSGE.obsm["spatial"] = adata.obsm["spatial"]
#     kmeans_adata_stage = KMeans(n_clusters=7, init='k-means++', random_state=0).fit(adata_HistoSGE.obsm["X_pca"])
#     adata_HistoSGE.obs['kmeans'] = kmeans_adata_stage.labels_.astype(str)
#     ari = adjusted_rand_score(adata.obs['layer'].astype(str), adata_HistoSGE.obs['kmeans'].astype(str))
#     sc.pl.spatial(adata_HistoSGE, basis="spatial", color='kmeans', title=f'KMeans+{ari}', spot_size=150)
#     ARI[i] = ari
#     break
# # 找出最大ARI值及其对应的i
# max_i = max(ARI, key=ARI.get)
# max_ari = ARI[max_i]
# print(f"Max ARI: {max_ari} at PCA components: {max_i}")

# pr_stage = np.zeros(adata_HistoSGE.shape[1])
# P_value = np.ones(adata_HistoSGE.shape[1])
# mse_values = np.zeros(adata_HistoSGE.shape[1])
# mae_values = np.zeros(adata_HistoSGE.shape[1])
# used_gene = adata_HistoSGE.var.index
#
# for it in tqdm(range(adata_HistoSGE.shape[1])):
#     pr_stage[it], P_value[it] = \
#         pearsonr(adata_HistoSGE[:, used_gene[it]].X.toarray().squeeze(), adata[:, used_gene[it]].X.toarray().squeeze())
#     mse_values[it] = mean_squared_error(adata_HistoSGE[:, used_gene[it]].X.toarray().squeeze(),
#                                         adata[:, used_gene[it]].X.toarray().squeeze())
#     mae_values[it] = mean_absolute_error(adata_HistoSGE[:, used_gene[it]].X.toarray().squeeze(),
#                                          adata[:, used_gene[it]].X.toarray().squeeze())
# mask = ~np.isnan(pr_stage)
# pr_stage_n = pr_stage[mask]
# used_gene_n = used_gene[mask]
# p_value = P_value[mask]
# print("section_id:", section_id, "PCC:", np.mean(pr_stage_n))
# print("section_id:", section_id, "AVG MSE:", np.mean(mse_values))
# print("section_id:", section_id, "AVG MAE:", np.mean(mae_values))
#
# sorted_indices = np.argsort(pr_stage_n)[::-1][:5]
# top_genes = [used_gene_n[idx] for idx in sorted_indices]
# top_pcc_values = [pr_stage_n[idx] for idx in sorted_indices]
# #
# #
# for i, gene in enumerate(top_genes):
#     print(f"Top {i + 1}: Gene Name = {gene}, PCC Value = {top_pcc_values[i]}")
#


#
import os

section_id = "FFPE"
save_model_path = f"../T4/{section_id}"
adata_sample = sc.read_h5ad(save_model_path + "/sampled_data.h5ad")
adata, _ = get_sectionData(section_id)
adata_recover = sc.read_h5ad(save_model_path + '/recovered_data.h5ad')

# show_gene = ["SOCS2", "IGHG1", "IGHG4", "ANGPTL7", "TNFRSF1B", "PDPN", 'MGP', 'CXCL14', 'MUC1', 'COX6C', 'IGKC']
# show_gene = ["GNAS", "FN1"]

# show_gene = ["MOBP", "MGP"]
# show_gene = ["Tbr1", "Pcp4"]
# show_gene = ["IGHG3", "SCD"]
show_gene = ["PPDPF", "CD24"]
pcc_show_gene = {gene: pearsonr(adata_recover[:, gene].X.toarray().squeeze(),
                                adata[:, gene].X.toarray().squeeze())[0]
                 for gene in show_gene}

print(pcc_show_gene)
# models = "HisHRST"
# fig_path = rf'C:\Users\DELL\Desktop\学术垃圾\HisHIST\recover\{models}\{section_id}/'
# os.makedirs(fig_path, exist_ok=True)
# sc.set_figure_params(dpi=300, figsize=(2.8, 3))
# sc.pl.embedding(adata_sample, basis="coord", color=show_gene, s=30, show=False)
# plt.savefig(fig_path + 'sample.pdf', format='pdf')
#
# adata_recover = sc.read_h5ad(f"../BC_ST/{section_id}/recovered_data.h5ad")
# adata_4x = sc.read_h5ad(f"../BC_ST/{section_id}/generated_data_4x.h5ad")
# adata_2x = sc.read_h5ad(f"../BC_ST/{section_id}/generated_data_2x.h5ad")
# adata_8x = sc.read_h5ad(f"../BC_ST/{section_id}/generated_data_8x.h5ad")
# X1 = adata_recover.X
# X2 = adata.X
# X1 = normalize(X1, axis=1)
# X2 = normalize(X2, axis=1)
# ssim_value = ssim(X1, X2, data_range=X2.max() - X2.min())
# print(ssim_value)
# exps = adata_4x.X

# from sklearn.preprocessing import MinMaxScaler
#
# exps = adata.X.toarray()
# scaler = MinMaxScaler()
# exps_scaled = scaler.fit_transform(exps.T).T
# adata.X = exps_scaled
#
# exps = adata_recover.X
# scaler = MinMaxScaler()
# exps_scaled = scaler.fit_transform(exps.T).T
# adata_recover.X = exps_scaled

# sc.set_figure_params(dpi=300, figsize=(2.7, 3))
# sc.pl.embedding(adata, basis="coord", color=show_gene, s=30, show=False)
# plt.savefig(fig_path + 'raw.pdf', format='pdf')
#
# sc.set_figure_params(dpi=300, figsize=(2.7, 3))
# sc.pl.embedding(adata_recover, basis="coord", color=show_gene, s=30, show=False)
# plt.savefig(fig_path + 'recover.pdf', format='pdf')

# exps = adata_4x.X
# scaler = MinMaxScaler()
# exps_scaled = scaler.fit_transform(exps.T).T
# adata_4x.X = exps_scaled
#
# exps = adata_2x.X
# scaler = MinMaxScaler()
# exps_scaled = scaler.fit_transform(exps.T).T
# adata_2x.X = exps_scaled
#
# exps = adata_8x.X
# scaler = MinMaxScaler()
# exps_scaled = scaler.fit_transform(exps.T).T
# adata_8x.X = exps_scaled
#
# hs = hotspot.Hotspot(adata, model='none', latent_obsm_key="coord")
# hs.create_knn_graph(weighted_graph=False, n_neighbors=30)
# hs_results = hs.compute_autocorrelations()
# hs_genes = hs_results.loc[hs_results.FDR < 0.05].index
#
# adata_2x = adata_2x[:, pd.DataFrame(adata_2x.X).apply(lambda x: x.sum(), axis=0) > 0]
# hs_8x = hotspot.Hotspot(adata_2x, model='none', latent_obsm_key="coord")
# hs_8x.create_knn_graph(weighted_graph=False, n_neighbors=30)
# hs_stage_results = hs_8x.compute_autocorrelations()
# hs_8x_genes = hs_stage_results.loc[hs_stage_results.FDR < 0.05].index
#
# g = venn2(subsets=[set(hs_genes), set(hs_8x_genes)], set_labels=('Raw', 'HisHRST'),
#           set_colors=("#4f6db4", "#ec7722"), alpha=0.6, normalize_to=1.0)
# g = venn2_circles(subsets=[set(hs_genes), set(hs_8x_genes)], linewidth=0.8, color="black")
# plt.show()
# g = venn2(subsets=[set(hs_genes[range(100)]), set(hs_8x_genes[range(100)])], set_labels=('Raw', 'HisHRST'),
#           set_colors=("#4f6db4", "#ec7722"), alpha=0.6, normalize_to=1.0)
# g = venn2_circles(subsets=[set(hs_genes[range(100)]), set(hs_8x_genes[range(100)])], linewidth=0.8, color="black")
# # plt.show()
# show_gene = hs_8x_genes[range(100)][~hs_8x_genes[range(100)].isin(hs_genes[range(100)])]
# print(show_gene)
# show_gene = ['CCL19', 'TMEM259']
# sc.set_figure_params(dpi=80, figsize=(2.8, 3))
# sc.pl.embedding(adata, basis="coord", color=show_gene, s=300, show=True)
# sc.set_figure_params(dpi=80, figsize=(2.8, 3))
# sc.pl.embedding(adata_8x, basis="coord", color=show_gene, s=80, show=True)
# sc.set_figure_params(dpi=80, figsize=(2.8, 3))
# sc.pl.embedding(adata_recover, basis="coord", color=show_gene, s=300, show=False)
#
#
# sc.set_figure_params(dpi=80, figsize=(2.8, 3))
# sc.pl.embedding(adata, basis="coord", color=show_gene, s=300, show=False)
# plt.savefig(rf'C:\Users\DELL\Desktop\学术垃圾\HisToSGE\Figture/B1_ori.pdf', format='pdf')


# sc.set_figure_params(dpi=80, figsize=(2.8, 3))
# sc.pl.embedding(adata_2x, basis="coord", color=show_gene, s=250, show=False)
# plt.savefig(rf'C:\Users\DELL\Desktop\学术垃圾\HisToSGE\Figture/B1_2x.pdf', format='pdf')
#
# sc.set_figure_params(dpi=80, figsize=(2.8, 3))
# sc.pl.embedding(adata_4x, basis="coord", color=show_gene, s=110, show=False)
# plt.savefig(rf'C:\Users\DELL\Desktop\学术垃圾\HisToSGE\Figture/B1_4x.pdf', format='pdf')
#
# sc.set_figure_params(dpi=80, figsize=(2.8, 3))
# sc.pl.embedding(adata_8x, basis="coord", color=show_gene, s=80, show=False)
# plt.savefig(rf'C:\Users\DELL\Desktop\学术垃圾\HisToSGE\Figture/B1_8x.pdf', format='pdf')


# adata_sc = sc.read_h5ad("D:\dataset\BreastCancer1/all_raw_data.h5ad")
# gene_list = list(np.load('D:\dataset\Her2st\data/her_hvg_cut_1000.npy', allow_pickle=True))
# sc.pp.highly_variable_genes(adata_sc, flavor="seurat_v3", n_top_genes=1000)
# sc.pp.normalize_total(adata_sc, target_sum=1e4)
# sc.pp.log1p(adata_sc)
# adata_sc.obs["celltype_major"].value_counts()
# sc.tl.rank_genes_groups(adata_sc, 'celltype_major', method='wilcoxon')
# print(adata_sc)
# DE_df = sc.get.rank_genes_groups_df(adata_sc, group=None, log2fc_min=4)
# DE_df = DE_df[DE_df.pvals_adj < 0.05]
# DE_df = DE_df[DE_df.names.isin(adata_4x.var.index)]
# print(DE_df)
#
# DE_df.group.value_counts()
#
# expr_raw = pd.DataFrame(sp.coo_matrix(adata.X).todense())
# expr_raw.columns = adata.var.index
# expr_raw.index = adata.obs.index
#
# expr_2x = pd.DataFrame(sp.coo_matrix(adata_2x.X).todense())
# expr_2x.columns = adata_2x.var.index
# expr_2x.index = adata_2x.obs.index
#
# expr_4x = pd.DataFrame(sp.coo_matrix(adata_4x.X).todense())
# expr_4x.columns = adata_4x.var.index
# expr_4x.index = adata_4x.obs.index
#
# expr_8x = pd.DataFrame(sp.coo_matrix(adata_8x.X).todense())
# expr_8x.columns = adata_8x.var.index
# expr_8x.index = adata_8x.obs.index
#
#
#
# print(expr_raw.shape, expr_4x.shape)
#
# for celltype in DE_df.group.unique():
#     adata.obs[celltype] = expr_raw.loc[:, DE_df.names[DE_df.group == celltype]].mean(axis=1)
#     adata_4x.obs[celltype] = expr_4x.loc[:, DE_df.names[DE_df.group == celltype]].mean(axis=1)
#     adata_2x.obs[celltype] = expr_2x.loc[:, DE_df.names[DE_df.group == celltype]].mean(axis=1)
#     adata_8x.obs[celltype] = expr_8x.loc[:, DE_df.names[DE_df.group == celltype]].mean(axis=1)
#
# adata.obsm['coord'][:, 1] = adata.obsm['coord'][:, 1] * (-1)
# adata_4x.obsm['coord'][:, 1] = adata_4x.obsm['coord'][:, 1] * (-1)
# adata_2x.obsm['coord'][:, 1] = adata_2x.obsm['coord'][:, 1] * (-1)
# adata_8x.obsm['coord'][:, 1] = adata_8x.obsm['coord'][:, 1] * (-1)
#
# show_celltype = ["B-cells", "T-cells",  "CAFs", "Cancer Epithelial", "Myeloid"]
#
# sc.set_figure_params(dpi=80, figsize=(2.7, 3))
# sc.pl.embedding(adata, basis="coord", color=show_celltype,cmap='plasma', s=300, show=False)
# plt.savefig(rf'C:\Users\DELL\Desktop\学术垃圾\HisToSGE\Figture/B1_cellytpe_ori.pdf', format='pdf')
#
# sc.set_figure_params(dpi=80, figsize=(2.7, 3))
# sc.pl.embedding(adata_2x, basis="coord", color=show_celltype,cmap='plasma', s=250, show=False)
# plt.savefig(rf'C:\Users\DELL\Desktop\学术垃圾\HisToSGE\Figture/B1_cellytpe_2x.pdf', format='pdf')
#
# sc.set_figure_params(dpi=80, figsize=(2.7, 3))
# sc.pl.embedding(adata_4x, basis="coord", color=show_celltype,cmap='plasma', s=110, show=False)
# plt.savefig(rf'C:\Users\DELL\Desktop\学术垃圾\HisToSGE\Figture/B1_cellytpe_4x.pdf', format='pdf')
#
# sc.set_figure_params(dpi=80, figsize=(2.7, 3))
# sc.pl.embedding(adata_8x, basis="coord", color=show_celltype,cmap='plasma', s=80, show=False)
# plt.savefig(rf'C:\Users\DELL\Desktop\学术垃圾\HisToSGE\Figture/B1_cellytpe_8x.pdf', format='pdf')


# show_gene = ["GNAS", "FN1", "FASN", "HLA-B", "SCD", "IGKC"]
#
# sc.set_figure_params(dpi=80, figsize=(2.7, 3))
# sc.pl.embedding(adata, basis="coord", color=show_gene, s=300, show=True)
#
# sc.set_figure_params(dpi=80, figsize=(2.7, 3))
# sc.pl.embedding(adata_4x, basis="coord", color=show_gene, s=100, show=True)
#
# model_raw = KMeans(n_clusters=3)
# model_stage = KMeans(n_clusters=3)
#
# model_raw.fit(adata.obs[DE_df.group.unique()])
# model_stage.fit(adata_4x.obs[DE_df.group.unique()])
#
# adata.obs["K-means"] = model_raw.predict(adata.obs[DE_df.group.unique()])
# adata_4x.obs["K-means"] = model_stage.predict(adata_4x.obs[DE_df.group.unique()])
#
# adata.obs["segmentation"] = adata.obs["K-means"].replace([0, 1, 2], ["R1", "R2", "R3"])
# adata_4x.obs["segmentation"] = adata_4x.obs["K-means"].replace([0, 1, 2], ["S1", "S2", "S3"])

# adata.uns['segmentation_colors'] = ['#0000FF', '#FFFF00', '#FF0000']
# adata_4x.uns['segmentation_colors'] = ['#0000FF', '#FFFF00', '#FF0000']
#
# sc.set_figure_params(dpi=80, figsize=(2.7, 3))
# sc.pl.embedding(adata, basis="coord", color="segmentation", title="", s=300, show=True)
#
# sc.set_figure_params(dpi=80, figsize=(2.7, 3))
# sc.pl.embedding(adata_4x, basis="coord", color="segmentation", title="", s=100, show=True)
#
# print()
# for i in range(100, 200):
# sc.pp.pca(adata)
# sc.tl.tsne(adata)
# # print(adata_HistoSGE)
#
# kmeans_adata_stage = KMeans(n_clusters=5, init='k-means++', random_state=0).fit(adata.obsm["X_pca"])
# adata.obs['kmeans'] = kmeans_adata_stage.labels_.astype(str)
# ari = adjusted_rand_score(adata.obs['label'].astype(str), adata.obs['kmeans'].astype(str))
# sc.pl.spatial(adata, basis="spatial", color='kmeans', title=f'KMeans+{ari}', spot_size=150)
# ARI[i] = ari
# break


# for i in range(1000):
#
#     kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)  # 661
#     kmeans_labels = kmeans.fit_predict(adata_4x.X)  # h_features
#
#     adata_4x.obs['kmeans'] = kmeans_labels.astype(str)
#
#     # ari = adjusted_rand_score(adata.obs['label'].astype(str), adata.obs['kmeans'].astype(str))
#     sc.pl.spatial(adata_4x,  color='kmeans', title='KMeans Clustering', spot_size=120, show=True)
#     # plt.savefig(rf'C:\Users\DELL\Desktop\学术垃圾\HisToSGE/{section_id}_{ari}.pdf', format='pdf')
#
#     # print(i, ari)
#     break


# ax = plt.gca()
# for spine in ax.spines.values():
#     spine.set_edgecolor('black')
#     spine.set_linewidth(1)
# plt.savefig(save_fig_path + "GeneTruth.pdf",RST format='pdf', bbox_inches="tight")
#
#
# sc.set_figure_params(dpi=80, figsize=(2.8, 3))
# sc.pl.embedding(adata_sample, basis="coord", color=show_gene, s=30, show=False)
# ax = plt.gca()
# for spine in ax.spines.values():
#     spine.set_edgecolor('black')
#     spine.set_linewidth(1)
# plt.savefig(save_fig_path + "GeneSample.pdf", format='pdf', bbox_inches="tight")
#
#
#

#
# # %%
# sc.set_figure_params(dpi=80, figsize=(2.8, 3))
# sc.pl.embedding(adata_HistoSGE, basis="coord", color=show_gene, s=30, show=True)
# ax = plt.gca()
# for spine in ax.spines.values():
#     spine.set_edgecolor('black')
#     spine.set_linewidth(1)
#
# positions = {
#     show_gene[0]: (10, 10),
#     show_gene[1]: (20, 20),
#     show_gene[2]: (30, 30),
# }
# # Add PCC annotations to the plot
# for gene in show_gene:
#     x, y = positions[gene]
#     ax.text(x, y, f"{gene}\nPCC: {pcc_show_gene[gene]:.2f}",
#             fontsize=8, ha='center', va='center', color='black', bbox=dict(facecolor='white', alpha=0.5))
#
# # plt.savefig(save_fig_path + "GeneRecovery.pdf", format='pdf', bbox_inches="tight")
# sc.set_figure_params(dpi=80, figsize=(2.8, 3))
# sc.pl.embedding(adata_HistoSGE, basis="coord", color=show_gene, s=30, show=False)
# ax = plt.gca()
# for spine in ax.spines.values():
#     spine.set_edgecolor('black')
#     spine.set_linewidth(1)
# plt.savefig(save_fig_path + "GeneRecovery.pdf", format='pdf', bbox_inches="tight")
