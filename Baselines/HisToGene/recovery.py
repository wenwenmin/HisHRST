import os

import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from utils import *
from predict import model_predict, sr_predict
from dataset import ViT_HER2ST, ViT_SKIN
from vismodel import HisToGene
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error

dataset = "T1"
if dataset == "T1":
    section_list = ["151507", "151508", "151509", "151510", "151669", "151670", "151671", "151672", "151673", "151674",
                    "151675",
                    "151676"]

    for section_id in section_list:
        section_id = "mouse_brain"
        print("当前section_id:", section_id)
        tag = f'T2_Mouse_brain'
        path1 = f'./T2_HistoGene/Mouse_brain/file_tmp'
        save_fig_path = rf"C:\Users\DELL\Desktop\学术垃圾\HisHIST\recover\HisToGene/{section_id}"

        os.makedirs(save_fig_path, exist_ok=True)
        model = HisToGene.load_from_checkpoint("model/4xmodel/last_train_" + tag + ".ckpt", n_layers=8,
                                               n_genes=1000, learning_rate=1e-4)
        device = torch.device('cuda')
        dataset_path = f'D:\dataset\MouseBrain'
        dataset = ViT_SKIN(dataset="T2", dataset_path=dataset_path, path1=path1, train=False, section_id=section_id)
        adata_sample = dataset.adata_sample
        adata = dataset.adata_copy
        test_loader = DataLoader(dataset, batch_size=1)

        adata_pred, adata_truth = model_predict(model, test_loader, device=device)

        adata_pred.obsm["coord"] = adata.obsm["coord"]
        use_gene = np.loadtxt(path1 + '/used_gene.txt', dtype=str)
        adata_pred.var.index = use_gene
        adata_pred.obs = adata.obs

        pr_stage = np.zeros(shape=(adata_pred.shape[1]))
        used_gene = adata_pred.var.index

        for it in tqdm(range(adata_pred.shape[1])):
            pr_stage[it] = \
                pearsonr(adata_pred[:, used_gene[it]].X.toarray().squeeze(),
                         adata[:, used_gene[it]].X.toarray().squeeze())[0]

        mask = ~np.isnan(pr_stage)
        pr_stage_n = pr_stage[mask]
        np.savetxt(save_fig_path + "/pcc.txt", pr_stage_n)
        print("avg PCC: ", np.mean(pr_stage_n))

        mse_values = np.zeros(adata_pred.shape[1])
        mae_values = np.zeros(adata_pred.shape[1])
        for it in tqdm(range(adata_pred.shape[1])):
            gene_expression_stage = adata_pred[:, used_gene[it]].X.toarray().squeeze()
            gene_expression = adata[:, used_gene[it]].X.toarray().squeeze()

            mse_values[it] = mean_squared_error(gene_expression, gene_expression_stage)
            mae_values[it] = mean_absolute_error(gene_expression, gene_expression_stage)
        mse_values_filtered = mse_values[~np.isnan(mse_values)]
        mae_values_filtered = mae_values[~np.isnan(mae_values)]
        np.savetxt(save_fig_path + '/mse_values.txt', mse_values_filtered, fmt='%f')
        np.savetxt(save_fig_path + '/mae_values.txt', mae_values_filtered, fmt='%f')
        print("AVG MSE:", np.mean(mse_values_filtered))
        print("AVG MAE:", np.mean(mae_values_filtered))

        # sc.set_figure_params(dpi=100, figsize=(3, 4))
        # pr_stage_common = np.loadtxt(save_fig_path + "/pcc.txt")

        show_gene = ["TBR1", "ENC1", "MOBP", "SNAP25", "MGP"]

        pcc_show_gene = {gene: pearsonr(adata_pred[:, gene].X.toarray().squeeze(),
                                        adata[:, gene].X.toarray().squeeze())[0]
                         for gene in show_gene}

        # %%
        sc.set_figure_params(dpi=300, figsize=(2.8, 3))
        sc.pl.embedding(adata_pred, basis="coord", color=show_gene, s=30, show=False)
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1)

        positions = {
            show_gene[0]: (10, 10),
            show_gene[1]: (20, 20),
            show_gene[2]: (30, 30),
            show_gene[3]: (40, 40),
            show_gene[4]: (50, 50)
        }
        # Add PCC annotations to the plot
        for gene in show_gene:
            x, y = positions[gene]
            ax.text(x, y, f"{gene}\nPCC: {pcc_show_gene[gene]:.2f}",
                    fontsize=8, ha='center', va='center', color='black', bbox=dict(facecolor='white', alpha=0.5))

        plt.savefig(save_fig_path + "GeneRecovery.pdf", format='pdf', bbox_inches="tight")

        # sc.pp.pca(adata, n_comps=30)
        # sc.pp.neighbors(adata, use_rep='X_pca')
        # sc.tl.umap(adata)
        # used_adata = adata[adata.obs['layer'].isna() == False,]
        # sc.tl.paga(used_adata, groups='layer')
        #
        # sc.pp.pca(adata_sample, n_comps=30)
        # sc.pp.neighbors(adata_sample, use_rep='X_pca')
        # sc.tl.umap(adata_sample)
        # used_adata_sample = adata_sample[adata_sample.obs['layer'].isna() == False,]
        # sc.tl.paga(used_adata_sample, groups='layer')
        #
        # sc.pp.pca(adata_pred, n_comps=30)
        # sc.pp.neighbors(adata_pred, use_rep='X_pca')
        # sc.tl.umap(adata_pred)
        # used_adata_stage = adata_pred[adata_pred.obs['layer'].isna() == False,]
        # sc.tl.paga(used_adata_stage, groups='layer')
        #
        # sc.set_figure_params(dpi=80, figsize=(4, 3))
        # sc.pl.paga_compare(used_adata, legend_fontsize=10, frameon=True, size=20, edge_width_scale=0.2,
        #                    threshold=0.01, fontsize=10,
        #                    title=section_id + '_raw', legend_fontoutline=2, show=False)
        # plt.savefig(save_fig_path + "UMAP_Raw.pdf", format='pdf', bbox_inches="tight")
        #
        # sc.set_figure_params(dpi=80, figsize=(4, 3))
        # sc.pl.paga_compare(used_adata_sample, legend_fontsize=10, frameon=True, size=20, edge_width_scale=0.2,
        #                    threshold=0.01, fontsize=10,
        #                    title=section_id + '_sample', legend_fontoutline=2, show=False)
        # plt.savefig(save_fig_path + "UMAP_Down-sampling.pdf", format='pdf', bbox_inches="tight")
        #
        # sc.set_figure_params(dpi=80, figsize=(4, 3))
        # sc.pl.paga_compare(used_adata_stage, legend_fontsize=10, frameon=True, size=20, edge_width_scale=0.2,
        #                    threshold=0.01, fontsize=10,
        #                    title=section_id + '_recoverd', legend_fontoutline=2, show=False)
        # plt.savefig(save_fig_path + "UMAP_Recovered.pdf", format='pdf', bbox_inches="tight")


        # common_non_nan = np.logical_and(~np.isnan(pr_stage), ~np.isnan(pr_vge))
        # pr_stage_common = pr_stage[common_non_nan]
        # pr_vge_common = pr_vge[common_non_nan]
        # distances = [pr_stage_common]  # , pr_vge_common
        # colors = ["#00BFC4"]  # , "#B3DE69"
        # box_color = "grey"
        #
        # fig, ax = plt.subplots()
        #
        # boxplot = ax.boxplot(distances, sym='k+', showfliers=False, widths=0.3, patch_artist=True)
        # for i, box in enumerate(boxplot['boxes']):
        #     box.set_facecolor(colors[i])
        #
        # violinplot = ax.violinplot(distances, showmedians=True, widths=0.8, showextrema=False)
        # for partname, part in violinplot.items():
        #     if partname == 'bodies':
        #         for pc, color in zip(part, colors):
        #             pc.set_alpha(0.5)
        #             pc.set_facecolor(color)
        #
        # ax.set_ylim(top=1)
        # ax.set_xticks([1])  # , 2
        # ax.set_xticklabels(['HistoGene'])  # , 'VGE'
        # ax.set_ylabel('Pearson Correlation', fontsize=20)
        # plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        # ax = plt.gca()
        # for spine in ax.spines.values():
        #     spine.set_edgecolor('black')
        #     spine.set_linewidth(2)
        # ytick_labels = plt.gca().get_yticklabels()
        # for label in ytick_labels:
        #     label.set_fontsize(20)
        # xtick_labels = plt.gca().get_xticklabels()
        # for label in xtick_labels:
        #     label.set_fontsize(20)
        #
        # for median, label in zip(boxplot['medians'], ['HistoGene']):  # , 'VGE'
        #     median_value = median.get_ydata()[0]
        #     x_position = median.get_xdata()[0]
        #     ax.text(x_position + 0.15, median_value + 0.3, f'{median_value:.4f}', ha='center', va='bottom', fontsize=20,
        #             color='black')
        # plt.savefig(save_fig_path + "pcc.pdf", format='pdf', bbox_inches="tight")

        break
elif dataset == "T2":
    tag = f'T2_Mouse_brain'
    path1 = f'./T2_HistoGene/Mouse_brain/file_tmp'
    dataset_path = f'D:\dataset\T2'

    save_fig_path = rf"C:\Users\DELL\Desktop\学术垃圾\Experiment\HistoGene\dataset_T2/"
    save_path = f'./T2_HistoGene/'

    os.makedirs(save_fig_path, exist_ok=True)
    model = HisToGene.load_from_checkpoint("model/last_train_" + tag + ".ckpt", n_layers=8,
                                           n_genes=1000, learning_rate=1e-5)
    device = torch.device('cuda')

    dataset = ViT_SKIN(dataset=dataset, dataset_path=dataset_path, path1=path1, train=False, section_id=None)
    adata_sample = dataset.adata_sample
    adata = dataset.adata_copy
    test_loader = DataLoader(dataset, batch_size=1)
    adata_pred, adata_truth = model_predict(model, test_loader, device=device)

    adata_pred.obsm["coord"] = adata.obsm["coord"]
    use_gene = np.loadtxt(path1 + '/used_gene.txt', dtype=str)
    adata_pred.var.index = use_gene
    adata_pred.obs = adata.obs

    pr_stage = np.zeros(shape=(adata_pred.shape[1]))
    mse_values = np.zeros(adata_pred.shape[1])
    mae_values = np.zeros(adata_pred.shape[1])
    used_gene = adata_pred.var.index

    for it in tqdm(range(adata_pred.shape[1])):
        pr_stage[it] = \
            pearsonr(adata_pred[:, used_gene[it]].X.toarray().squeeze(), adata[:, used_gene[it]].X.toarray().squeeze())[
                0]
        mse_values[it] = mean_squared_error(adata[:, used_gene[it]].X.toarray().squeeze(),
                                            adata_pred[:, used_gene[it]].X.toarray().squeeze())
        mae_values[it] = mean_absolute_error(adata[:, used_gene[it]].X.toarray().squeeze(),
                                             adata_pred[:, used_gene[it]].X.toarray().squeeze())

    mask = ~np.isnan(pr_stage)
    pr_stage_n = pr_stage[mask]
    np.savetxt(save_fig_path + "/pcc.txt", pr_stage_n)
    print("avg PCC: ", np.mean(pr_stage_n))

    mse_values_filtered = mse_values[~np.isnan(mse_values)]
    mae_values_filtered = mae_values[~np.isnan(mae_values)]
    np.savetxt(save_fig_path + '/mse_values.txt', mse_values_filtered, fmt='%f')
    np.savetxt(save_fig_path + '/mae_values.txt', mae_values_filtered, fmt='%f')
    print("AVG MSE:", np.mean(mse_values_filtered))
    print("AVG MAE:", np.mean(mae_values_filtered))
    show_gene = ["Ttr", "Prkcd", "Nrgn", "Pmch", "Mbp"]
    sc.set_figure_params(dpi=80, figsize=(2.8, 3))
    sc.pl.embedding(adata_pred, basis="coord", color=show_gene, s=30, show=False)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)
    plt.savefig(save_fig_path + "GeneRecovery.pdf", format='pdf', bbox_inches="tight")

elif dataset == "IDC":
    tag = f'IDC'
    path1 = f'./IDC/file_tmp'
    dataset_path = f'D:\dataset\T2'

    save_fig_path = rf"C:\Users\DELL\Desktop\学术垃圾\Experiment\HistoGene\IDC/"
    save_path = f'./IDC/'

    os.makedirs(save_fig_path, exist_ok=True)
    model = HisToGene.load_from_checkpoint("model/last_train_" + tag + ".ckpt", n_layers=8,
                                           n_genes=1000, learning_rate=1e-5)
    device = torch.device('cuda')

    dataset = ViT_SKIN(dataset=dataset, dataset_path=dataset_path, path1=path1, train=False, section_id=None)
    adata_sample = dataset.adata_sample
    adata = dataset.adata_copy
    test_loader = DataLoader(dataset, batch_size=1)
    adata_pred, adata_truth = model_predict(model, test_loader, device=device)

    adata_pred.obsm["coord"] = adata.obsm["coord"]
    use_gene = np.loadtxt(path1 + '/used_gene.txt', dtype=str)
    adata_pred.var.index = use_gene
    adata_pred.obs = adata.obs

    pr_stage = np.zeros(shape=(adata_pred.shape[1]))
    mse_values = np.zeros(adata_pred.shape[1])
    mae_values = np.zeros(adata_pred.shape[1])
    used_gene = adata_pred.var.index

    for it in tqdm(range(adata_pred.shape[1])):
        pr_stage[it] = \
            pearsonr(adata_pred[:, used_gene[it]].X.toarray().squeeze(), adata[:, used_gene[it]].X.toarray().squeeze())[
                0]
        mse_values[it] = mean_squared_error(adata[:, used_gene[it]].X.toarray().squeeze(),
                                            adata_pred[:, used_gene[it]].X.toarray().squeeze())
        mae_values[it] = mean_absolute_error(adata[:, used_gene[it]].X.toarray().squeeze(),
                                             adata_pred[:, used_gene[it]].X.toarray().squeeze())

    mask = ~np.isnan(pr_stage)
    pr_stage_n = pr_stage[mask]
    np.savetxt(save_fig_path + "/pcc.txt", pr_stage_n)
    print("avg PCC: ", np.mean(pr_stage_n))

    mse_values_filtered = mse_values[~np.isnan(mse_values)]
    mae_values_filtered = mae_values[~np.isnan(mae_values)]
    np.savetxt(save_fig_path + '/mse_values.txt', mse_values_filtered, fmt='%f')
    np.savetxt(save_fig_path + '/mae_values.txt', mae_values_filtered, fmt='%f')
    print("AVG MSE:", np.mean(mse_values_filtered))
    print("AVG MAE:", np.mean(mae_values_filtered))
    show_gene = ["Ttr", "Prkcd", "Nrgn", "Pmch", "Mbp"]
    sc.set_figure_params(dpi=80, figsize=(2.8, 3))
    sc.pl.embedding(adata_pred, basis="coord", color=show_gene, s=30, show=False)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)
    plt.savefig(save_fig_path + "GeneRecovery.pdf", format='pdf', bbox_inches="tight")