import os
import random
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torch.utils.data import DataLoader
from dataset import T1, model_predict
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import scanpy as sc
import time

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
def loss_function(outputs, labels):
    criterion = nn.SmoothL1Loss()
    num_gene = outputs.shape[1]

    loss = 0

    for i in range(num_gene):
        loss += criterion(outputs[:, i], labels[:, i]) / num_gene

    return loss

section_list = ["151507", "151508", "151509", "151510", "151669", "151670", "151671", "151672", "151673", "151674", "151675",
                    "151676"]

seed = 0
set_seed(seed)
for section_id in section_list:
    start_time = time.time()
    save_fig_path = rf"C:\Users\DELL\Desktop\学术垃圾\Experiment\DeepSpace\dataset_T1/{section_id}/"
    path1 = f'./T1_DeepSpace/{section_id}/file_tmp'
    os.makedirs(save_fig_path, exist_ok=True)
    os.makedirs(path1, exist_ok=True)
    net = torchvision.models.vgg16(pretrained=True)
    # change the last unit of VGG16
    net.classifier[6] = nn.Linear(in_features=4096, out_features=1000)

    # net.train()
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    net = net.to(device)
    optims = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)

    # section_id = "151507"
    dataset = T1(train=True, section_id=section_id)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)

    train_epoch = 500
    with tqdm(range(train_epoch), total=train_epoch, desc='Epochs') as epoch:
        for j in epoch:
            train_re_loss = []

            for patch, pos, exp in train_loader:
                patch = patch.to(torch.float32).to(device)
                exp = exp.to(torch.float32).to(device)
                patch = patch.permute(0, 3, 1, 2)
                optims.zero_grad()

                xrecon = net(patch)
                recon_loss = loss_function(xrecon, exp).to(device)

                recon_loss.backward()
                optims.step()
                train_re_loss.append(recon_loss.item())

            epoch_info = 'recon_loss: %.5f' % \
                         (torch.mean(torch.FloatTensor(train_re_loss))),
            epoch.set_postfix_str(epoch_info)

    torch.save(net, f"./model/{section_id}.pth")
    end_time = time.time()
    elapsed_time = end_time - start_time

    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    time_str = f"{int(hours)}小时 {int(minutes)}分 {int(seconds)}秒"

    print(f"Total training time: {time_str}")

    with open(path1 + f"/training_time_{section_id}.txt", "w", encoding='utf-8') as file:
        file.write(f"Total training time: {time_str}\n")



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


    # sc.set_figure_params(dpi=100, figsize=(3, 4))
    # pr_stage_common = np.loadtxt(save_fig_path + "/pcc.txt")
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
    # ax.set_xticklabels(['STNet'])  # , 'VGE'
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
    # for median, label in zip(boxplot['medians'], ['DeepSpace']):  # , 'VGE'
    #     median_value = median.get_ydata()[0]
    #     x_position = median.get_xdata()[0]
    #     ax.text(x_position + 0.15, median_value + 0.3, f'{median_value:.4f}', ha='center', va='bottom', fontsize=20,
    #             color='black')
    # plt.savefig(save_fig_path + "pcc.pdf", format='pdf', bbox_inches="tight")