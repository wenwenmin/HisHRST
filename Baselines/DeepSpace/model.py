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
    save_fig_path = rf"..\dataset_T1/{section_id}/"
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
