import warnings

import torch
from tqdm import tqdm

from utils import *

warnings.filterwarnings('ignore')

MODEL_PATH = ''


def model_predict_5fold(model, test_loader, adata=None, attention=True, device=torch.device('cpu')):
    model.eval()
    model = model.to(device)
    preds = None

    all_preds = []
    all_gt = []
    all_ct = []

    with torch.no_grad():
        for patch, position, exp, center, adj in tqdm(test_loader):
            patch, position, adj = patch.to(device), position.to(device), adj.to(device)

            pred = model(patch, position, adj)

            pred = pred.squeeze(0)
            exp = exp.squeeze(0)
            center = center.squeeze(0)

            all_preds.append(pred.cpu().numpy())  # 转换为numpy并存储
            all_gt.append(exp.cpu().numpy())
            all_ct.append(center.cpu().numpy())
    adata_list = []
    adata_gt_list = []
    for preds, gt, ct in zip(all_preds, all_gt, all_ct):
        adata = ann.AnnData(preds)
        adata.obsm['spatial'] = ct
        adata_gt = ann.AnnData(gt)
        adata_gt.obsm['spatial'] = ct
        adata_list.append(adata)
        adata_gt_list.append(adata_gt)

    return adata_list, adata_gt_list
            # if preds is None:
            #     preds = pred.squeeze()
            #     ct = center
            #     gt = exp
            # else:
            #     preds = torch.cat((preds, pred), dim=0)
            #     ct = torch.cat((ct, center), dim=0)
            #     gt = torch.cat((gt, exp), dim=0)  #
    # preds = preds.cpu().squeeze().numpy()
    # ct = ct.cpu().squeeze().numpy()
    # gt = gt.cpu().squeeze().numpy()
    #
    # adata = ann.AnnData(preds)
    # adata.obsm['spatial'] = ct
    #
    # adata_gt = ann.AnnData(gt)
    # adata_gt.obsm['spatial'] = ct
    #
    # return adata, adata_gt



def model_predict(model, test_loader, adata=None, attention=True, device=torch.device('cpu')):
    model.eval()
    model = model.to(device)
    preds = None

    with torch.no_grad():
        for patch, position, exp, center, adj in tqdm(test_loader):
            patch, position, adj = patch.to(device), position.to(device), adj.to(device)

            pred = model(patch, position, adj)

            pred = pred.squeeze(0)
            exp = exp.squeeze(0)
            center = center.squeeze(0)

            if preds is None:
                preds = pred.squeeze()
                ct = center
                gt = exp
            else:
                preds = torch.cat((preds, pred), dim=0)
                ct = torch.cat((ct, center), dim=0)
                gt = torch.cat((gt, exp), dim=0)  #
    preds = preds.cpu().squeeze().numpy()
    ct = ct.cpu().squeeze().numpy()
    gt = gt.cpu().squeeze().numpy()

    adata = ann.AnnData(preds)
    adata.obsm['spatial'] = ct

    adata_gt = ann.AnnData(gt)
    adata_gt.obsm['spatial'] = ct

    return adata, adata_gt

def model_predict_visium(model, test_loader, device='cuda'):
    model = model.to(device)
    model.eval()
    preds = None
    ct = None
    gt = None
    loss = 0
    adatas, adata_gts = [], []
    with torch.no_grad():
        for patch, position, exp, center, adj in tqdm(test_loader):
            patch, position, adj = patch.to(device), position.to(device), adj.to(device)
            pred = model(patch, position, adj)[0]
            preds = pred.squeeze().cpu().numpy()
            ct = center.squeeze().cpu().numpy()
            gt = exp.squeeze().cpu().numpy()
            adata = ad.AnnData(preds)
            adata.obsm['spatial'] = ct
            adata_gt = ad.AnnData(gt)
            adata_gt.obsm['spatial'] = ct

            adatas.append(adata)
            adata_gts.append(adata_gt)
    adata = ad.concat(adatas)
    adata_gt = ad.concat(adata_gts)
    return adata, adata_gt


def sr_predict(model, test_loader, device=torch.device('cpu')):
    model.eval()
    model = model.to(device)
    preds = None
    with torch.no_grad():
        for patch, position, center in tqdm(test_loader):

            patch, position = patch.to(device), position.to(device)
            pred = model(patch, position)

            if preds is None:
                preds = pred.squeeze()
                ct = center
            else:
                preds = torch.cat((preds, pred), dim=0)
                ct = torch.cat((ct, center), dim=0)
    preds = preds.cpu().squeeze().numpy()
    ct = ct.cpu().squeeze().numpy()
    adata = ann.AnnData(preds)
    adata.obsm['spatial'] = ct

    return adata
