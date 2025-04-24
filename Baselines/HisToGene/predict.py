import torch
from torch.utils.data import DataLoader
from utils import *
from vismodel import HisToGene
import warnings
from dataset import ViT_HER2ST, ViT_SKIN
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings('ignore')

MODEL_PATH = ''


def model_predict(model, test_loader, adata=None, attention=True, device=torch.device('cpu')):
    model.eval()
    model = model.to(device)
    preds = None
    with torch.no_grad():
        for patch, position, exp, center in tqdm(test_loader):

            patch, position = patch.to(device), position.to(device)
            pred = model(patch, position)
            if preds is None:
                preds = pred.squeeze()
                pos = position.squeeze()
                ct = center.squeeze()
                gt = exp.squeeze()
            else:
                pred = pred.squeeze()
                center = center.squeeze()
                exp = exp.squeeze()
                position = position.squeeze()

                preds = torch.cat((preds, pred), dim=0)
                ct = torch.cat((ct, center), dim=0)
                gt = torch.cat((gt, exp), dim=0)
                pos = torch.cat((pos, position), dim=0)

    preds = preds.cpu().squeeze().numpy()
    ct = ct.cpu().squeeze().numpy()
    gt = gt.cpu().squeeze().numpy()
    pos = pos.cpu().squeeze().numpy()


    adata = ann.AnnData(preds)
    adata.obsm['spatial'] = ct

    adata.obsm['coord'] = pos

    adata_gt = ann.AnnData(gt)
    adata_gt.obsm['spatial'] = ct
    adata_gt.obsm['coord'] = pos
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
        for patch, position, exp, center in tqdm(test_loader):
            patch, position = patch.to(device), position.to(device)
            pred = model(patch, position)
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


def sr_predict(model, test_loader, attention=True, device=torch.device('cpu')):
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


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # for fold in [5,11,17,26]:
    for fold in range(12):
        # fold=30
        # tag = '-vit_skin_aug'
        # tag = '-cnn_her2st_785_32_cv'
        tag = '-vit_her2st_785_32_cv'
        # tag = '-cnn_skin_134_cv'
        # tag = '-vit_skin_134_cv'
        ds = 'HER2'
        # ds = 'Skin'

        print('Loading model ...')
        # model = STModel.load_from_checkpoint('model/last_train_'+tag+'.ckpt')
        # model = VitModel.load_from_checkpoint('model/last_train_'+tag+'.ckpt')
        # model = STModel.load_from_checkpoint("model/last_train_"+tag+'_'+str(fold)+".ckpt")
        model = STModel.load_from_checkpoint("model/last_train_" + tag + '_' + str(fold) + ".ckpt")
        model = model.to(device)
        # model = torch.nn.DataParallel(model)
        print('Loading data ...')

        # g = list(np.load('data/her_hvg_cut_1000.npy',allow_pickle=True))
        g = list(np.load('data/skin_hvg_cut_1000.npy', allow_pickle=True))

        # dataset = SKIN(train=False,ds=ds,fold=fold)
        dataset = ViT_HER2ST(train=False, mt=False, sr=True, fold=fold)
        # dataset = ViT_SKIN(train=False,mt=False,sr=False,fold=fold)
        # dataset = VitDataset(diameter=112,sr=True)

        test_loader = DataLoader(dataset, batch_size=16, num_workers=4)
        print('Making prediction ...')

        adata_pred, adata = model_predict(model, test_loader, attention=False)
        # adata_pred = sr_predict(model,test_loader,attention=True)

        adata_pred.var_names = g
        print('Saving files ...')
        adata_pred = comp_tsne_km(adata_pred, 4)
        # adata_pred = comp_umap(adata_pred)
        print(fold)
        print(adata_pred)

        adata_pred.write('processed/test_pred_' + ds + '_' + str(fold) + tag + '.h5ad')
        # adata_pred.write('processed/test_pred_sr_'+ds+'_'+str(fold)+tag+'.h5ad')

        # quit()


def get_R(data1, data2, dim=1, func=pearsonr):
    adata1 = data1.X
    adata2 = data2.X
    r1, p1 = [], []
    for g in range(data1.shape[dim]):
        if dim == 1:
            r, pv = func(adata1[:, g], adata2[:, g])
        elif dim == 0:
            r, pv = func(adata1[g, :], adata2[g, :])
        r1.append(r)
        p1.append(pv)
    r1 = np.array(r1)
    p1 = np.array(p1)
    return r1, p1


def calculate_mse(data1, data2, dim=1):
    adata1 = data1.X
    adata2 = data2.X
    mse_values = []
    mse = None
    for g in range(data1.shape[dim]):
        if dim == 1:
            mse = np.mean((adata1[:, g] - adata2[:, g]) ** 2)
        elif dim == 0:
            mse = np.mean((adata1[g, :] - adata2[g, :]) ** 2)
        mse_values.append(mse)

    return np.array(mse_values)


def calculate_mae(data1, data2, dim=1):
    adata1 = data1.X
    adata2 = data2.X
    mae_values = []
    mae = None
    for g in range(data1.shape[dim]):
        if dim == 1:
            mae = np.mean(np.abs(adata1[:, g] - adata2[:, g]))
        elif dim == 0:
            mae = np.mean(np.abs(adata1[g, :] - adata2[g, :]))
        mae_values.append(mae)

    return np.array(mae_values)


if __name__ == '__main__':
    main()
