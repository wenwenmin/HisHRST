import torch
import os
import numpy as np
import pandas as pd
import scanpy as sc
# from utils import *
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from torchvision import transforms
from torch.utils.data import Dataset
from tqdm import tqdm

class T1(torch.utils.data.Dataset):
    """Some Information about T1"""

    def __init__(self, section_id, train=True, gene_list=None, ds=None, sr=False, aug=False, norm=False):
        super(T1, self).__init__()

        self.dir = f'D:\dataset\DLPFC/{section_id}'
        self.path1 = f'./T1_DeepSpace/{section_id}/file_tmp'
        os.makedirs(self.path1, exist_ok=True)
        self.r = 224 // 4
        self.section_id = section_id
        self.down_ratio = 0.5
        self.coord_sf = 77
        self.use_gene = []
        image_path = os.path.join(self.dir, f"spatial/{section_id}_full_image.tif")
        self.image = torch.Tensor(np.array(Image.open(image_path)))
        self.adata_copy = None
        self.sample_coor_df = None
        self.fill_coor_df = None
        self.sample_spatial_df = None
        self.fill_spatial_df = None
        self.sample_barcode = None
        self.sample_index = None

        self.train = train
        self.sr = sr
        self.aug = aug
        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.5, 0.5, 0.5),
            transforms.ToTensor()
        ])
        self.norm = norm
        if self.aug:
            self.image = self.transforms(self.image)

        print('Loading metadata...')
        self.fill_meta = self.get_meta(section_id)
        self.fill_meta_exp = self.fill_meta[self.use_gene].values
        self.fill_center = np.floor(self.fill_meta[['pixel_x', 'pixel_y']].values).astype(int)
        self.fill_loc = self.fill_meta[['x', 'y']].values

        self.sample_meta = self.fill_meta.loc[self.sample_barcode]
        self.sample_meta_exp = self.sample_meta[self.use_gene].values
        self.sample_center = np.floor(self.sample_meta[['pixel_x', 'pixel_y']].values).astype(int)
        self.sample_loc = self.sample_meta[['x', 'y']].values

        self.sample_patches, self.fill_patches = self.get_patches()
        # sample_patches_tensor = torch.stack(self.sample_patches)
        # print(sample_patches_tensor.shape)

        self.gene_set = list(self.use_gene)

        self.adata_sample = self.adata_copy[self.sample_barcode]

        if self.train:
            self.lengths = len(self.sample_patches)
        else:
            self.lengths = len(self.fill_patches)

    def __getitem__(self, index):
        i = index
        if self.train:
            centers = self.sample_center[i]
            exps = self.sample_meta_exp[i]
            loc = self.sample_loc[i]
        else:

            centers = self.fill_center[i]
            exps = self.fill_meta_exp[i]
            loc = self.fill_loc[i]
        positions = torch.LongTensor(loc)
        patch_dim = 3 * self.r * self.r * 4

        exps = torch.Tensor(exps)
        if self.train:
            patch = self.sample_patches[i]
            return patch, positions, exps
        else:
            patch = self.fill_patches[i]
            return patch, positions, exps, torch.Tensor(centers)

    def __len__(self):
        return self.lengths

    def get_patches(self):
        sample_patches = []
        fill_patches = []
        for i in range(len(self.sample_center)):
            y, x = self.sample_center[i]
            patch = np.array(self.image[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :])
            sample_patches.append(patch)
        for i in range(len(self.fill_center)):
            y, x = self.fill_center[i]
            patch = np.array(self.image[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :])
            fill_patches.append(patch)

        return np.array(sample_patches), np.array(fill_patches)

    def get_cnt(self, section_id):

        adata = sc.read_visium(path=self.dir, count_file=f'{section_id}_filtered_feature_bc_matrix.h5')
        adata.var_names_make_unique()
        adata.obsm["coord"] = adata.obs.loc[:, ['array_col', 'array_row']].to_numpy()
        Ann_df = pd.read_csv(os.path.join(self.dir, section_id + '_truth.txt'), sep='\t', header=None, index_col=0)
        Ann_df.columns = ['Ground Truth']
        adata.obs['layer'] = Ann_df.loc[adata.obs_names, 'Ground Truth']
        adata.uns['layer_colors'] = ['#1f77b4', '#ff7f0e', '#49b192', '#d62728', '#aa40fc', '#8c564b', '#e377c2']

        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=1000)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        self.adata_copy = adata.copy()
        if self.train:
            self.use_gene = np.array(self.adata_copy.var.index[self.adata_copy.var.highly_variable])
            np.savetxt(self.path1 + "/used_gene.txt", self.use_gene, fmt='%s')
        else:
            self.use_gene = np.loadtxt(self.path1 + "/used_gene.txt", dtype=str)

        exp_df = pd.DataFrame(self.adata_copy.X.todense(), index=adata.obs.index, columns=adata.var.index)

        return exp_df

    def get_pos(self):
        self.fill_coor_df = pd.DataFrame(self.adata_copy.obsm["coord"])
        self.fill_coor_df.index = self.adata_copy.obs.index
        self.fill_coor_df.columns = ["x", "y"]

        self.fill_spatial_df = pd.DataFrame(self.adata_copy.obsm['spatial'].copy())
        self.fill_spatial_df.index = self.adata_copy.obs.index
        self.fill_spatial_df.columns = ["pixel_x", "pixel_y"]
        if self.train:
            self.sample_index = np.random.choice(range(self.fill_coor_df.shape[0]),
                                                 size=round(self.down_ratio * self.fill_coor_df.shape[0]),
                                                 replace=False)

            self.sample_index = setToArray(set(self.sample_index))

            self.sample_coor_df = self.fill_coor_df.iloc[self.sample_index]
            self.sample_spatial_df = self.fill_spatial_df.iloc[self.sample_index]

            self.sample_barcode = self.fill_coor_df.index[self.sample_index]
            self.sample_spatial_df.index = self.adata_copy[self.sample_barcode].obs.index
            self.sample_spatial_df.columns = ["pixel_x", "pixel_y"]

            del_index = setToArray(set(range(self.fill_coor_df.shape[0])) - set(self.sample_index))
            np.savetxt(self.path1 + "/all_barcode.txt", self.adata_copy.obs.index, fmt='%s')
            np.savetxt(self.path1 + "/sample_index.txt", self.sample_index, fmt='%s')
            np.savetxt(self.path1 + "/del_index.txt", del_index, fmt='%s')
            np.savetxt(self.path1 + "/sample_barcode.txt", self.fill_coor_df.index[self.sample_index], fmt='%s')
            np.savetxt(self.path1 + "/del_barcode.txt", self.fill_coor_df.index[del_index], fmt='%s')
        else:
            self.sample_index = np.loadtxt(self.path1 + "/sample_index.txt", dtype=int)
            self.sample_barcode = np.loadtxt(self.path1 + "/sample_barcode.txt", dtype=str)

            self.sample_coor_df = self.fill_coor_df.iloc[self.sample_index]
            self.sample_spatial_df = self.fill_spatial_df.iloc[self.sample_index]

            self.sample_barcode = self.fill_coor_df.index[self.sample_index]
            self.sample_spatial_df.index = self.adata_copy[self.sample_barcode].obs.index
            self.sample_spatial_df.columns = ["pixel_x", "pixel_y"]

        return self.sample_coor_df, self.fill_coor_df, self.sample_spatial_df, self.fill_spatial_df

    def get_meta(self, section_id, gene_list=None):
        cnt = self.get_cnt(section_id)
        sample_coor_df, fill_coor_df, sample_spatial_df, fill_spatial_df = self.get_pos()
        meta = cnt.join(fill_coor_df, how='inner')
        fill_meta = meta.join(fill_spatial_df, how='inner')
        return fill_meta


class T2(torch.utils.data.Dataset):
    """Some Information about T1"""

    def __init__(self, train=True, gene_list=None, ds=None, sr=False, aug=False, norm=False):
        super(T2, self).__init__()

        self.dir = f'D:\dataset\MouseBrain'
        self.path1 = f'./T2_DeepSpace/file_tmp'
        os.makedirs(self.path1, exist_ok=True)
        self.r = 224 // 4
        self.down_ratio = 0.5
        self.coord_sf = 77
        self.use_gene = []
        image_path = os.path.join(self.dir, f"V1_Adult_Mouse_Brain_Coronal_Section_1_image.tif")
        image_array = np.array(Image.open(image_path).convert("RGB"))
        self.image = torch.Tensor(image_array)
        self.adata_copy = None
        self.sample_coor_df = None
        self.fill_coor_df = None
        self.sample_spatial_df = None
        self.fill_spatial_df = None
        self.sample_barcode = None
        self.sample_index = None

        self.train = train
        self.sr = sr
        self.aug = aug
        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.5, 0.5, 0.5),
            transforms.ToTensor()
        ])
        self.norm = norm
        if self.aug:
            self.image = self.transforms(self.image)

        print('Loading metadata...')
        self.fill_meta = self.get_meta()
        self.fill_meta_exp = self.fill_meta[self.use_gene].values
        self.fill_center = np.floor(self.fill_meta[['pixel_x', 'pixel_y']].values).astype(int)
        self.fill_loc = self.fill_meta[['x', 'y']].values

        self.sample_meta = self.fill_meta.loc[self.sample_barcode]
        self.sample_meta_exp = self.sample_meta[self.use_gene].values
        self.sample_center = np.floor(self.sample_meta[['pixel_x', 'pixel_y']].values).astype(int)
        self.sample_loc = self.sample_meta[['x', 'y']].values

        self.sample_patches, self.fill_patches = self.get_patches()
        # sample_patches_tensor = torch.stack(self.sample_patches)
        # print(sample_patches_tensor.shape)

        self.gene_set = list(self.use_gene)

        self.adata_sample = self.adata_copy[self.sample_barcode]

        if self.train:
            self.lengths = len(self.sample_patches)
        else:
            self.lengths = len(self.fill_patches)

    def __getitem__(self, index):
        i = index
        if self.train:
            centers = self.sample_center[i]
            exps = self.sample_meta_exp[i]
            loc = self.sample_loc[i]
        else:

            centers = self.fill_center[i]
            exps = self.fill_meta_exp[i]
            loc = self.fill_loc[i]
        positions = torch.LongTensor(loc)
        patch_dim = 3 * self.r * self.r * 4

        exps = torch.Tensor(exps)
        if self.train:
            patch = self.sample_patches[i]
            return patch, positions, exps
        else:
            patch = self.fill_patches[i]
            return patch, positions, exps, torch.Tensor(centers)

    def __len__(self):
        return self.lengths

    def get_patches(self):
        sample_patches = []
        fill_patches = []
        for i in range(len(self.sample_center)):
            y, x = self.sample_center[i]
            patch = np.array(self.image[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :])
            sample_patches.append(patch)
        for i in range(len(self.fill_center)):
            y, x = self.fill_center[i]
            patch = np.array(self.image[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :])
            fill_patches.append(patch)

        return np.array(sample_patches), np.array(fill_patches)

    def get_cnt(self):

        adata = sc.read_visium(path=self.dir, count_file=f'filtered_feature_bc_matrix.h5')
        adata.var_names_make_unique()
        adata.obsm["coord"] = adata.obs.loc[:, ['array_col', 'array_row']].to_numpy()

        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=1000)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        self.adata_copy = adata.copy()
        if self.train:
            self.use_gene = np.array(self.adata_copy.var.index[self.adata_copy.var.highly_variable])
            np.savetxt(self.path1 + "/used_gene.txt", self.use_gene, fmt='%s')
        else:
            self.use_gene = np.loadtxt(self.path1 + "/used_gene.txt", dtype=str)

        exp_df = pd.DataFrame(self.adata_copy.X.todense(), index=adata.obs.index, columns=adata.var.index)

        return exp_df

    def get_pos(self):
        self.fill_coor_df = pd.DataFrame(self.adata_copy.obsm["coord"])
        self.fill_coor_df.index = self.adata_copy.obs.index
        self.fill_coor_df.columns = ["x", "y"]

        self.fill_spatial_df = pd.DataFrame(self.adata_copy.obsm['spatial'].copy())
        self.fill_spatial_df.index = self.adata_copy.obs.index
        self.fill_spatial_df.columns = ["pixel_x", "pixel_y"]
        if self.train:
            self.sample_index = np.random.choice(range(self.fill_coor_df.shape[0]),
                                                 size=round(self.down_ratio * self.fill_coor_df.shape[0]),
                                                 replace=False)

            self.sample_index = setToArray(set(self.sample_index))

            self.sample_coor_df = self.fill_coor_df.iloc[self.sample_index]
            self.sample_spatial_df = self.fill_spatial_df.iloc[self.sample_index]

            self.sample_barcode = self.fill_coor_df.index[self.sample_index]
            self.sample_spatial_df.index = self.adata_copy[self.sample_barcode].obs.index
            self.sample_spatial_df.columns = ["pixel_x", "pixel_y"]

            del_index = setToArray(set(range(self.fill_coor_df.shape[0])) - set(self.sample_index))
            np.savetxt(self.path1 + "/all_barcode.txt", self.adata_copy.obs.index, fmt='%s')
            np.savetxt(self.path1 + "/sample_index.txt", self.sample_index, fmt='%s')
            np.savetxt(self.path1 + "/del_index.txt", del_index, fmt='%s')
            np.savetxt(self.path1 + "/sample_barcode.txt", self.fill_coor_df.index[self.sample_index], fmt='%s')
            np.savetxt(self.path1 + "/del_barcode.txt", self.fill_coor_df.index[del_index], fmt='%s')
        else:
            self.sample_index = np.loadtxt(self.path1 + "/sample_index.txt", dtype=int)
            self.sample_barcode = np.loadtxt(self.path1 + "/sample_barcode.txt", dtype=str)

            self.sample_coor_df = self.fill_coor_df.iloc[self.sample_index]
            self.sample_spatial_df = self.fill_spatial_df.iloc[self.sample_index]

            self.sample_barcode = self.fill_coor_df.index[self.sample_index]
            self.sample_spatial_df.index = self.adata_copy[self.sample_barcode].obs.index
            self.sample_spatial_df.columns = ["pixel_x", "pixel_y"]

        return self.sample_coor_df, self.fill_coor_df, self.sample_spatial_df, self.fill_spatial_df

    def get_meta(self, gene_list=None):
        cnt = self.get_cnt()
        sample_coor_df, fill_coor_df, sample_spatial_df, fill_spatial_df = self.get_pos()
        meta = cnt.join(fill_coor_df, how='inner')
        fill_meta = meta.join(fill_spatial_df, how='inner')
        return fill_meta

import numpy as np

def setToArray(
        setInput,
        dtype='int64'
):
    """ This function transfer set to array.
        Args:
            setInput: set need to be trasnfered to array.
            dtype: data type.

        Return:
            arrayOutput: trasnfered array.
    """
    arrayOutput = np.zeros(len(setInput), dtype=dtype)
    index = 0
    for every in setInput:
        arrayOutput[index] = every
        index += 1
    return arrayOutput

import anndata as ann
def model_predict(model, test_loader, adata=None, attention=True, device=torch.device('cpu')):
    model.eval()
    model = model.to(device)
    preds = None
    with torch.no_grad():
        for patch, position, exp, center in tqdm(test_loader):

            patch, position = patch.to(device), position.to(device)
            patch = patch.permute(0, 3, 1, 2)
            pred = model(patch)
            if preds is None:
                preds = pred
                # pos = position.squeeze()
                ct = center
                gt = exp
            else:
                pred = pred
                center = center
                exp = exp
                # position = position.squeeze()

                preds = torch.cat((preds, pred), dim=0)
                ct = torch.cat((ct, center), dim=0)
                gt = torch.cat((gt, exp), dim=0)
                # pos = torch.cat((pos, position), dim=0)

    preds = preds.cpu().squeeze().numpy()
    ct = ct.cpu().squeeze().numpy()
    gt = gt.cpu().squeeze().numpy()
    print(preds.shape)
    print(ct.shape)
    print(gt.shape)

    adata = ann.AnnData(preds)
    adata.obsm['spatial'] = ct
    # adata.obsm['coord'] = pos

    adata_gt = ann.AnnData(gt)
    adata_gt.obsm['spatial'] = ct
    # adata_gt.obsm['coord'] = pos
    return adata, adata_gt