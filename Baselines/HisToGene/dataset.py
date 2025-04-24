import glob
from PIL import Image
import scprep as scp
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import pandas as pd
import numpy as np
import torch
import scanpy as sc
import scipy.sparse as sp
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import os
from utils import *

from collections import defaultdict as dfd
class ViT_HER2ST(torch.utils.data.Dataset):
    """Some Information about HER2ST"""

    def __init__(self, train=True, gene_list=None, ds=None, sr=False, name="B1"):
        super(ViT_HER2ST, self).__init__()

        self.cnt_dir = r"D:\dataset\Her2st\data\ST-cnts"
        self.img_dir = r"D:\dataset\Her2st\data\ST-imgs"
        self.pos_dir = r"D:\dataset\Her2st\data\ST-spotfiles"
        self.lbl_dir = r'D:\dataset\Her2st\data\ST-pat\lbl'
        self.r = 224 // 4
        self.path1 = rf"./BC_ST/{name}"
        os.makedirs(self.path1, exist_ok=True)

        self.down_ratio = 0.5
        gene_list = list(np.load('D:\dataset\Her2st\data/her_hvg_cut_1000.npy', allow_pickle=True))
        self.gene_list = gene_list
        self.train = train
        self.sr = sr

        if train:
            self.names = name
        else:
            self.names = name

        print('Loading imgs...')
        self.img_dict = {name: torch.Tensor(np.array(self.get_img(self.names)))}

        print('Loading metadata...')
        self.meta_dict = {name: self.get_meta(self.names)}

        self.gene_set = list(gene_list)
        self.fill_exp_dict = {
            i: scp.transform.log(scp.normalize.library_size_normalize(m["fill_meta"][self.gene_set].values)) for i, m in
            self.meta_dict.items()}
        self.sample_exp_dict = {
            i: scp.transform.log(scp.normalize.library_size_normalize(m["sample_meta"][self.gene_set].values)) for i, m
            in
            self.meta_dict.items()}
        self.sample_center_dict = {i: np.floor(m["sample_meta"][['pixel_x', 'pixel_y']].values).astype(int) for i, m in
                                   self.meta_dict.items()}
        self.fill_center_dict = {i: np.floor(m["fill_meta"][['pixel_x', 'pixel_y']].values).astype(int) for i, m in
                                 self.meta_dict.items()}
        self.sample_loc_dict = {i: m["sample_meta"][['x', 'y']].values for i, m in self.meta_dict.items()}
        self.fill_loc_dict = {i: m["fill_meta"][['x', 'y']].values for i, m in self.meta_dict.items()}

        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = {0:"B1"}

        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.5, 0.5, 0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=180),
            transforms.ToTensor()
        ])


    def filter_helper(self):
        a = np.zeros(len(self.gene_list))
        n = 0
        for i, exp in self.exp_dict.items():
            n += exp.shape[0]
            exp[exp > 0] = 1
            for j in range((len(self.gene_list))):
                a[j] += np.sum(exp[:, j])

    def __getitem__(self, index):
        if self.train:
            i = index
            im = self.img_dict[self.id2name[i]]
            im = im.permute(1, 0, 2)
            # im = torch.Tensor(np.array(self.im))
            exps = self.sample_exp_dict[self.id2name[i]]
            centers = self.sample_center_dict[self.id2name[i]]
            loc = self.sample_loc_dict[self.id2name[i]]
            positions = torch.LongTensor(loc)
            patch_dim = 3 * self.r * self.r * 4
        else:
            i = index
            im = self.img_dict[self.id2name[i]]
            im = im.permute(1, 0, 2)
            # im = torch.Tensor(np.array(self.im))
            exps = self.fill_exp_dict[self.id2name[i]]
            centers = self.fill_center_dict[self.id2name[i]]
            loc = self.fill_loc_dict[self.id2name[i]]
            positions = torch.LongTensor(loc)
            patch_dim = 3 * self.r * self.r * 4

        if self.sr:
            centers = torch.LongTensor(centers)
            max_x = centers[:, 0].max().item()
            max_y = centers[:, 1].max().item()
            min_x = centers[:, 0].min().item()
            min_y = centers[:, 1].min().item()
            r_x = (max_x - min_x) // 30
            r_y = (max_y - min_y) // 30

            centers = torch.LongTensor([min_x, min_y]).view(1, -1)
            positions = torch.LongTensor([0, 0]).view(1, -1)
            x = min_x
            y = min_y

            while y < max_y:
                x = min_x
                while x < max_x:
                    centers = torch.cat((centers, torch.LongTensor([x, y]).view(1, -1)), dim=0)
                    positions = torch.cat((positions, torch.LongTensor([x // r_x, y // r_y]).view(1, -1)), dim=0)
                    x += 56
                y += 56

            centers = centers[1:, :]
            positions = positions[1:, :]

            n_patches = len(centers)
            patches = torch.zeros((n_patches, patch_dim))
            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :]
                patches[i] = patch.flatten()

            return patches, positions, centers

        else:
            n_patches = len(centers)
            patches = torch.zeros((n_patches, patch_dim))
            exps = torch.Tensor(exps)

            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :]
                patches[i] = patch.flatten()

            if self.train:
                return patches, positions, exps
            else:
                return patches, positions, exps, torch.Tensor(centers)

    def __len__(self):
        return len(self.fill_exp_dict)

    def get_img(self, name):
        pre = self.img_dir + '/' + name[0] + '/' + name
        fig_name = os.listdir(pre)[0]
        path = pre + '/' + fig_name
        im = Image.open(path)
        return im

    def get_cnt(self, name):
        path = self.cnt_dir + '/' + name + '.tsv'
        df = pd.read_csv(path, sep='\t', index_col=0)

        return df

    def get_pos(self, name):
        path = self.pos_dir + '/' + name + '_selection.tsv'
        df = pd.read_csv(path, sep='\t')
        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i]) + 'x' + str(y[i]))
        df['id'] = id

        self.fill_df = df[['id', 'x', 'y', "pixel_x", "pixel_y"]]
        self.fill_df.index = df.index
        self.fill_df.columns = ['id', 'x', 'y', "pixel_x", "pixel_y"]

        if self.train:
            self.sample_index = np.random.choice(range(self.fill_df.shape[0]),
                                                 size=round(self.down_ratio * self.fill_df.shape[0]),
                                                 replace=False)

            self.sample_index = setToArray(set(self.sample_index))

            self.sample_df = self.fill_df.iloc[self.sample_index]

            del_index = setToArray(set(range(self.fill_df.shape[0])) - set(self.sample_index))
            np.savetxt(self.path1 + "/all_barcode.txt", df.index, fmt='%s')
            np.savetxt(self.path1 + "/sample_index.txt", self.sample_index, fmt='%s')
            np.savetxt(self.path1 + "/del_index.txt", del_index, fmt='%s')
        else:
            self.sample_index = np.loadtxt(self.path1 + "/sample_index.txt", dtype=int)
            self.sample_df = self.fill_df.iloc[self.sample_index]

        return self.sample_df, self.fill_df

    def get_lbl(self, name):
        # path = self.pos_dir+'/'+name+'_selection.tsv'
        path = self.lbl_dir + '/' + name + '_labeled_coordinates.tsv'
        df = pd.read_csv(path, sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i]) + 'x' + str(y[i]))
        df['id'] = id
        df.drop('pixel_x', inplace=True, axis=1)
        df.drop('pixel_y', inplace=True, axis=1)
        df.drop('x', inplace=True, axis=1)
        df.drop('y', inplace=True, axis=1)

        return df

    def get_meta(self, name, gene_list=None):
        cnt = self.get_cnt(name)
        sample_df, fill_df = self.get_pos(name)
        # pos = self.get_pos(name)
        fill_meta = cnt.join((fill_df.set_index('id')))
        sample_meta = cnt.merge(sample_df, left_index=True, right_on='id')
        sample_meta.drop('id', inplace=True, axis=1)

        return {"sample_meta": sample_meta, "fill_meta": fill_meta}

    def get_overlap(self, meta_dict, gene_list):
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set & set(i.columns)
        return list(gene_set)


class ViT_SKIN(torch.utils.data.Dataset):
    """Some Information about ViT_SKIN"""

    def __init__(self,dataset, dataset_path, path1, section_id, train=True, gene_list=None, ds=None, sr=False, aug=False, norm=False):
        super(ViT_SKIN, self).__init__()
        self.dataset = dataset
        self.dir = dataset_path
        self.path1 = path1
        os.makedirs(self.path1, exist_ok=True)
        self.r = 224 // 4
        self.section_id = section_id
        self.down_ratio = 0.5
        self.coord_sf = 77
        self.use_gene = []
        if dataset == "T2":
            image_path = os.path.join(self.dir, f"V1_Adult_Mouse_Brain_Coronal_Section_1_image.tif")
            # image_array = np.array(Image.open(image_path).convert("RGB"))
            # image_array = image_array.astype(np.float32)
            self.image = torch.Tensor(np.array(Image.open(image_path)).astype(np.float32))
            print(self.image.shape)
        else:
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

        self.sample_train_exp = np.array_split(self.sample_meta_exp, np.ceil(len(self.sample_meta_exp) / 200))

        self.sample_train_patches = np.array_split(self.sample_patches, np.ceil(len(self.sample_patches) / 200))
        self.sample_train_pos = np.array_split(self.sample_loc, np.ceil(len(self.sample_loc) / 200))
        self.sample_train_center = np.array_split(self.sample_center, np.ceil(len(self.sample_center) / 200))

        self.fill_test_exp = np.array_split(self.fill_meta_exp, np.ceil(len(self.fill_meta_exp) / 200))
        self.fill_test_patches = np.array_split(self.fill_patches, np.ceil(len(self.fill_patches) / 200))
        self.fill_test_pos = np.array_split(self.fill_loc, np.ceil(len(self.fill_loc) / 200))
        self.fill_test_center = np.array_split(self.fill_center, np.ceil(len(self.fill_center) / 200))

        self.adata_sample = self.adata_copy[self.sample_barcode]

        if self.train:
            self.lengths = len(self.sample_train_patches)
        else:
            self.lengths = len(self.fill_test_patches)

        if self.sr:
            self.lengths = len(self.fill_test_patches)

    def __getitem__(self, index):
        i = index
        if self.train:
            centers = self.sample_train_center[i]
            exps = self.sample_train_exp[i]
            loc = self.sample_train_pos[i]
        else:

            centers = self.fill_test_center[i]
            exps = self.fill_test_exp[i]
            loc = self.fill_test_pos[i]
        positions = torch.LongTensor(loc)
        patch_dim = 3 * self.r * self.r * 4



        if self.sr:
            centers = torch.LongTensor(centers)
            max_x = centers[:, 0].max().item()
            max_y = centers[:, 1].max().item()
            min_x = centers[:, 0].min().item()
            min_y = centers[:, 1].min().item()
            r_x = (max_x - min_x) // 30
            r_y = (max_y - min_y) // 30

            centers = torch.LongTensor([min_x, min_y]).view(1, -1)
            positions = torch.LongTensor([0, 0]).view(1, -1)
            x = min_x
            y = min_y

            while y < max_y:
                x = min_x
                while x < max_x:
                    centers = torch.cat((centers, torch.LongTensor([x, y]).view(1, -1)), dim=0)
                    positions = torch.cat((positions, torch.LongTensor([x // r_x, y // r_y]).view(1, -1)), dim=0)
                    x += 56
                y += 56

            centers = centers[1:, :]
            positions = positions[1:, :]

            n_patches = len(centers)
            patches = torch.zeros((n_patches, patch_dim))
            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :]
                patches[i] = patch.flatten()

            return patches, positions, centers

        else:
            exps = torch.Tensor(exps)
            if self.train:
                patch = self.sample_train_patches[i]
                return patch, positions, exps
            else:
                patch = self.fill_test_patches[i]
                return patch, positions, exps, torch.Tensor(centers)

    def __len__(self):
        return self.lengths

    def get_patches(self):
        sample_patches = []
        fill_patches = []
        for i in range(len(self.sample_center)):
            y, x = self.sample_center[i]
            patch = np.array(self.image[(x - self.r):(x + self.r), (y - self.r):(y + self.r)])
            sample_patches.append(patch)
        for i in range(len(self.fill_center)):
            y, x = self.fill_center[i]
            patch = np.array(self.image[(x - self.r):(x + self.r), (y - self.r):(y + self.r)])
            fill_patches.append(patch)

        return np.array(sample_patches), np.array(fill_patches)

    def get_cnt(self, section_id):
        if self.dataset == "T1":
            adata = sc.read_visium(path=self.dir, count_file=f'{section_id}_filtered_feature_bc_matrix.h5')
            adata.var_names_make_unique()
            adata.obsm["coord"] = adata.obs.loc[:, ['array_col', 'array_row']].to_numpy()
            Ann_df = pd.read_csv(os.path.join(self.dir, section_id + '_truth.txt'), sep='\t', header=None, index_col=0)
            Ann_df.columns = ['Ground Truth']
            adata.obs['layer'] = Ann_df.loc[adata.obs_names, 'Ground Truth']
            adata.uns['layer_colors'] = ['#1f77b4', '#ff7f0e', '#49b192', '#d62728', '#aa40fc', '#8c564b', '#e377c2']

        else:
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

    def get_meta(self, section_id, gene_list=None):
        cnt = self.get_cnt(section_id)
        sample_coor_df, fill_coor_df, sample_spatial_df, fill_spatial_df = self.get_pos()
        meta = cnt.join(fill_coor_df, how='inner')
        fill_meta = meta.join(fill_spatial_df, how='inner')
        return fill_meta



class ViT_Anndata(torch.utils.data.Dataset):
    def __init__(self, adata_dict, train_set, gene_list, train=True, r=4, norm=False, flatten=True, ori=True, adj=True,
                 prune='NA', neighs=4):
        super(ViT_Anndata, self).__init__()

        self.r = 224 // r

        names = list(adata_dict.keys())

        self.ori = ori
        self.adj = adj
        self.norm = norm
        self.train = train
        self.flatten = flatten
        self.gene_list = gene_list
        samples = names
        tr_names = train_set
        te_names = list(set(samples) - set(tr_names))

        if train:
            self.names = tr_names
        else:
            self.names = te_names

        print("Eval set: ", te_names)

        self.adata_dict = {k: v for k, v in adata_dict.items() if k in self.names}

        print('Loading imgs...')
        #        self.img_dict = {i:torch.Tensor(np.array(self.get_img(i))) for i in self.names}
        path_img_dict = {}
        self.img_dict = {}
        for name in self.names:
            name_orig = list(self.adata_dict[name].uns['spatial'])[0]
            path = self.adata_dict[name].uns["spatial"][name_orig]["metadata"]["source_image_path"]

            if path in path_img_dict:
                self.img_dict[name] = path_img_dict[path]
            else:
                path_img_dict[path] = torch.Tensor(np.array(read_tiff(path)))
                self.img_dict[name] = path_img_dict[path]

            # self.img_dict[name] = torch.Tensor(np.array(self.img_dict[name]))
            self.img_dict[name] = self.img_dict[name]

        del path_img_dict

        self.gene_set = list(gene_list)
        if self.norm:
            self.exp_dict = {
                i: sc.pp.scale(
                    scp.transform.log(scp.normalize.library_size_normalize(m.to_df()[self.gene_set].values))).astype(
                    np.float64)
                for i, m in self.adata_dict.items()
            }
        else:
            self.exp_dict = {
                i: scp.transform.log(scp.normalize.library_size_normalize(m.to_df()[self.gene_set].values)).astype(
                    np.float64)
                for i, m in self.adata_dict.items()
            }
        if self.ori:
            self.ori_dict = {i: m.to_df()[self.gene_set].values.astype(np.float64) for i, m in self.adata_dict.items()}
            self.counts_dict = {}
            for i, m in self.ori_dict.items():
                n_counts = m.sum(1)
                sf = n_counts / np.median(n_counts)
                self.counts_dict[i] = sf.astype(np.float64)
        self.center_dict = {
            i: np.floor(m.obsm["spatial"].astype(np.int64)).astype(int)
            for i, m in self.adata_dict.items()
        }
        self.loc_dict = {i: m.obs[['array_col', 'array_row']].values for i, m in self.adata_dict.items()}
        if self.adj:
            self.adj_dict = {
                i: calcADJ(m, neighs, pruneTag=prune)
                for i, m in self.loc_dict.items()
            }
        self.patch_dict = dfd(lambda: None)
        self.lengths = [i.n_obs for i in self.adata_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(self.names))

    def filter_helper(self):
        a = np.zeros(len(self.gene_list))
        n = 0
        for i, exp in self.exp_dict.items():
            n += exp.shape[0]
            exp[exp > 0] = 1
            for j in range((len(self.gene_list))):
                a[j] += np.sum(exp[:, j])

    def __getitem__(self, index):
        ID = self.id2name[index]
        im = self.img_dict[ID].permute(1, 0, 2)

        exps = self.exp_dict[ID]
        if self.ori:
            oris = self.ori_dict[ID]
            sfs = self.counts_dict[ID]

        centers = self.center_dict[ID]
        loc = self.loc_dict[ID]

        positions = torch.LongTensor(loc)
        patch_dim = 3 * self.r * self.r * 4
        exps = torch.Tensor(exps)
        n_patches = len(centers)

        patches = torch.zeros((n_patches, patch_dim))
        exps = torch.Tensor(exps)

        for i in range(n_patches):
            center = centers[i]
            x, y = center
            patch = im[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :]
            patches[i] = patch.flatten()

        if self.train:
            return patches, positions, exps
        else:
            return patches, positions, exps, torch.Tensor(centers)

    def __len__(self):
        return len(self.exp_dict)

    def get_img(self, name):
        name_orig = list(self.adata_dict[name].uns['spatial'])[0]
        path = self.adata_dict[name].uns["spatial"][name_orig]["metadata"]["source_image_path"]
        im = read_tiff(path)
        return im

