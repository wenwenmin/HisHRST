import os
import pandas as pd
import torch
import numpy as np

from PIL import Image
import torchvision.transforms as transforms
import scprep as scp
import scanpy as sc
from sklearn.model_selection import KFold


class DATA_BRAIN(torch.utils.data.Dataset):
    def __init__(self, train=True, gene_list=None, ds=None, fold=0):
        super(DATA_BRAIN, self).__init__()
        self.dir = 'D:\dataset\DLPFC'
        self.r = 224 // 2

        sample_names = ['151507', '151508', '151509', '151510', '151669', '151670', '151671', '151672', '151673',
                        '151674', '151675', '151676']

        gene_list = list(np.load('common_highly_variable_genes.npy'))
        self.gene_list = gene_list
        self.train = train
        samples = sample_names
        te_names = [samples[fold]]
        tr_names = list(set(samples) - set(te_names))

        if train:
            names = tr_names
        else:
            names = te_names

        print('Loading imgs...')
        self.img_dict = {i: self.get_img(i) for i in names}
        print('Loading metadata...')
        self.meta_dict = {i: self.get_meta(i) for i in names}

        self.exp_dict = {i: scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_list].values)) for
                         i, m in self.meta_dict.items()}
        self.center_dict = {i: np.floor(m[['pixel_x', 'pixel_y']].values).astype(int) for i, m in
                            self.meta_dict.items()}
        self.loc_dict = {i: m[['x', 'y']].values for i, m in self.meta_dict.items()}
        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(names))

        self.transforms = transforms.Compose([
            # transforms.ColorJitter(0.5, 0.5, 0.5),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(degrees=180),
            # transforms.ToTensor()
            transforms.RandomChoice([
                transforms.RandomRotation((0, 0)),
                transforms.RandomRotation((90, 90)),
                transforms.RandomRotation((180, 180)),
                transforms.RandomRotation((270, 270))
            ]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        i = 0
        item = {}
        while index >= self.cumlen[i]:
            i += 1
        idx = index
        if i > 0:
            idx = index - self.cumlen[i - 1]
        exp = self.exp_dict[self.id2name[i]][idx]
        center = self.center_dict[self.id2name[i]][idx]
        loc = self.loc_dict[self.id2name[i]][idx]

        exp = torch.Tensor(exp)
        loc = torch.Tensor(loc)

        x, y = center
        patch = self.img_dict[self.id2name[i]].crop((x - self.r, y - self.r, x + self.r, y + self.r))
        if self.train:
            patch = self.transforms(patch)
        else:
            patch = transforms.ToTensor()(patch)
        if self.train:
            return patch, loc, exp
        else:
            return patch, loc, exp, torch.Tensor(center)

    def __len__(self):
        return self.cumlen[-1]

    def get_img(self, name):
        image_path = os.path.join(self.dir, f"{name}/spatial/{name}_full_image.tif")
        im = Image.open(image_path)
        return im

    def get_adata(self, section_id):
        adata = sc.read_visium(path=self.dir + f"/{section_id}",
                               count_file=f'{section_id}_filtered_feature_bc_matrix.h5')
        adata.var_names_make_unique()
        adata.obsm["coord"] = adata.obs.loc[:, ['array_col', 'array_row']].to_numpy()
        Ann_df = pd.read_csv(os.path.join(self.dir + f"/{section_id}", section_id + '_truth.txt'), sep='\t',
                             header=None, index_col=0)
        Ann_df.columns = ['Ground Truth']
        adata.obs['layer'] = Ann_df.loc[adata.obs_names, 'Ground Truth']
        adata.uns['layer_colors'] = ['#1f77b4', '#ff7f0e', '#49b192', '#d62728', '#aa40fc', '#8c564b', '#e377c2']

        return adata

    def get_pos(self, adata):
        spot_coord_df = pd.DataFrame(adata.obsm["coord"].copy())
        spot_coord_df.index = adata.obs.index
        spot_coord_df.columns = ["x", "y"]

        image_spatial_df = pd.DataFrame(adata.obsm['spatial'].copy())
        image_spatial_df.index = adata.obs.index
        image_spatial_df.columns = ["pixel_x", "pixel_y"]
        return spot_coord_df, image_spatial_df

    def get_meta(self, section_id, gene_list=None):
        adata = self.get_adata(section_id)
        spot_coord_df, image_spatial_df = self.get_pos(adata)
        exp_df = pd.DataFrame(adata.X.todense(), index=adata.obs.index, columns=adata.var.index)

        meta = exp_df.join(spot_coord_df, how='inner')
        fill_meta = meta.join(image_spatial_df, how='inner')
        return fill_meta


class HER2ST5FOLD(torch.utils.data.Dataset):
    """Some Information about HER2ST"""

    def __init__(self, train=True, gene_list=None, ds=None, fold=0, n_splits=5):
        super(HER2ST5FOLD, self).__init__()
        self.cnt_dir = 'D:\dataset\Her2st\data\ST-cnts'
        self.img_dir = 'D:\dataset\Her2st\data\ST-imgs'
        self.pos_dir = 'D:\dataset\Her2st\data\ST-spotfiles'
        self.lbl_dir = 'D:\dataset\Her2st\data\ST-pat'
        self.r = 224 // 2
        # gene_list = list(np.load('data/her_hvg.npy',allow_pickle=True))
        gene_list = list(np.load('data/her_hvg_cut_1000.npy', allow_pickle=True))
        self.gene_list = gene_list
        names = os.listdir(self.cnt_dir)
        names.sort()
        names = [i[:2] for i in names]
        self.train = train

        # samples = ['A1','B1','C1','D1','E1','F1','G2','H1']
        samples = names[1:33]
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(kf.split(samples))

        tr_idx, te_idx = splits[fold]
        tr_names = [samples[i] for i in tr_idx]
        te_names = [samples[i] for i in te_idx]

        # te_names = [samples[fold]]
        # tr_names = list(set(samples) - set(te_names))
        if train:
            names = tr_names
        else:

            names = te_names
        print('Loading imgs...')
        self.img_dict = {i: self.get_img(i) for i in names}
        print('Loading metadata...')
        self.meta_dict = {i: self.get_meta(i) for i in names}

        self.gene_set = list(gene_list)
        self.exp_dict = {i: scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)) for i, m in
                         self.meta_dict.items()}
        self.center_dict = {i: np.floor(m[['pixel_x', 'pixel_y']].values).astype(int) for i, m in
                            self.meta_dict.items()}
        self.loc_dict = {i: m[['x', 'y']].values for i, m in self.meta_dict.items()}

        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(names))

        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.5, 0.5, 0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=180),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        i = 0
        while index >= self.cumlen[i]:
            i += 1
        idx = index
        if i > 0:
            idx = index - self.cumlen[i - 1]

        exp = self.exp_dict[self.id2name[i]][idx]
        center = self.center_dict[self.id2name[i]][idx]
        loc = self.loc_dict[self.id2name[i]][idx]

        # if self.cls or self.train==False:

        exp = torch.Tensor(exp)
        loc = torch.Tensor(loc)

        x, y = center
        patch = self.img_dict[self.id2name[i]].crop((x - self.r, y - self.r, x + self.r, y + self.r))
        if self.train:
            patch = self.transforms(patch)
        else:
            patch = transforms.ToTensor()(patch)

        if self.train:
            return patch, loc, exp
        else:
            return patch, loc, exp, torch.Tensor(center)

    def __len__(self):
        return self.cumlen[-1]

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
        # path = self.pos_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path, sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i]) + 'x' + str(y[i]))
        df['id'] = id

        return df

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
        pos = self.get_pos(name)
        meta = cnt.join((pos.set_index('id')))
        self.max_x = 0
        self.max_y = 0
        loc = meta[['x', 'y']].values
        self.max_x = max(self.max_x, loc[:, 0].max())
        self.max_y = max(self.max_y, loc[:, 1].max())
        return meta

    def get_overlap(self, meta_dict, gene_list):
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set & set(i.columns)
        return list(gene_set)
