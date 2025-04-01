import pandas as pd
import numpy as np
import torch
import scanpy as sc
import scipy.sparse as sp
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import os
from graph_construction import calcADJ


class MyDatasetTrans(Dataset):
    """Operations with the datasets."""

    def __init__(self, train, normed_data, coor_df, image, transform=None):
        """
        Args:
            normed_data: Normalized data extracted from original AnnData object.
            coor_df: Spatial location extracted from original AnnData object.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = normed_data.values.T
        self.coor_df = coor_df.values
        self.image = image
        self.transform = transform

        self.coord = np.array_split(self.coor_df, np.ceil(len(self.coor_df) / 50))
        self.exp = np.array_split(self.data, np.ceil(len(self.data) / 50))
        if train:
            self.image_feature_112 = np.array_split(self.image['sample_features_112'],
                                                np.ceil(len(self.image['sample_features_112']) / 50))
        else:
            self.image_feature_112 = np.array_split(self.image['fill_features_112'],
                                                    np.ceil(len(self.image['fill_features_112']) / 50))

        # if train:
        # self.image_feature_16 = np.array_split(self.image['sample_features_16'],
        #                                        np.ceil(len(self.image['sample_features_16']) / 50))
        #
        # self.image_feature_48 = np.array_split(self.image['sample_features_48'],
        #                                        np.ceil(len(self.image['sample_features_48']) / 50))
        # self.image_feature_112 = np.array_split(self.image['sample_features_112'],
        #                                         np.ceil(len(self.image['sample_features_112']) / 50))
        # self.image_feature_224 = np.array_split(self.image['sample_features_224'],
        #                                         np.ceil(len(self.image['sample_features_224']) / 50))
        # else:
        # self.image_feature_16 = np.array_split(self.image['fill_features_16'],
        #                                        np.ceil(len(self.image['fill_features_16']) / 50))
        #
        # self.image_feature_48 = np.array_split(self.image['fill_features_48'],
        #                                        np.ceil(len(self.image['fill_features_48']) / 50))
        # self.image_feature_112 = np.array_split(self.image['fill_features_112'],
        #                                         np.ceil(len(self.image['fill_features_112']) / 50))
        # self.image_feature_224 = np.array_split(self.image['fill_features_224'],
        #                                         np.ceil(len(self.image['fill_features_224']) / 50))
        # self.adj = [calcADJ(coord=i, k=4, pruneTag='NA') for i in self.coord]

    def __len__(self):
        return len(self.coord)

    def __getitem__(self, idx):
        exp = torch.tensor(self.exp[idx])
        coord = torch.tensor(self.coord[idx])
        # image_16 = torch.tensor(self.image_feature_16[idx]).unsqueeze(1)
        # image_48 = torch.tensor(self.image_feature_48[idx]).unsqueeze(1)
        image_112 = torch.tensor(self.image_feature_112[idx])
        # image_224 = torch.tensor(self.image_feature_224[idx]).unsqueeze(1)
        # image = torch.concat((image_16, image_48, image_112, image_224), dim=1)
        # adj = self.adj[idx]
        sample = (exp, coord, image_112)

        return sample


class MyDatasetTrans2(Dataset):
    """Operations with the datasets."""

    def __init__(self, coor_df, image, transform=None):
        """
        Args:
            normed_data: Normalized data extracted from original AnnData object.
            coor_df: Spatial location extracted from original AnnData object.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.coor_df = coor_df.values
        self.image = image
        self.transform = transform

        self.coord = np.array_split(self.coor_df, np.ceil(len(self.coor_df) / 50))
        self.image_feature = np.array_split(self.image['fill_features_112'],
                       np.ceil(len(self.image['fill_features_112']) / 50))


    def __len__(self):
        return len(self.coord)

    def __getitem__(self, idx):
        coord = torch.tensor(self.coord[idx])
        image = torch.tensor(self.image_feature[idx])

        sample = (coord, image)

        return sample
