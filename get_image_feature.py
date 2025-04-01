from PIL import Image
import numpy as np
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login, hf_hub_download
import torch
from torch.utils.data import Dataset
import cv2
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from numpy.fft import fft2, fftshift

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
login("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")

# login with your User Access Token, found at https://huggingface.co/settings/tokens
# pretrained=True needed to load UNI weights (and download weights for the first time)
# init_values need to be passed in to successfully load LayerScale parameters (e.g. - block.0.ls1.gamma)
model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))


class roi_dataset(Dataset):
    def __init__(self, img,
                 ):
        super().__init__()
        self.transform = transform

        self.images_lst = img

    def __len__(self):
        return len(self.images_lst)

    def __getitem__(self, idx):
        pil_image = Image.fromarray(self.images_lst[idx].astype('uint8'))
        image = self.transform(pil_image)
        return image


def crop_image(img, x, y, crop_size=None):
    if crop_size is None:
        crop_size = [50, 50]
    left = x - crop_size[0] // 2
    top = y - crop_size[1] // 2

    # 计算子图右下角的坐标
    right = left + crop_size[0]
    bottom = top + crop_size[1]

    if img.ndim == 3:
        cropped_img = img[top:bottom, left:right, :]
    else:
        cropped_img = img[top:bottom, left:right]

    return cropped_img


def get_patch(img, x, y, patch_size=None):
    if patch_size == 48:
        left = x - 24
        top = y - 24
        right = x + 24
        bottom = y + 24
    elif patch_size == 112:
        left = x - 56
        top = y - 56
        right = x + 56
        bottom = y + 56
    elif patch_size == 224:
        left = x - 112
        top = y - 112
        right = x + 112
        bottom = y + 112
    else:
        left = x - 8
        top = y - 8
        right = x + 8
        bottom = y + 8

    if img.ndim == 3:
        cropped_img = img[top:bottom, left:right, :]
    else:
        cropped_img = img[top:bottom, left:right]

    return cropped_img


def UNI_features(img_path, spatial, patch_size):
    model.eval()
    model.to(device)

    img = cv2.imread(img_path)
    img = np.array(img)

    sub_images = []

    spot_num = len(spatial)
    loc = spatial
    if isinstance(loc, pd.DataFrame):
        loc = loc.values
    # 遍历spot以提取子图像
    for i in range(spot_num):
        x = loc[i, 0]
        y = loc[i, 1]
        sub_image = get_patch(img, x, y, patch_size=patch_size)

        sub_images.append(sub_image)

    sub_images = np.array(sub_images)

    test_datat = roi_dataset(sub_images)
    database_loader = torch.utils.data.DataLoader(test_datat, batch_size=512, shuffle=False)

    feature_embs = []

    with torch.inference_mode():
        for batch in database_loader:
            batch = batch.to(device)
            feature_emb = model(batch)
            feature_embs.append(feature_emb.cpu())
        feature_embs = np.concatenate(feature_embs, axis=0)

    return feature_embs


