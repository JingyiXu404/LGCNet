import torch.utils.data as data
import torch
import PIL.Image as Image
# from data_process import *
import os
import random
from math import ceil
from torchvision.transforms import transforms
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import DataLoader
import cv2
import math
from glob import glob
from tqdm import tqdm,trange
import random
Image.MAX_IMAGE_PIXELS = None
import torch.nn as nn
import matplotlib.pyplot as plt
from skimage import io,transform
import imageio
from torchvision.transforms import Compose
import platform
sysstr = platform.system()

def train_transform(degree=180):

    return transforms.Compose([
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=degree),
        transforms.ColorJitter(),
    ])
def padding_size(x, d):
    x = x + 2
    return math.ceil(x / d) * d - x
def tensor2uint(img: torch.Tensor) -> np.ndarray:
    img = img.data.squeeze().float().cpu().numpy()
    # img = np.uint8(img * 255.0)
    return img
def imread_uint(path: str, n_channels: int = 3) -> np.ndarray:
    #  input: path
    # output: HxWx3(RGB or GGG), or HxWx1 (G)
    if n_channels == 1:
        img = cv2.imread(path, 0)
        img = np.expand_dims(img, axis=2)  # HxWx1
    elif n_channels == 3:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = img #cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise NotImplementedError
    return img
class FNFDataset(data.Dataset):
    def __init__(self, root1, root2, sigma, patch_size, normalization=False, transform=False, type = 'train'):
        self.root1 = root1
        self.root2 = root2
        self.file1_lists = []
        self.file2_lists = []
        for dir in np.array([x.path for x in os.scandir(self.root1)]):
            files = np.array([x.path.decode('utf-8') for x in os.scandir(dir)])
            self.file1_lists = np.append(self.file1_lists, files)
        for dir in np.array([x.path for x in os.scandir(self.root2)]):
            files = np.array([x.path.decode('utf-8') for x in os.scandir(dir)])
            self.file2_lists = np.append(self.file2_lists, files)
        self.file1_lists.sort()
        self.file2_lists.sort()

        self.image1_files = self.file1_lists
        self.image2_files = self.file2_lists
        self.sigma = sigma
        self.num_img = len(self.file1_lists)
        self.patch_size = patch_size
        self.normalization = normalization
        self.transform = transform
        self.type = type
        if self.transform:
            self.train_transform = train_transform()


    def __getitem__(self, index):
        img1_path = self.image1_files[index]
        img2_path = self.image2_files[index]
        if self.transform:
            img1 = self.train_transform(Image.open(self.image1_files[index]))
            img1 = np.array(img1)
            img2 = self.train_transform(Image.open(self.image2_files[index]))
            img2 = np.array(img2)
        else:
            img1 = imread_uint(self.image1_files[index])
            img2 = imread_uint(self.image2_files[index])

        # crop
        if self.type == 'train':
            H, W = img1.shape[:2]
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch1 = img1[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            patch2 = img2[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
        else: #type == 'test'
            patch1 = img1
            patch2 = img2

        patch1_gt = patch1
        # cv2.imwrite('patch1_gt.png', patch1_gt)
        # cv2.imwrite('patch2.png', patch2)

        patch1 = torch.from_numpy(np.ascontiguousarray(np.array(patch1))).permute(2, 0, 1).float()/255.
        patch2 = torch.from_numpy(np.ascontiguousarray(np.array(patch2))).permute(2, 0, 1).float()/255.
        patch1_gt = torch.from_numpy(np.ascontiguousarray(np.array(patch1_gt))).permute(2, 0, 1).float()/255.
        patch1 = patch1 + (self.sigma/255) * torch.randn_like(patch1)
        # patch1_2 = tensor2uint(patch1.detach().float())
        # cv2.imwrite('patch1.png', patch1_2)
        return patch1, patch2, patch1_gt

    def __len__(self):
        return len(self.image1_files)


if __name__ =="__main__":
    root_path_train_1 = '/temp_disk2/XJY/DCDicL/data/denoising/train_nonflash/'
    root_path_train_2 = '/temp_disk2/XJY/DCDicL/data/denoising/train_flash/'
    root_path_test_1 = '/temp_disk2/XJY/DCDicL/data/denoising/test_nonflash/'
    root_path_test_2 = '/temp_disk2/XJY/DCDicL/data/denoising/test_flash/'
    patch_size = 128
    sigma = 25
    dr_dataset_train = FNFDataset(root1=root_path_train_1,root2=root_path_train_2,sigma = sigma,
                                  patch_size=patch_size, normalization=False, transform=False, type='train')
    loader_train = DataLoader(dr_dataset_train, batch_size=1, num_workers=0, shuffle=True)
    dr_dataset_test = FNFDataset(root1=root_path_test_1, root2=root_path_test_2, sigma=sigma,
                                  patch_size=patch_size, normalization=False, transform=False, type='test')
    loader_test = DataLoader(dr_dataset_test, batch_size=1, num_workers=0, shuffle=False)
    for packs in tqdm(loader_test):
        input1, input2, input1_gt = packs
        print(input1.shape)
        print(input2.shape)
        print(input1_gt.shape)

