import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from tqdm import tqdm

import pandas as pd
import numpy as np
from random import shuffle
import json



def process_annotations(annotations_path, neg_frac=0.3, train_test_split=0.75, base_dir= None):
    data = pd.read_csv(annotations_path, header=None, names=['img_path', 'mask_path'])
    train_data = data.sample(frac=0.75)
    test_data = data.drop(train_data.index)
    train_data.reset_index(drop=True, inplace=True)
    img_paths = train_data['img_path']
    mask_paths = train_data['mask_path']
    test_paths = test_data['img_path']
    if neg_frac ==0:
        neg_paths = None
    else:
        if base_dir is None:
            neg_paths = data['img_path'].sample(frac=neg_frac).apply(lambda x: x.replace('_anonymized', ''))
        else:
            neg_paths = data['img_path'].sample(frac=neg_frac).apply(lambda x: os.path.join(base_dir, '/'.join(x.split('/')[1:])))
    return img_paths, mask_paths, test_paths, neg_paths
# data = pd.read_csv('/home/msouda/Datasets/new_synth_masks_anonymized_bis/annotations.csv', header=None, names=['img_path', 'mask_path'])#.sample(frac=0.5)

# neg_paths = None

class MaskImageDataset(Dataset):
    def __init__(self, img_paths, mask_paths, neg_paths=None, dataset_dir = '/home/msouda/Datasets', transform=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.dataset_dir = dataset_dir
        self.neg_paths = neg_paths
        if neg_paths is not None:
            self.img_paths = np.concatenate((self.img_paths, self.neg_paths))
            self.mask_paths = np.concatenate((self.mask_paths, len(self.neg_paths)*['']))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img_path = os.path.join(self.dataset_dir, img_path)
        img = Image.open(img_path).convert("RGB")
        mask_path = self.mask_paths[idx]
        if mask_path == '':
            mask = torch.zeros((1, 256, 256))
        else:
            mask_path = os.path.join(self.dataset_dir, mask_path)
            mask = Image.open(mask_path).convert("L")
            mask = self.transform(mask)
        x = self.transform(img)
        # mask = (((mask-mask.min())/(mask.max()-mask.min()))*2).long()
        y = torch.reshape(mask, (mask.shape[1], mask.shape[2])).long()
        return x, y
    

class TestMaskImageDataset(Dataset):
    def __init__(self, img_paths, mask_paths, dataset_dir = '/home/msouda/Datasets', transform=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.dataset_dir = dataset_dir

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        img_path = os.path.join(self.dataset_dir, img_path)
        mask_path = os.path.join(self.dataset_dir, mask_path)
        img_name = img_path.split('/')[-1]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        x = self.transform(img)
        mask = self.transform(mask)
        mask = (((mask-mask.min())/(mask.max()-mask.min()))*2).long()
        y = torch.reshape(mask, (mask.shape[1], mask.shape[2]))
        return x, y, mask, img_name
    
