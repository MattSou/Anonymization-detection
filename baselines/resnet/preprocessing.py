from torchvision import transforms

import os
import pandas as pd

from torch.utils.data import Dataset
from PIL import Image

import numpy as np
import json

def preprocess(data_cfg, pr_cfg):
    dataset_path = os.path.join(data_cfg['root'], data_cfg['name'])
    metadata_keyframes_file = os.path.join(dataset_path,'metadata_keyframes.csv')
    metadata_videos_file = os.path.join(dataset_path, 'metadata_videos.csv')
    annotation_file = os.path.join(dataset_path,'annotations.csv')  

    kf_df, metadata_keyframes, metadata_videos, annotations = prepare_dataframe(metadata_keyframes_file=metadata_keyframes_file, metadata_videos_file=metadata_videos_file, annotation_file=annotation_file, dataset_name=data_cfg['name'])

    train_val_split_frac = pr_cfg['train_val_frac']
    train_keyframes, val_keyframes = train_val_split(kf_df, metadata_videos, frac=train_val_split_frac)
    train_videos, val_videos = train_keyframes['video_id'].unique().tolist(), val_keyframes['video_id'].unique().tolist()

    if pr_cfg['deduplication']:
        cluster_file = os.path.join(dataset_path, pr_cfg['deduplication_file'])
        clusters = json.load(open(cluster_file))
        deduplicate_img_list = pd.Series(np.concatenate([clusters[key] for key in clusters.keys()])).apply(lambda x: x.split('Datasets/')[1])
        train_keyframes = train_keyframes.query('img_path in @deduplicate_img_list').reset_index(drop=True)
        val_keyframes = val_keyframes.query('img_path in @deduplicate_img_list').reset_index(drop=True)
    
    base_transform = transforms.Compose([
            eval('transforms.'+x['name']+'('+','.join([f"{key}={value}" for key, value in x['params'].items()])+')') for x in pr_cfg['base_transformations']
        ])
    
    if pr_cfg['data_augmentation']['augmentation']:
        aug_transform = transforms.Compose([
            eval('transforms.'+x['name']+'('+','.join([f"{key}={value}" for key, value in x['params'].items()])+')') for x in pr_cfg['data_augmentation']['transformations']
        ])
        train_dataset = AugImageDataset(dataframe=train_keyframes, dataset_dir=data_cfg['root'], transform=base_transform, aug_transform=aug_transform, aug_factor=3)
    else :
        train_dataset = CustomImageDataset2(dataframe=train_keyframes, dataset_dir=data_cfg['root'], transform=base_transform)
    val_dataset = CustomImageDataset2(dataframe=val_keyframes, dataset_dir=data_cfg['root'], transform=base_transform)

    return train_dataset, val_dataset, train_videos, val_videos

def test_preprocess(data_cfg, pr_cfg):
    dataset_path = os.path.join(data_cfg['root'], data_cfg['name'])
    annotation_file = os.path.join(dataset_path,'annotations.csv')

    annotation = pd.read_csv(annotation_file, header=None, names=['img_path', 'label'])
    annotation = annotation.assign(img_path = annotation['img_path'].apply(lambda x: data_cfg['name']+'/'+x))

    base_transform = transforms.Compose([
            eval('transforms.'+x['name']+'('+','.join([f"{key}={value}" for key, value in x['params'].items()])+')') for x in pr_cfg['base_transformations']
        ])

    test_dataset = CustomImageDataset2(dataframe=annotation, dataset_dir=data_cfg['root'], transform=base_transform)

    return test_dataset


def prepare_dataframe(metadata_keyframes_file, metadata_videos_file, annotation_file, dataset_name):
    metadata_keyframes = pd.read_csv(metadata_keyframes_file)
    metadata_videos = pd.read_csv(metadata_videos_file).rename(columns={'annotated.Anonym': 'annotated'}).query('annotated == True')
    metadata_keyframes = metadata_keyframes[metadata_keyframes['video_id'].isin(metadata_videos['video_id'])]
    annotation = pd.read_csv(annotation_file, header=None, names=['img_name', 'label'])
    annotation = annotation.assign(img_name = annotation['img_name'].apply(lambda x: os.path.join(dataset_name,x)))
    kf_df = metadata_keyframes.merge(annotation, right_on='img_name', left_on='keyframe_id').drop(columns='img_name').rename(columns={'keyframe_id': 'img_path'})
    print(f"Keyframes metadata loaded : {metadata_keyframes.shape[0]} keyframes")
    return kf_df, metadata_keyframes, metadata_videos, annotation

def train_val_split(kf_df, metadata_videos, frac = 0.8):
    indices = list(range(metadata_videos.shape[0]))
    print("Shuffling data...")
    np.random.shuffle(indices)
    print("Splitting data...")
    train_indices = indices[:int(frac * metadata_videos.shape[0])]
    val_indices = indices[int(frac * metadata_videos.shape[0]):]
    print("Creating datasets...")
    train_keyframes = kf_df.query('video_id in @metadata_videos.iloc[@train_indices].video_id')
    val_keyframes = kf_df.query('video_id in @metadata_videos.iloc[@val_indices].video_id')
    print("Datasets created.")

    count = train_keyframes['label'].value_counts()
    print(f"Training set : {count[1]} positive samples out of {count.sum()} ({count[1]/count.sum()*100:.2f}%) on {len(train_indices)} videos")

    count = val_keyframes['label'].value_counts()
    print(f"val set : {count[1]} positive samples out of {count.sum()} ({count[1]/count.sum()*100:.2f}%) on {len(val_indices)} videos")

    print(f"train - val split fraction : {100*train_keyframes.shape[0]/kf_df.shape[0]:.2f} %")

    return train_keyframes, val_keyframes
    

#CUSTOM IMAGE DATASET SIMPLE
class CustomImageDataset2(Dataset):
    def __init__(self, dataframe, dataset_dir = '/home/msouda/Datasets', transform=None, target_transform=None):
        self.img_labels = dataframe[['img_path', 'label']]
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

#IMAGE AUGMENTATION DATASET
class AugImageDataset(Dataset):
    def __init__(self, dataframe, dataset_dir = '/home/msouda/Datasets', transform=None, aug_transform=None, aug_factor=2):
        self.img_labels = dataframe[['img_path', 'label']].assign(img_path = dataframe['img_path'].apply(lambda x: os.path.join(dataset_dir, x)))
        self.files = self.img_labels['img_path'].tolist()
        self.labels = self.img_labels['label'].tolist()
        self.data = [(self.files[i], 'none', self.labels[i]) for i in range(len(self.files))]
        self.aug = (aug_factor-1)*[(self.files[i], 'aug', self.labels[i]) for i in range(len(self.files))]
        self.data.extend(self.aug)
        self.transform = transform
        if aug_factor>1:
            assert aug_transform is not None, "You need to provide an augmentation transform"
        self.aug_transform = aug_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        img_path = data[0]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        if data[1] == 'aug':
            image = self.aug_transform(image)

        label = data[2]
        return image, label
    
