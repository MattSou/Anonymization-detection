import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import yaml
import os
from tqdm import tqdm
from SegNet import SegNet
from utils import TestMaskImageDataset, process_annotations
from torchmetrics.segmentation import MeanIoU
from PIL import Image

import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True, help='Name of the model')
parser.add_argument('--annotations_path', type=str, required=True, help='Path to the annotations file containing image and mask paths')
parser.add_argument('--config', type=str, required=True, help='Path to the config file')
args = parser.parse_args()

with open(args.config, 'r') as file:
    config = yaml.safe_load(file)

img_paths, mask_paths, test_paths, neg_paths = process_annotations(args.annotations_path, neg_frac=0, train_test_split=1, base_dir=config['dataset']['base_dir'])

transform = transforms.Compose([
    eval('transforms.'+x['name']+'('+','.join([f"{key}={value}" for key, value in x['params'].items()])+')') for x in config['preprocessing']['base_transformations']
])

    

dataset = TestMaskImageDataset(img_paths, mask_paths, dataset_dir = config['dataset']['root'], transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config['num_workers'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SegNet(out_chn=config['num_classes']).to(device)
model.load_state_dict(torch.load(args.model_path))
model.to(device)

with torch.no_grad():
    model.eval()
    miou_metric = MeanIoU(num_classes=config['num_classes'], include_background=True, per_class=True).to(device)
    acc = 0
    miou = 0
    for x, y, mask, img_name in tqdm(dataloader):
        x = x.to(device)
        y = y.to(device)
        y_hat = model(x)
        y_hat = torch.argmax(y_hat, dim=1)

        if 'accuracy' in config['metrics']:
            acc += (y_hat.squeeze().cpu().numpy() == mask.squeeze().numpy()).mean()
        if 'mIoU' in config['metrics']:
            miou += miou_metric(y_hat, y.long())

        y_hat = y_hat.squeeze().cpu().numpy()
        Image.fromarray(y_hat.astype('uint8')).save(os.path.join(config['save']['predictions'], img_name[0].replace('.jpg', '.png')))

        


if 'accuracy' in config['metrics']:
    print(f'Accuracy: {acc/len(dataloader)}')
if 'mIoU' in config['metrics']:
    print(f'mIoU: {miou/len(dataloader)}')



