import torch
from torchvision import transforms

import pandas as pd
import os

import argparse

import yaml

from utils import compute_embeddings

parser = argparse.ArgumentParser(description='Train a model on a dataset')

parser.add_argument('--output', type=str, required=True, help='Path to the output JSON file of embeddings')
parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')

args = parser.parse_args()

with open(args.config, 'r') as file:
    config = yaml.safe_load(file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.hub.load('facebookresearch/dinov2', config['model']['name'])

transform = transforms.Compose([
            eval('transforms.'+x['name']+'('+','.join([f"{key}={value}" for key, value in x['params'].items()])+')') for x in config['preprocessing']['base_transformations']
        ])

model.to(device)

annotation_file = os.path.join(config['dataset']['root'], config['dataset']['name'], 'annotations.csv')

annotations = pd.read_csv(annotation_file, header=None, names=["file", "label"]).assign(file=lambda x: x['file'].apply(lambda x: os.path.join(config['dataset']['root'], config['dataset']['name'], x)))

files = annotations["file"].tolist()


compute_embeddings(model, device, files, args.output, transform)







