import torch
from torchvision import transforms
import numpy as np
import pandas as pd
import os
import pickle
import argparse

import yaml

from utils import compute_embeddings

parser = argparse.ArgumentParser(description='Train a model on a dataset')

parser.add_argument('--model_path', type=str, required=True, help='Path to the pickle model to use')
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

write = (config['save']['embeddings'] != '')

embeddings = compute_embeddings(model, device, files, config['save']['embeddings'], transform, write=write)


clf = pickle.load(open(args.model_path, "rb"))


labels = annotations.set_index("file")["label"].to_dict()

X = np.array(list(embeddings.values())).reshape(-1,1536)
y = [labels[file] for file in files]


y_pred = clf.predict(X)

accuracy = (y_pred == y).sum() / len(y)
precision = ((y_pred == y) & (y_pred == 1)).sum() / (y_pred == 1).sum()
recall = ((y_pred == y) & (y_pred == 1)).astype(int).sum() / (np.array(y) == 1).sum()
f1_score = 2 * precision * recall / (precision + recall)

for metric in config['metrics']:
    print(f'{metric} : {eval(metric)}')

results = pd.DataFrame({'file': files, 'label': y, 'prediction': y_pred})
results.to_csv(config['save']['results'])



