import json
import numpy as np
import pandas as pd
import os
from sklearn import svm
import pickle

import argparse

import yaml

from utils import compute_embeddings

parser = argparse.ArgumentParser(description='Train a model on a dataset')

parser.add_argument('--embeddings', type=str, required=True, help='Path to the output JSON file of embeddings')
parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')

args = parser.parse_args()

with open(args.config, 'r') as file:
    config = yaml.safe_load(file)

annotation_file = os.path.join(config['dataset']['root'], config['dataset']['name'], 'annotations.csv')


annotations = pd.read_csv(annotation_file, header=None, names=["file", "label"]).assign(file=lambda x: x['file'].apply(lambda x: os.path.join(config['dataset']['root'], config['dataset']['name'], x)))

files = annotations["file"].tolist()

with open(args.embeddings, "r") as f:
    embeddings = json.load(f)

embedding_list = list(embeddings.values())
file_list = list(embeddings.keys())

labels = annotations.set_index("file")["label"].to_dict()


y = [labels[file] for file in file_list]


train_test_split = 0.8

train_size = int(len(y) * train_test_split)

X_train = np.array(embedding_list[:train_size]).reshape(-1,1536)
X_test = np.array(embedding_list[train_size:]).reshape(-1,1536)

y_train = y[:train_size]
y_test = y[train_size:]



clf = svm.SVC(gamma='scale')
clf.fit(X_train, y_train)


with open(os.path.join(config['save']['model'], config['model']['name']+'.pkl'), "wb") as f:
    pickle.dump(clf, f)


print(f'Accuracy on the validation dataset : {clf.score(X_test, y_test):.2f}')



