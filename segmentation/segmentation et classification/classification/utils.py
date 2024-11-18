import torch
from torchvision import transforms as T
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image


def classifier_in_features(input_size):
    x = np.floor((input_size - 2*0 - (11-1)-1)/4 + 1)
    x = np.floor((x + 2*0 - (3-1)-1)/2 + 1)
    x = np.floor((x + 2*2 - (5-1)-1)/1 + 1)
    x = np.floor((x + 2*0 - (3-1)-1)/2 + 1)
    x = np.floor((x + 2*2 - (5-1)-1)/1 + 1)
    x = np.floor((x + 2*1 - (3-1)-1)/1 + 1)
    x = np.floor((x + 2*1 - (3-1)-1)/1 + 1)
    x = np.floor((x + 2*0 - (3-1)-1)/2 + 1)
    return x.astype(int)

class ConditionalResize(T.Resize):

    def forward(self, img):
        if img.size[1] > self.size and img.size[0] > self.size:
            return img
        return T.functional.resize(img, self.size, self.interpolation, self.max_size, self.antialias)

class PatchDataset(Dataset):
    def __init__(self, annotations_file, transform=None):
        self.img_labels = pd.read_csv(annotations_file, header=None, names=['img_path', 'label'])
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path, label = self.img_labels.iloc[idx]
        # img_path = self.img_labels.iloc[idx, 0]
        image = Image.open(img_path)
        # label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        
        return image, label

class CustomTestImageDataset(Dataset):
    def __init__(self, annotations_file, transform=None):
        self.img_labels = pd.read_csv(annotations_file, header=None, names=['img_path', 'label'])
        # print(len(self.img_labels))
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx, 0]
        image = Image.open(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label

def preds_and_metrics(model, data, loader, num_classes, metrics=['accuracy', 'recall', 'precision', 'f1_score'], device='cuda'):
    model.to(device)
    model.eval()
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    total = 0
    scores = {f'score{i}': [] for i in range(num_classes)}
    with torch.no_grad():
        for inputs, labels in tqdm(loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            sc = torch.nn.functional.softmax(outputs, dim=1).cpu().detach().numpy()
            for i in range(num_classes):
                scores[f'score{i}'].extend(sc[:, i])
            _, predicted = torch.max(outputs, 1)

    results = data.img_labels.copy()
    for i in range(num_classes):
        results[f'score{i}'] = scores[f'score{i}']
    results['predicted'] = np.argmax(np.array([scores[f'score{i}'] for i in range(num_classes)]).T, axis=1)

    count_per_class = {i: {} for i in range(num_classes)}
    
    
    for i in range(num_classes):
        TP_i = ((results['label']==i) & (results['predicted']==i)).sum()
        TN_i = ((results['label']!=i) & (results['predicted']!=i)).sum()
        FP_i = ((results['label']!=i) & (results['predicted']==i)).sum()
        FN_i = ((results['label']==i) & (results['predicted']!=i)).sum()
        count_per_class[i]['TP'] = TP_i
        count_per_class[i]['TN'] = TN_i
        count_per_class[i]['FP'] = FP_i
        count_per_class[i]['FN'] = FN_i
        print(f'Class {i}: TP={TP_i}, TN={TN_i}, FP={FP_i}, FN={FN_i}, total={TP_i+TN_i+FP_i+FN_i}')
    

    micro_metrics = {}

    TP = sum([count_per_class[i]['TP'] for i in range(num_classes)])
    TN = sum([count_per_class[i]['TN'] for i in range(num_classes)])
    FP = sum([count_per_class[i]['FP'] for i in range(num_classes)])
    FN = sum([count_per_class[i]['FN'] for i in range(num_classes)])
    for metric in metrics:
        if metric == 'accuracy':
            micro_metrics['accuracy'] = TP/(TP+FP)
        if metric == 'precision':
            micro_metrics['precision'] = TP/(TP+FP)
        if metric == 'recall':
            micro_metrics['recall'] = TP/(TP+FN)
        if metric == 'f1_score':
            micro_metrics['f1_score'] = 2*TP/(2*TP+FP+FN)
    
    per_class_metrics = {}
    for i in range(num_classes):
        per_class_metrics[i] = {}
        if 'accuracy' in metrics:
            per_class_metrics[i]['accuracy'] = (count_per_class[i]['TP']+count_per_class[i]['TN'])/(count_per_class[i]['TP']+count_per_class[i]['TN']+count_per_class[i]['FP']+count_per_class[i]['FN'])
        if 'precision' in metrics:
            per_class_metrics[i]['precision'] = count_per_class[i]['TP']/(count_per_class[i]['TP']+count_per_class[i]['FP'])
        if 'recall' in metrics:
            per_class_metrics[i]['recall'] = count_per_class[i]['TP']/(count_per_class[i]['TP']+count_per_class[i]['FN'])
        if 'f1_score' in metrics:
            per_class_metrics[i]['f1_score'] = 2*count_per_class[i]['TP']/(2*count_per_class[i]['TP']+count_per_class[i]['FP']+count_per_class[i]['FN'])

    macro_metrics = {}
    if 'accuracy' in metrics:
        macro_metrics['accuracy'] = np.mean([per_class_metrics[i]['accuracy'] for i in range(num_classes)])
    if 'precision' in metrics:
        macro_metrics['precision'] = np.mean([per_class_metrics[i]['precision'] for i in range(num_classes)])
    if 'recall' in metrics:
        macro_metrics['recall'] = np.mean([per_class_metrics[i]['recall'] for i in range(num_classes)])
    if 'f1_score' in metrics:
        macro_metrics['f1_score'] = np.mean([per_class_metrics[i]['f1_score'] for i in range(num_classes)])

    test_metrics = {'micro': micro_metrics, 'macro': macro_metrics, 'per_class': per_class_metrics}

    return results, test_metrics