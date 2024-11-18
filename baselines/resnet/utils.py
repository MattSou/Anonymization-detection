import torch
import torchvision
from torch import nn
import numpy as np
from tqdm import tqdm


def create_resnet50(num_classes, pretrained):
    resnet = torchvision.models.resnet50(pretrained=pretrained)
    resnet.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
    for param in resnet.parameters():
        param.requires_grad = True
    return resnet


def model_eval(model, loader, current_metrics, criterion = None, device = 'cuda'):
    metrics = [key for key in current_metrics.keys()]
    eval_loss = False
    if 'loss' in metrics:
        assert criterion is not None, "You need to provide a criterion to compute the loss"
        eval_loss = True
    model.eval()
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    total = 0
    c_1 = 0
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in tqdm(loader):
            #clear_output(wait=True)
            #print(i)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if eval_loss:
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
            #print(outputs)
            _, predicted = torch.max(outputs, 1)
            #print(predicted)
            #print(labels)
            c_1+=predicted.sum().item()
            total += labels.size(0)
            TP+=((predicted == 1) & (labels == 1)).sum().item()
            TN+=((predicted == 0) & (labels == 0)).sum().item()
            FP+=((predicted == 1) & (labels == 0)).sum().item()
            FN+=((predicted == 0) & (labels == 1)).sum().item()
            #if 1 in predicted:
                #print('1 detected')
    if eval_loss:
        epoch_loss = running_loss / total
        current_metrics['loss'].append(epoch_loss)

    accuracy = (TP + TN) / total
    precision = TP / (TP + FP) if TP+FP!=0 else 0
    recall = TP / (TP + FN)
    f1_score = 2 * precision * recall / (precision + recall) if precision!=0 else 0
    
    if 'accuracy' in metrics:
        current_metrics['accuracy'].append(accuracy)
    
    if 'precision' in metrics:
        current_metrics['precision'].append(precision)

    if 'recall' in metrics:
        current_metrics['recall'].append(recall)

    if 'f1_score' in metrics:
        current_metrics['f1_score'].append(precision)

    return current_metrics
    
def preds_and_metrics(model, data, loader, metrics=['accuracy', 'recall', 'precision', 'f1_score'], device='cuda'):
    model.to(device)
    model.eval()
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    total = 0
    c_1 = 0
    score0 = []
    score1 = []
    with torch.no_grad():
        for inputs, labels in tqdm(loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            sc = torch.nn.functional.softmax(outputs, dim=1).cpu().detach().numpy()
            score0.extend(sc[:,0])
            score1.extend(sc[:,1])

            _, predicted = torch.max(outputs, 1)
            #print(predicted)
            #print(labels)
            c_1+=predicted.sum().item()
            total += labels.size(0)
            TP+=((predicted == 1) & (labels == 1)).sum().item()
            TN+=((predicted == 0) & (labels == 0)).sum().item()
            FP+=((predicted == 1) & (labels == 0)).sum().item()
            FN+=((predicted == 0) & (labels == 1)).sum().item()

    results = data.img_labels.copy()
    results['score0'] = score0
    results['score1'] = score1
    results['predicted'] = np.argmax(np.array([score0, score1]), axis=0)

    accuracy = (TP + TN) / total
    precision = TP / (TP + FP) if TP+FP!=0 else 0
    recall = TP / (TP + FN)
    f1_score = 2 * precision * recall / (precision + recall) if precision!=0 else 0

    test_metrics = {}
    
    if 'accuracy' in metrics:
        test_metrics['accuracy']=accuracy
    
    if 'precision' in metrics:
        test_metrics['precision'] = precision

    if 'recall' in metrics:
        test_metrics['recall'] = recall

    if 'f1_score' in metrics:
        test_metrics['f1_score'] = f1_score


    return results, test_metrics
