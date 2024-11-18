import torch
from torch import optim, nn
import numpy as np
from torchvision import transforms as T
from tqdm import tqdm
import os
import argparse
import yaml
import json
from model import SAlexNet
from utils import PatchDataset, CustomTestImageDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description='Train a model on a dataset')

parser.add_argument('--name', type=str, required=True, help='Name of the model')
parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')

args = parser.parse_args()


#CHARGEMENT DES PARAMETRES
with open(args.config, 'r') as file:
    config = yaml.safe_load(file)

#PARAMETRES
data_cfg = config['dataset']
pr_cfg = config['preprocessing']
model_cfg = config['model']
tr_cfg = config['train']
save_cfg = config['save']
if not os.path.exists(save_cfg['model']):
    os.makedirs(save_cfg['model'])


transform = T.Compose([
    eval('T.'+x['name']+'('+','.join([f"{key}={value}" for key, value in x['params'].items()])+')') for x in pr_cfg['transformations']
])

patch_transform = T.Compose([
    eval('T.'+x['name']+'('+','.join([f"{key}={value}" for key, value in x['params'].items()])+')') for x in pr_cfg['patch_transformations']
])
    

train_dataset = PatchDataset(data_cfg['train_annotations'], transform=patch_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=tr_cfg['batch_size'], shuffle=True, num_workers=tr_cfg['num_workers'])

test_dataset = CustomTestImageDataset(data_cfg['test_annotations'], transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=tr_cfg['batch_size'], shuffle=False, num_workers=tr_cfg['num_workers'])


model = SAlexNet(num_classes = model_cfg['num_classes'], input_size=model_cfg['input_size']).to(device)

#======PARAMETRES======

#CHOIX DE LA LOSS
criterion = eval(tr_cfg['loss'])

#OPTIMIZER
opt_cfg = tr_cfg['optimizer']
optimizer = eval(f"optim.{opt_cfg['name']}(model.parameters(), **opt_cfg['params'])")

sch_cfg = tr_cfg['scheduler']
scheduler = eval(f"optim.lr_scheduler.{sch_cfg['name']}(optimizer, **sch_cfg['params'])")

#LR SCHEDULER
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

#======================



save_epochs = np.arange(0, tr_cfg['n_epochs']-1, config['save_epochs_interval'])
val_epochs = np.arange(0, tr_cfg['n_epochs'], config['val_epochs_interval'])
val_metrics = {key : [] for key in config['val_metrics']}

train_loss_list = []

#========================

#ENTRAINEMENT


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.train()

for epoch in range(tr_cfg['n_epochs']):
    epoch_loss = 0
    for (images, labels) in tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch: {epoch}, Loss: {epoch_loss/len(train_loader)}")
    train_loss_list.append(epoch_loss/len(train_loader))
    scheduler.step(epoch_loss)
    # if epoch % 15 == 0 and epoch+90 != 0:
    if epoch in save_epochs:
        torch.save(model.state_dict(), os.path.join(save_cfg['model'], args.name+f'_{epoch}.pth'))
    if epoch in val_epochs and config['val_metrics']!=[]:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            test_loss= 0
            for (images, labels) in tqdm(test_loader):
                # print(images)
                # print(labels)
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                tl = criterion(outputs, labels)
                test_loss += tl.item()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        if 'loss' in config['val_metrics']:
            val_metrics['loss'].append(test_loss/len(test_loader))
            print(f"Test Loss: {test_loss/len(test_loader)}")
        if 'accuracy' in config['val_metrics']:
            val_metrics['accuracy'].append(100 * correct / total)
            print(f"Test Accuracy: {100 * correct / total}")
        model.train()

torch.save(model.state_dict(), os.path.join(save_cfg['model'], args.name+'_final.pth'))




metadata_path = os.path.join(save_cfg['metadata'], args.name+'_metadata.json')


training_metadata = {
    "model_name": args.name,
    "config" : args.config,
    "train" : {
        "loss_list" : train_loss_list
    },
    "val" : val_metrics
    
}


with open(metadata_path, "w") as outfile: 
    json.dump(training_metadata, outfile)