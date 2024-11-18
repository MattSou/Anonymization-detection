import torch
from torch import nn, optim


import os

import numpy as np
import json
import argparse

from tqdm import tqdm

import yaml
from utils import model_eval, create_resnet50

from preprocessing import preprocess

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
tr_cfg = config['train']
save_cfg = config['save']
if not os.path.exists(save_cfg['model']):
    os.makedirs(save_cfg['model'])

train_dataset, val_dataset, train_videos, val_videos = preprocess(data_cfg, pr_cfg)


resnet = create_resnet50(config['model']['num_classes'], config['model']['pretrained'])


#CHARGER LES DONNEES
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=tr_cfg['batch_size'], shuffle=True, num_workers=tr_cfg['num_workers'])
print(len(train_loader), 'batches in the training loader')

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=tr_cfg['batch_size'], shuffle=False, num_workers=tr_cfg['num_workers'])
print(len(val_loader), 'batches in the validation loader')



#======PARAMETRES======

#CHOIX DE LA LOSS
criterion = eval(tr_cfg['loss'])

#OPTIMIZER
opt_cfg = tr_cfg['optimizer']
optimizer = eval(f"optim.{opt_cfg['name']}(resnet.parameters(), lr=opt_cfg['learning_rate'], weight_decay=opt_cfg['weight_decay'])")

#LR SCHEDULER
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

#======================


resnet.to(device)

save_epochs = np.arange(0, tr_cfg['n_epochs']-1, config['save_epochs_interval'])
val_epochs = np.arange(0, tr_cfg['n_epochs'], config['val_epochs_interval'])
val_metrics = {key : [] for key in config['val_metrics']}

train_loss_list = []

#========================

#ENTRAINEMENT


for epoch in range(tr_cfg['n_epochs']):
    resnet.train()
    train_running_loss = 0.0

    for inputs, labels in tqdm(train_loader):

        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = resnet(inputs) 

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_running_loss += loss.item() * inputs.size(0)

    train_epoch_loss = train_running_loss / len(train_loader.dataset)
    train_loss_list.append(train_epoch_loss)

    print(f"Epoch [{epoch+1}/{tr_cfg['n_epochs']}], Train loss: {train_epoch_loss:.4f}")

    if epoch in val_epochs:
        val_metrics = model_eval(resnet, val_loader, val_metrics, criterion = criterion, device = device)
        for metric in val_metrics.keys():
            print(f"{metric} : {val_metrics[metric][-1]}")

    if epoch in save_epochs:
        torch.save(resnet.state_dict(), os.path.join(save_cfg['model'], f"{args.name}_epoch{epoch}.pth"))

model_path = args.name+'_final.pth'
torch.save(resnet.state_dict(), os.path.join(save_cfg['model'],model_path))




metadata_path = os.path.join(save_cfg['metadata'], args.name+'_metadata.json')


training_metadata = {
    "model_name": args.name,
    "train_video_list" : train_videos,
    "val_video_list" : val_videos,
    "config" : args.config,
    "train" : {
        "loss_list" : train_loss_list
    },
    "val" : val_metrics
    
}


with open(metadata_path, "w") as outfile: 
    json.dump(training_metadata, outfile)