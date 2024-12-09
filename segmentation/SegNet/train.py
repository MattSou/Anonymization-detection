import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import yaml
import os
from tqdm import tqdm
from SegNet import SegNet
from utils import MaskImageDataset, process_annotations

import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, required=True, help='Name of the model')
parser.add_argument('--annotations_path', type=str, required=True, help='Path to the annotations file containing image and mask paths')
parser.add_argument('--config', type=str, required=True, help='Path to the config file')
args = parser.parse_args()

with open(args.config, 'r') as file:
    config = yaml.safe_load(file)

img_paths, mask_paths, test_paths, neg_paths = process_annotations(args.annotations_path, neg_frac=config['preprocessing']['neg_frac'], train_test_split=config['preprocessing']['train_test_split'], base_dir=config['dataset']['base_dir'])

transform = transforms.Compose([
    eval('transforms.'+x['name']+'('+','.join([f"{key}={value}" for key, value in x['params'].items()])+')') for x in config['preprocessing']['base_transformations']
])

    

dataset = MaskImageDataset(img_paths, mask_paths, neg_paths=neg_paths, dataset_dir = config['dataset']['root'], transform=transform)
dataloader = DataLoader(dataset, batch_size=config['train']['batch_size'], shuffle=True, num_workers=config['train']['num_workers'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SegNet(out_chn=config['num_classes']).to(device)

criterion = eval(config['train']['loss']+"(weight=torch.tensor(config['train']['loss_weights'])).to(device)")
#criterion = nn.CrossEntropyLoss()
optimizer = eval('torch.optim.'+config['train']['optimizer']['name']+"(model.parameters(), lr=config['train']['optimizer']['learning_rate'], momentum=config['train']['optimizer']['momentum'], weight_decay=config['train']['optimizer']['weight_decay'])")
scheduler = eval('torch.optim.lr_scheduler.'+config['train']['scheduler']['name']+'(optimizer, **config["train"]["scheduler"]["params"])')

losses = []

num_epochs = config['train']['n_epochs']
for epoch in range(num_epochs):
    epoch_loss = 0
    for i, (images, masks) in enumerate(tqdm(dataloader)):
        images = images.to(device)
        masks = masks.to(device)
        #print(masks)
        optimizer.zero_grad()
        outputs = model(images)
        # print('here')
        #print(outputs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        # if (i+1) % 5 == 0:
        #     print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item()}')
    scheduler.step(loss.item())
    losses.append(epoch_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss}')

torch.save(model.state_dict(), os.path.join(config['save']['model'], args.name+'.pth'))

with open(os.path.join(config['save']['metadata'], args.name+'.json'), 'w') as f:
    json.dump({'img_paths': img_paths.to_list(), 'test_paths': test_paths.to_list(), 'mask_paths': mask_paths.to_list(), 'neg_paths': [], 'losses': losses }, f)


