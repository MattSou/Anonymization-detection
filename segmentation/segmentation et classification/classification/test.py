import torch
import argparse
from torchvision import transforms as T
import json
import yaml
from model import SAlexNet
from utils import CustomTestImageDataset, preds_and_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Train a model on a dataset')

parser.add_argument('--model_path', type=str, required=True, help='Path to the model to use')
parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')

args = parser.parse_args()

with open(args.config, 'r') as file:
    config = yaml.safe_load(file)


model = SAlexNet(num_classes = config['num_classes'], input_size=config['input_size']).to(device)
model.load_state_dict(torch.load(args.model_path))

data_cfg = config['dataset']
pr_cfg = config['preprocessing']
save_cfg = config['save']

transform = T.Compose([
    eval('T.'+x['name']+'('+','.join([f"{key}={value}" for key, value in x['params'].items()])+')') for x in pr_cfg['transformations']
])

test_dataset = CustomTestImageDataset(data_cfg['test_annotations'], transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=config['num_workers'])

test_results, test_metrics = preds_and_metrics(model, test_dataset, test_loader, config['num_classes'], metrics = config['metrics'], device=device)



test_results.to_csv(config['save']['predictions'])

with open(config['save']['metrics'], 'w') as file:
    json.dump(test_metrics, file)
