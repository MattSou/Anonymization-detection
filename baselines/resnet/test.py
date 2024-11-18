import torch
import argparse

import yaml
from preprocessing import test_preprocess

from utils import create_resnet50, preds_and_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Train a model on a dataset')

parser.add_argument('--model_path', type=str, required=True, help='Path to the model to use')
parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')

args = parser.parse_args()

with open(args.config, 'r') as file:
    config = yaml.safe_load(file)


resnet = create_resnet50(config['num_classes'], False)
resnet.load_state_dict(torch.load(args.model_path))

data_cfg = config['dataset']
pr_cfg = config['preprocessing']
save_cfg = config['save']

test_data = test_preprocess(data_cfg, pr_cfg)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=config['num_workers'])

test_results, test_metrics = preds_and_metrics(resnet, test_data, test_dataloader, metrics = config['metrics'], device=device)



test_results.to_csv(config['save']['predictions'])

for metric in config['metrics']:
    print(f'{metric} : {test_metrics[metric]:.2f}')

