from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--base_dir", 
                    type = str, 
                    help="choose the directory of the base dataset")
parser.add_argument('--anonym_dir', 
                    type = str, 
                    help="choose the directory of the anonymized dataset")
parser.add_argument('--target_dir', 
                    type = str, 
                    help="target_directory for your synthetic dataset")
parser.add_argument('--annotation_file',     
                    type = str, 
                    help="annotation file for the dataset")

args = parser.parse_args()

ann = pd.read_json(args.annotation_file).T

img_paths = ann.reset_index(names=['img_path']).query('label==1').assign(mask_path = lambda df: df['img_path'].apply(lambda x: x.replace(args.base_dir, args.target_dir))).reset_index(drop=True)

for i, row in tqdm(img_paths.iterrows()):
    img = Image.open(row['img_path'])
    x1, y1, x2, y2 = row['coord']['x1'], row['coord']['y1'], row['coord']['x2'], row['coord']['y2']    
    sigma_y, sigma_x = (y2-y1)//2, (x2-x1)//2
    c = x1+sigma_x, y1+sigma_y
    x, y =np.mgrid[0:img.size[1], 0:img.size[0]]
    mask=((x-c[0])**2/(sigma_x+10)**2 + (y-c[1])**2/(sigma_y+10)**2 <= 1)
    dir = row['mask_path'].split("/")[:-1]
    if not os.path.exists("/".join(dir)):
        os.makedirs("/".join(dir))
    # plt.imshow(img)
    # plt.imshow(mask, cmap='gray', alpha=0.2)
    # plt.show()
    # print("=========================================")
    plt.imsave(row['mask_path'], mask, cmap='gray')
    
mask_annotations = img_paths.assign(anon_path = lambda df: df['img_path'].apply(lambda x: x.replace(args.base_dir, args.anonym_dir)))
mask_annotations[['anon_path', 'mask_path']].to_csv(os.path.join(args.target_dir, 'mask_annotations.csv'), index=False, header=False)