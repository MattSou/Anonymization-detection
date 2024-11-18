import torch
from PIL import Image
from torch import nn
import numpy as np
from tqdm import tqdm
import json


def load_image(img: str, transform) -> torch.Tensor:
    """
    Load an image and return a tensor that can be used as an input to DINOv2.
    """
    img = Image.open(img)

    transformed_img = transform(img)[:3].unsqueeze(0)

    return transformed_img

def compute_embeddings(model, device, files: list, out, transform, write = True) -> dict:
    """
    Create an index that contains all of the images in the specified list of files.
    """
    all_embeddings = {}
    
    with torch.no_grad():
      for i, file in enumerate(tqdm(files)):
        embeddings = model(load_image(file, transform).to(device))

        all_embeddings[file] = np.array(embeddings[0].cpu().numpy()).reshape(1, -1).tolist()

    if write:
        with open(out, "w") as f:
            f.write(json.dumps(all_embeddings))
    
    return all_embeddings

