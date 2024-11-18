import torch
from torch import optim, nn
import numpy as np
from torchvision import transforms as T
from tqdm import tqdm
import os
import argparse
import yaml
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd


def compute_patch_numbers(img):
    n,p = img.size
    x = (n//64 -1)
    _x = n%64
    y = (p//64 -1)
    _y = p%64
    start_x = np.random.randint(0,_x) if _x > 0 else 0
    start_y = np.random.randint(0,_y) if _y > 0 else 0
    patches = [(start_x+i*64,start_y+j*64) for i in range(x) for j in range(y)]
    return patches

def gaussian_white_noise(image, mean, sigma):
    """Add Gaussian white noise to an image."""
    noise = np.random.normal(mean, sigma, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def generate_random_curve(h, w, num_points=5):
    """
    Generate a random curve using spline interpolation.
    
    Args:
        h (int): Height of the image.
        w (int): Width of the image.
        num_points (int): Number of control points for the curve.
    
    Returns:
        np.ndarray: Binary mask with the curve drawn.
    """
    # Generate random control points
    x = np.linspace(0, w, num_points)
    y = np.random.randint(0, h, num_points)

    # Create a spline representation of the curve
    tck, u = splprep([x, y], s=0)
    u_fine = np.linspace(0, 1, w)
    x_fine, y_fine = splev(u_fine, tck)

    curve_mask = np.zeros((h, w), dtype=np.uint8)
    for i in range(len(x_fine) - 1):
        cv2.line(curve_mask, (int(x_fine[i]), int(y_fine[i])), (int(x_fine[i+1]), int(y_fine[i+1])), 1, thickness=1)

    return curve_mask

def generate_random_binary_segmentation(image, proportion=0.5):
    """
    Generate a random binary segmentation inside an image.
    
    Args:
        image (numpy.ndarray): The input image.
        proportion (float): The proportion of the first zone (0 to 1).
    
    Returns:
        numpy.ndarray: The binary segmentation mask.
    """
    h, w = image.shape[:2]
    binary_mask = np.zeros((h, w), dtype=np.uint8)

    proportion = np.random.uniform(proportion, 0.9)

    while True:
        curve_mask = generate_random_curve(h, w)
        mask = cv2.floodFill(curve_mask.copy(), None, (0, 0), 1)[1]
        area = np.sum(mask)
        if proportion * h * w * 0.95 < area < proportion * h * w * 1.05:
            binary_mask = mask
            break

    return binary_mask

def generate_random_continuous_mask(h, w, proportion=0.5, num_shapes=10, shape_size_range=(30, 100)):
    """
    Generate a random continuous mask with a specified proportion.

    Args:
        h (int): Height of the image.
        w (int): Width of the image.
        proportion (float): The proportion of the first zone (0 to 1).
        num_shapes (int): Number of random shapes to generate.
        shape_size_range (tuple): Range of sizes for the random shapes.

    Returns:
        numpy.ndarray: The binary segmentation mask.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    
    while True:
        #mask.fill(0)
        for _ in range(num_shapes):
            shape_type = np.random.choice(['circle', 'rectangle'])
            size = np.random.randint(shape_size_range[0], shape_size_range[1])
            
            if shape_type == 'circle':
                center = (np.random.randint(0, w), np.random.randint(0, h))
                radius = size // 2
                cv2.circle(mask, center, radius, 1, -1)
            elif shape_type == 'rectangle':
                top_left = (np.random.randint(0, w - size), np.random.randint(0, h - size))
                bottom_right = (top_left[0] + size, top_left[1] + size)
                cv2.rectangle(mask, top_left, bottom_right, 1, -1)
        
        # Ensure continuity with morphological operations
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((21,21), np.uint8))
        
        # Calculate the proportion of the filled area
        area = np.sum(mask)
        #print(area)
        if proportion * h * w  < area:
            break

    mask = cv2.GaussianBlur(mask.astype(np.float32), (5,5), 0)
    
    return mask

def compute_n_shapes_and_shape_size(h,w):
    x = np.sqrt(h*w)
    a11 = 75/(512-128)
    a12 = (30-7.5)/(512-128)
    b11 = 100 - a11*512
    b12 = 30 - a12*512
    a2 = 2/(512-128)
    b2 = 10 - a2*512

    num_shapes = np.round(a2*x + b2).astype(int)
    shape_size_range = (np.round(a12*x + b12).astype(int),np.round(a11*x + b11).astype(int))
    return num_shapes, shape_size_range

def rotate_img(img):
    rot = np.random.choice([0, 90, 180, 270])
    img = np.rot90(img, k=rot//90)
    return img

def load_image(path):
    """Load an image from file."""
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    return image

def normalize_psf(psf):
    """Normalize the PSF so that its sum is 1."""
    return psf / np.sum(psf)

def apply_psf(image, psf):
    """Apply the PSF to the image using convolution."""
    psf_normalized = normalize_psf(psf)
    blurred_image = np.zeros_like(image)
    
    # Convolve each channel separately
    for i in range(3):
        blurred_image[:, :, i] = convolve2d(image[:, :, i], psf_normalized, mode='same', boundary='wrap')
    
    return blurred_image

def blur(image, PSF, show=False):
    # Load the image
    

    

    # Apply the PSF to the image
    blurred_image = apply_psf(image, PSF.psf)
    m, V = np.random.random()*4 -2, np.random.random()*9+1
    #blurred_image = gaussian_white_noise(blurred_image, m, np.sqrt(V))

    if show:
        # Display the original and blurred images
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title('Original Image')
        plt.imshow(image)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title('Blurred Image')
        plt.imshow(blurred_image)
        plt.axis('off')

        plt.show()

    return blurred_image

def generate_blurred_image(image_path, PSF, show = False):
    image = load_image(image_path)
    #image = rotate_img(image)
    blurred_image = blur(image, PSF, show = show)
    x = np.random.choice([0, 1])
    if x==0:
        #binary_segmentation = generate_random_binary_segmentation(blurred_image, proportion=0.8)
        num_shapes, shape_size_range = compute_n_shapes_and_shape_size(image.shape[0], image.shape[1])
        binary_segmentation = generate_random_continuous_mask(image.shape[0], image.shape[1], proportion=0.4, num_shapes=num_shapes, shape_size_range=shape_size_range)
        binary_segmentation = np.stack([binary_segmentation]*3, axis=-1)
        final_img = blurred_image*binary_segmentation + image*(1-binary_segmentation)
    else:
        final_img = blurred_image
    final_img = final_img.astype(np.uint8)
    if show:
        plt.imshow(final_img)
        plt.axis('off')
        plt.show()
    return final_img


class GaussianPSF:
    def __init__(self, sigma=None):
        if sigma is None:
            self.sigma = np.random.uniform(3,5)
        else:
            self.sigma = sigma
        self.size = 6*np.ceil(self.sigma).astype(int)
        #print(f"sigma: {self.sigma}, size: {self.size}")
        self.psf = self._create_psf()

    def _create_psf(self):
        x = np.linspace(-self.size // 2, self.size // 2, self.size)
        y = np.linspace(-self.size // 2, self.size // 2, self.size)
        x, y = np.meshgrid(x, y)
        psf = np.exp(-(x**2 + y**2) / (2 * self.sigma**2))
        return psf / np.sum(psf)

class DefocusPSF:
    def __init__(self, radius=None):
        if radius is None:
            self.radius = np.random.randint(5,15)
        else:
            self.radius = radius
        self.psf = self._create_psf()
    
    def _create_psf(self):
        x = np.linspace(-self.radius, self.radius, 2 * self.radius + 1)
        y = np.linspace(-self.radius, self.radius, 2 * self.radius + 1)
        x, y = np.meshgrid(x, y)
        psf = np.zeros_like(x)
        psf[x**2 + y**2 <= self.radius**2] = 1
        return psf / np.sum(psf)
    
class MotionPSF:
    def __init__(self, length=None, angle=None):
        if length is None:
            self.length = np.random.randint(15,25)
        else:
            self.length = length

        if angle is None:
            self.angle = np.random.uniform(0, np.pi)
        else:
            self.angle = angle

        self.psf = self._create_psf()
    
    def _create_psf(self):
        x = np.linspace(-self.length // 2, self.length // 2, self.length)
        y = np.linspace(-self.length // 2, self.length // 2, self.length)
        x, y = np.meshgrid(x, y)
        threshold = np.abs(x/5)
        threshold[np.where(threshold < np.abs(y/5))] = np.abs(y/5)[np.where(threshold < np.abs(y/5))]
        psf = np.zeros_like(x)
        psf[(np.abs(x * np.sin(self.angle) + y * np.cos(self.angle)) <= threshold) & (x**2 + y**2 <= self.length**2/4)] = 1
        return psf / np.sum(psf)
    
class AnonymizePSF:
    def __init__(self, size=None):
        if size is None:
            size = np.random.randint(25,35)
        self.size = size, size
        self.psf = self._create_psf()

    def _create_psf(self):
        x = np.linspace(-self.size[0] // 2, self.size[0] // 2, self.size[0])
        y = np.linspace(-self.size[1] // 2, self.size[1] // 2, self.size[1])
        x, y = np.meshgrid(x, y)
        psf = np.ones_like(x)
        return psf / np.sum(psf)

class CreatePatchDataset(Dataset):
    def __init__(self, annotations_file, transform=None):
        self.old_img_labels = pd.read_csv(annotations_file, header=None)
        records = []
        for i, (img_path, label) in self.old_img_labels.iterrows():
            img = Image.open(img_path)
            patches = compute_patch_numbers(img)
            for patch in patches:
                records.append({'img_path': img_path, 'label': label, 'top': patch[1], 'left': patch[0]})
        
        self.img_labels = pd.DataFrame.from_records(records) 
        
        assert self.img_labels.duplicated().sum() == 0, 'Duplicated data'

        print(len(self.img_labels))
            
        # print(len(self.img_labels))
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path, label, top, left = self.img_labels.iloc[idx]
        new_img_path = img_path[:-4] + '_'+str(top) + '_' + str(left) + '.jpg'
        # img_path = self.img_labels.iloc[idx, 0]
        image = Image.open(img_path)
        # label = self.img_labels.iloc[idx, 1]
        image = T.functional.crop(image, top, left, 128, 128)
        if self.transform:
            image = self.transform(image)
        
        return image, new_img_path, label

