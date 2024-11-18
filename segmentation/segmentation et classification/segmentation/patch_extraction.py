import numpy as np
import matplotlib.pyplot as plt
import os
import sys

script_dir = os.path.dirname( __file__ )
slic_dir = os.path.join( script_dir, '..', 'slic' )
sys.path.append(slic_dir)
from slic import slic
import warnings

def compute_integral_image(binary_image):
    integral_image = np.cumsum(np.cumsum(binary_image, axis=0), axis=1)
    return integral_image

def sum_square(integral_image, top_left, size):
    i, j = top_left
    if i == 0 and j == 0:
        return integral_image[size-1][size-1]
    elif i == 0:
        return integral_image[size-1][j+size-1] - integral_image[size-1][j-1]
    elif j == 0:
        return integral_image[i+size-1][size-1] - integral_image[i-1][size-1]
    else:
        return (integral_image[i+size-1][j+size-1]
                - integral_image[i-1][j+size-1]
                - integral_image[i+size-1][j-1]
                + integral_image[i-1][j-1])

def find_max_square(binary_image, size):
    rows, cols = binary_image.shape
    integral_image = compute_integral_image(binary_image)
    max_sum = 0
    max_top_left = (0, 0)
    
    for i in range(rows - size + 1):
        for j in range(cols - size + 1):
            current_sum = sum_square(integral_image, (i, j), size)
            if current_sum > max_sum:
                max_sum = current_sum
                max_top_left = (i, j)
    
    return max_top_left, max_sum

def find_max_squares(binary_image, SVD_image, blur_image, size, patch_labels, segments):
    classes= slic.normalize(SVD_image.copy())+1-slic.normalize(blur_image.copy())+0.25*slic.normalize(binary_image.copy())
    plt.imshow(classes, cmap='gray')
    plt.title('Final superpixels blur score')
    plt.colorbar()
    plt.show()
    _classes = classes.copy()
    max_squares = []
    blur_patches = [patch_n for patch_n in  patch_labels if patch_labels[patch_n]==1]
    select_patches = []
    n=1
    while len(select_patches)<len(blur_patches) and n!=len(select_patches):
        n = len(select_patches)
        max_square, max_sum = find_max_square(classes, size)
        max_squares.append((max_square, max_sum))
        i, j = max_square
        classes[i:i+size, j:j+size] = -5
        sp= np.unique(segments[i:i+size, j:j+size].flatten())
        select_patches+= [patch_n for patch_n in sp if patch_n in blur_patches and patch_n not in select_patches]
        #print(len(select_patches))
    if len(max_squares)<=1:
        print(f'{len(max_squares)} square of size {size} found')
        return max_squares, _classes
    else :
        print(f'{len(max_squares)-1} squares of size {size} found')
        return max_squares[:-1], _classes

def plot_max_squares(img, classes_map, max_squares, size):
    plt.imshow(img)
    image= np.array(img)
    plt.imshow(classes_map, alpha=0.5, cmap='gray')
    for (i, j), _ in max_squares:
        plt.gca().add_patch(plt.Rectangle((j, i), size, size, linewidth=1, edgecolor='r', facecolor='none', alpha=0.5))
    plt.axis('off')
    plt.show()

    n_square = len(max_squares)
    squares = []
    
    if n_square>=1:
        fig, axs = plt.subplots(1, n_square, figsize=(5*n_square, 5))
        if n_square == 1:
            axs = [axs]
        for i, (top_left, _) in enumerate(max_squares):
            x, y = top_left
            axs[i].imshow(image[x:x+size, y:y+size])
            axs[i].axis('off')
            squares.append(image[x:x+size, y:y+size])
        plt.show()

    return squares

def produce_patches(img):
    warnings.filterwarnings('ignore')
    img_size = min(img.size[0], img.size[1])
    size = max(128, img_size//3)
    
    segments, patch_labels, binary, SVD, D =slic.slic(img, SVD_threshold = 0.95, blur_threshold = 0.2, show = True)
    max_squares, classes = find_max_squares(binary.copy(),SVD,D, size, patch_labels, segments)
    squares = plot_max_squares(img,binary, max_squares, size)

    return segments, patch_labels, binary, SVD, D, max_squares, squares, classes