import cv2
import numpy as np
import matplotlib.pyplot as ax
from sklearn.mixture import GaussianMixture as GMM
from scipy.optimize import curve_fit
from tqdm import tqdm
from skimage.segmentation import slic as skimage_slic
from scipy.stats import entropy as scipy_entropy
import matplotlib.pyplot as plt
import cupy as cp

def get_patch_region_size(image):
    """ Get patch size for the image
    - Args:
        - image (cupy.ndarray): image to get patch size
    """
    n,p = image.shape[:2]
    n = max(n,p)
    n= n//100 +1
    n = n*100
    patch_size = n//250 +1
    patch_size*=5
    region_size = patch_size-10
    return patch_size, region_size



def convert_to_LAB(img):
    """ Convert image to LAB color space
    - Args:
        - img (numpy.ndarray): RGB image to convert
    """
    img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    LAB = cp.asarray(img)

    return LAB


def rgb_to_gray(carray):
    """Convert cupy array to gray_scale
    - Args:
        - carray (cupy.ndarray): RGB image
    """
    gray = carray[:,:,0]*0.299 + carray[:,:,1]*0.587 + carray[:,:,2]*0.114
    print(gray.shape)
    return gray

def divide_into_patches(image: cp.ndarray, patch_size: int, overlap: int) -> list:
    """Divide the image into overlapping patches.

    Args:
        image (cupy.ndarray): Grayscale image.
        patch_size (int): Size of each patch.
        overlap (int): Overlap between patches.

    Returns:
        list: List of patches.
    """
    h, w = image.shape[:2]
    patches = []
    for y in range(0, h - patch_size + 1, patch_size - overlap):
        for x in range(0, w - patch_size + 1, patch_size - overlap):
            patches.append((y, y + patch_size, x, x + patch_size))
    return patches

def divide_patch_into_regions(patch: cp.ndarray, region_size: int, overlap: int) -> list:
    """Divide a patch into overlapping regions.

    Args:
        patch (cp.ndarray): Grayscale patch.
        region_size (int): Size of each region.
        overlap (int): Overlap between regions.

    Returns:
        list: List of regions.
    """
    h, w = patch.shape[:2]
    regions = []
    for y in range(0, h - region_size + 1, region_size - overlap):
        for x in range(0, w - region_size + 1, region_size - overlap):
            regions.append(patch[y:y + region_size, x:x + region_size])
    return regions


def compute_gradients(image: np.ndarray) -> tuple:
    """Compute gradients of the image in x and y directions.

    Args:
        image (numpy.ndarray): Grayscale image to compute gradients.

    Returns:
        tuple: cupy gradients in x and y directions.
    """
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    return cp.asarray(grad_x), cp.asarray(grad_y)

def compute_gradient_magnitude(grad_x: cp.ndarray, grad_y: cp.ndarray) -> cp.ndarray:
    """Compute the gradient magnitude.

    Args:
        grad_x (cupy.ndarray): Gradient in x direction.
        grad_y (cupy.ndarray): Gradient in y direction.

    Returns:
        cupy.ndarray: Gradient magnitude.
    """
    return cp.sqrt(grad_x**2 + grad_y**2)

def compute_sigma(image: cp.ndarray) -> float:
    """Compute sigma for the Gaussian kernel.

    Args:
        image (cupy.ndarray): Grayscale image.

    Returns:
        float: Estimated sigma.
    """
    img = image.get()
    grad_x, grad_y = compute_gradients(image)
    grad_magnitude = compute_gradient_magnitude(grad_x, grad_y)
    gmm = GMM(n_components=2, random_state=42)
    gmm.fit(grad_magnitude.flatten().reshape(-1, 1)/255.0)
    return gmm.covariances_.flatten()[1]

def compute_contrast(image: cp.ndarray) -> float:
    """Compute contrast of the image.

    Args:
        image (cupy.ndarray): Grayscale image.

    Returns:
        float: Image contrast.
    """
    min_val, max_val = cp.min(image).astype(cp.float32), cp.max(image).astype(cp.float32) 
    #print(min_val, max_val)
    contrast = (max_val - min_val) / (max_val + min_val + 1e-6)
    #print(contrast)
    return contrast

def compute_Q(image: cp.ndarray, patch_size: int = 10, region_size: int = 5, overlap: int = 0) -> cp.ndarray:
    """Compute Q matrix for the image.

    Args:
        image (cupy.ndarray): RGB image.
        patch_size (int): Size of each patch.
        region_size (int): Size of each region.
        overlap (int): Overlap between patches and regions.

    Returns:
        cupy.ndarray: Q matrix.
    """
    #image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image_gray = rgb_to_gray(image)
    #print(image_gray)
    h, w = image_gray.shape
    Q = cp.zeros((h, w))

    patches = divide_into_patches(image_gray, patch_size, overlap)
    contrasts = []
    for (y_min, y_max, x_min, x_max) in tqdm(patches):
        patch = image_gray[y_min:y_max, x_min:x_max]
        regions = divide_patch_into_regions(patch, region_size, 4)
        sigma = compute_sigma(patch)
        max_contrast = max(compute_contrast(region) for region in regions)
        contrasts.append(max_contrast)
        
        if max_contrast == 0:
            max_contrast = 1e-3
        Q[y_min:y_max, x_min:x_max] = sigma / max_contrast
        #clear_output(wait=True)
        #break

    return Q

# Example usage:
# image = cv2.imread('path/to/your/image.jpg')
# Q_matrix = compute_Q(image)
# ax.imshow(Q_matrix, cmap='hot')
# ax.title('Q Matrix')
# ax.show()


def compute_saturation(pixel: cp.ndarray) -> float:
    """Compute saturation for a pixel.

    Args:
        pixel (cupy.ndarray): RGB pixel.

    Returns:
        float: Saturation of the pixel.
    """
    x = cp.sum(pixel)
    y = cp.min(pixel)
    return 1 - (3 / x) * y if x != 0 else 1

def compute_saturation_matrix(image: cp.ndarray) -> cp.ndarray:
    """Compute saturation matrix for the image.

    Args:
        image (cupy.ndarray): RGB image.

    Returns:
        cp.ndarray: Saturation matrix.
    """
    sums = cp.sum(image, axis=2)
    mins = cp.min(image, axis=2)
    with cp._cupyx.errstate(divide='ignore', invalid='ignore'):
        saturation = 1 - (3 / sums) * mins
        saturation[sums == 0] = 1
    return saturation

def divide_into_patches(image: cp.ndarray, patch_size: int, overlap: int) -> list:
    """Divide the image into overlapping patches.

    Args:
        image (cupy.ndarray): Image to divide.
        patch_size (int): Size of each patch.
        overlap (int): Overlap between patches.

    Returns:
        list: List of patches.
    """
    h, w = image.shape[:2]
    patches = []
    step = patch_size - overlap
    for y in range(0, h - patch_size + 1, step):
        for x in range(0, w - patch_size + 1, step):
            patches.append((y, y + patch_size, x, x + patch_size))
    return patches

def compute_Sa(image: cp.ndarray, patch_size: int = 5, overlap: int = 0) -> cp.ndarray:
    """Compute maximum saturation matrix for the image.

    Args:
        image (cupy.ndarray): RGB image.
        patch_size (int): Size of each patch.
        overlap (int): Overlap between patches.

    Returns:
        cp.ndarray: Maximum saturation matrix.
    """
    sat = compute_saturation_matrix(image)
    max_sat = cp.max(sat)

    h, w = image.shape[:2]
    Sa_patches = cp.zeros((h, w))

    patches = divide_into_patches(image, patch_size, overlap)
    for (y_min, y_max, x_min, x_max) in patches:
        patch_saturation = sat[y_min:y_max, x_min:x_max]
        max_patch_saturation = cp.max(patch_saturation)
        Sa_patches[y_min:y_max, x_min:x_max] = max_patch_saturation

    return Sa_patches / max_sat -1


def linear_fit(x, a, b):
    return a * x + b

def compute_alpha(img):
    """ Compute local power spectrum slope for the image
    - Args:
        - img (cupy.ndarray): RGB image
    """
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = rgb_to_gray(img)
    f = cp.fft.fft2(img)
    S = cp.abs(f)**2/(f.shape[0]*f.shape[1])
    polar_S = {}
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            r = int(cp.sqrt((i)**2 + (j)**2))
            #print(r)
            if r not in polar_S:
                polar_S[r] = 0
            polar_S[r]+=(S[i,j])
    frequencies = cp.array([r for r in polar_S if r != 0 and polar_S[r] != 0])
    ps = cp.array([polar_S[r] for r in frequencies])
    #return frequencies, ps

    log_frequencies = cp.log(frequencies)
    log_ps = cp.log(ps)

    if len(log_frequencies) == 0:
        return 1
    

    popt, _ = curve_fit(linear_fit, log_frequencies, log_ps)
    alpha = -popt[0]
    return alpha

def compute_alpha_matrix(img, patch_size=20):
    """ Compute local power spectrum slope matrix for the image
    - Args:
        - img (cupy.ndarray): RGB image
    """
    alpha_0 = compute_alpha(img)
    print(alpha_0)
    h, w = img.shape[:2]
    alpha_matrix = cp.zeros((h, w))

    patches = divide_into_patches(img, patch_size, 0)
    for (y_min, y_max, x_min, x_max) in patches:
        patch = img[y_min:y_max, x_min:x_max]
        alpha = compute_alpha(patch)
        alpha_matrix[y_min:y_max, x_min:x_max] = alpha
    _a = cp.nan_to_num(alpha_matrix)
    mini = cp.min(alpha_matrix[alpha_matrix>0].flatten())
    _a = cp.clip(_a, mini, max(_a.flatten()))
    return _a


def normalize(image):
    """ Normalize image to have values between 0 and 1
    - Args:
        - image (cupy.ndarray): image to normalize
    """
    return (image - cp.min(image)) / (cp.max(image) - cp.min(image))


def blur_distance_matrix(img, patch_size=20, region_size=10):
    """ Blur distance matrix D with a gaussian kernel
    - Args:
        - image (cupy.ndarray): RGB image
    """
    #img = np.array(image)
    h, w = img.shape[:2]

    alpha_0 = compute_alpha(img)
    print(f"alpha_0: {alpha_0}")
    alpha_matrix = cp.zeros((h, w))
    
    sat = compute_saturation_matrix(img)
    max_sat = cp.max(sat)
    Sa_patches = cp.zeros((h, w))

    #image_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    image_gray = rgb_to_gray(img)
    #print(image_gray)
    Q = cp.zeros((h, w))

    patches = divide_into_patches(img, patch_size, 0)
    for (y_min, y_max, x_min, x_max) in tqdm(patches):
        #Power spectrum slope
        patch = img[y_min:y_max, x_min:x_max]
        alpha = compute_alpha(patch)
        alpha_matrix[y_min:y_max, x_min:x_max] = alpha
        
        #Saturation
        patch_saturation = sat[y_min:y_max, x_min:x_max]
        max_patch_saturation = cp.max(patch_saturation)
        Sa_patches[y_min:y_max, x_min:x_max] = max_patch_saturation

        #Gradient histogram
        patch = image_gray[y_min:y_max, x_min:x_max]
        n_regions = patch_size//region_size+1
        overlap = (n_regions*region_size - patch_size) // (n_regions-1)
        regions = divide_patch_into_regions(patch, region_size, overlap)
        sigma = compute_sigma(patch)
        max_contrast = max(compute_contrast(region) for region in regions)
        #contrasts.append(max_contrast)
        if max_contrast == 0:
            max_contrast = 1e-3
        Q[y_min:y_max, x_min:x_max] = sigma / max_contrast

    _a = cp.nan_to_num(alpha_matrix)
    mini = cp.min(alpha_matrix[alpha_matrix>0].flatten())
    _a = cp.clip(_a, mini, max(_a.flatten()))
        

    Sa = Sa_patches / max_sat -1


    D = normalize(normalize(Q) + 0.5*normalize(Sa) + normalize(1/cp.sqrt(_a)))
    return D


def scale_8_bits(image):
    """ Scale image to 8 bits
    - Args:
        - image (cupy.ndarray): image to scale
    """
    S = (image - cp.min(image)) / (cp.max(image) - cp.min(image)) * 255
    S = S.astype(cp.uint8)
    return S

def generate_superpixel(img, patch_size=20, region_size=10):
    """ Generate superpixels for the image
    - Args:
        - img (cupy.ndarray): RGB image
    """
    D = blur_distance_matrix(img, patch_size=patch_size, region_size=region_size)
    S = scale_8_bits(D)
    LAB_img = convert_to_LAB(img)
    dist_img = cp.concatenate([LAB_img, S.reshape(S.shape[0], S.shape[1], 1)], axis=2)
    segments = skimage_slic(dist_img, n_segments=150, compactness=0.5)

    return segments, S

def superpixel_plot(im,seg,title = "Superpixels"):
    """
    Given an image (nXmX3) and pixelwise class mat (nXm),
    1. Consider each class as a superpixel
    2. Calculate mean superpixel value for each class
    3. Replace the RGB value of each pixel in a class with the mean value

    Inputs:
    im: Input image
    seg: Segmentation map
    title: Title of the plot

    Output: None
    Creates a plot
    """
    clust = cp.unique(seg)
    mapper_dict = {i: im[seg == i].mean(axis = 0)/255. for i in clust}

    seg_img =  cp.zeros((seg.shape[0],seg.shape[1],3))
    for i in clust:
        seg_img[seg == i] = mapper_dict[i]

    return seg_img

def entropy_map(gray_img, segments):
    """ Compute the normalized entropy map for the image
    - Args:
        - gray_img (cupy.ndarray): GRAY image
        - segments (cupy.ndarray): superpixel segments
    """
    entropy = {}
    n_segments = cp.max(segments)
    for i in range(n_segments):
        x = cp.array(gray_img)[segments == i+1]

        _, counts = cp.unique(x, return_counts=True)
        entropy[i+1]= scipy_entropy(counts)

    entropy_map = cp.zeros_like(gray_img, dtype=cp.float32)
    for i in range(n_segments):
        entropy_map[segments == i+1] = entropy[i+1]
    
    return normalize(entropy_map)

def SVD_map(gray_img, segments, threshold = 6):
    """ Compute the normalized map for the image
    - Args:
        - gray_img (cupy.ndarray): GRAY image
        - segments (cupy.ndarray): superpixel segments
        - threshold (int): threshold for significant singular values
    """
    SVD_metric = {}
    n_segments = cp.max(segments)
    for i in range(n_segments):

        mask = segments == i+1
        #print(mask)

        # Extract the bounding box of the superpixel
        y, x = cp.where(mask)
        min_x, max_x = cp.min(x), cp.max(x)
        min_y, max_y = cp.min(y), cp.max(y)
        sq_superpixel = gray_img[min_y:max_y, min_x:max_x]
        U,S,V = cp.linalg.svd(sq_superpixel)
        SVD_metric[i+1] = cp.sum(S[:6])/cp.sum(S)

    SVD_map = cp.zeros_like(gray_img, dtype=cp.float32)
    for i in range(n_segments):
        SVD_map[segments == i+1] = SVD_metric[i+1]
    
    return SVD_map

def classify_patches(image, segments, D, SVD_threshold = 0.93, entropy_threshold = 0.8, D_threshold = 0.2, show = False):
    """ Classify patches of the image
    - Args:
        - image (cupy.ndarray): RGB image
        - segments (cupy.ndarray): superpixel segments
        - SVD_threshold (float): threshold for SVD_metric
        - entropy_threshold (float): threshold for entropy_metric
        - D_threshold (float): threshold for blur_metric
    """
    #gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_img = rgb_to_gray(image)
    SVD = SVD_map(gray_img, segments)
    #entropy = entropy_map(gray_img, segments)
    entropy = cp.zeros_like(gray_img, dtype=cp.float32)
    n_segments = cp.max(segments)
    patch_labels = {}
    if show:
        D_map = cp.zeros_like(gray_img, dtype=cp.float32)
    for i in range(n_segments):
        SVD_metric = SVD[segments == i+1].mean()
        entropy_metric = entropy[segments == i+1].mean()
        blur_metric = D[segments == i+1].mean()
        
        if show:
            D_map[segments == i+1] = blur_metric
        if SVD_metric > SVD_threshold and entropy_metric < entropy_threshold and blur_metric < D_threshold*255:
            patch_labels[i+1] = 1
        else:
            patch_labels[i+1] =0
        
        

    classes = cp.zeros_like(gray_img, dtype=cp.float32)
    for i in range(n_segments):
        classes[segments == i+1] = patch_labels[i+1]
        
    return patch_labels, classes, SVD, entropy

def slic(image, entropy_threshold = 0.8, SVD_threshold = 0.95, blur_threshold = 0.2, show = False):
    """ Generate superpixels for the image
    - Args:
        - image (PIL.Image): RGB image
    """
    #image = np.array(image)
    image = cp.asarray(image)
    patch_size, region_size = get_patch_region_size(image)
    print(patch_size, region_size)
    segments, D = generate_superpixel(image, patch_size=patch_size, region_size=region_size)

    patch_labels, classes, SVD , entropy = classify_patches(image, segments, D, show = show, SVD_threshold = SVD_threshold, entropy_threshold = entropy_threshold, D_threshold = blur_threshold)

    if show:
        fig, ax = plt.subplots(1, 6, figsize=(35, 5))

        ax[0].imshow(image.get())
        ax[0].set_title('Original Image')

        im = ax[1].imshow(D.get(), cmap='hot')
        ax[1].set_title('Blur Metric')
        plt.colorbar(im, ax=ax[1])

        im = ax[2].imshow(entropy.get(), cmap='hot')
        ax[2].set_title('Entropy Metric')
        plt.colorbar(im, ax=ax[2])

        im = ax[3].imshow(SVD.get(), cmap='hot')
        ax[3].set_title('SVD Metric')
        plt.colorbar(im, ax=ax[3])
        
        ax[4].imshow(superpixel_plot(image, segments).get())
        ax[4].set_title('Superpixels')

        ax[5].imshow(image.get())
        ax[5].imshow(classes.get(), cmap='gray', alpha=0.5)
        ax[5].set_title('Classified Image')

        plt.show()
    return segments, patch_labels, classes, SVD, D
    



