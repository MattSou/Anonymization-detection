import cv2
import numpy as np
from pymatting import estimate_alpha_cf
import matplotlib.pyplot as plt
from matplotlib.image import imread
import sys
import os

def checkImage(image):
    """
    Args:
        image: input image to be checked
    Returns:
        binary image
    Raises:
        RGB image, grayscale image, all-black, and all-white image

    """
    if len(image.shape) > 2:
        print("ERROR: non-binary image (RGB)"); sys.exit();

    smallest = image.min(axis=0).min(axis=0) # lowest pixel value: 0 (black)
    largest  = image.max(axis=0).max(axis=0) # highest pixel value: 1 (white)

    if (smallest == 0 and largest == 0):
        print("ERROR: non-binary image (all black)"); sys.exit()
    elif (smallest == 255 and largest == 255):
        print("ERROR: non-binary image (all white)"); sys.exit()
    elif (smallest > 0 or largest < 255 ):
        print("ERROR: non-binary image (grayscale)"); sys.exit()
    else:
        return True
    

def trimap(image, name, size, path, dilations = 1, erosions=False):
    """
    This function creates a trimap based on simple dilation algorithm
    Inputs [4]: a binary image (black & white only), name of the image, dilation pixels
                the last argument is optional; i.e., how many iterations will the image get eroded
    Output    : a trimap
    """
    checkImage(image)
    row    = image.shape[0]
    col    = image.shape[1]
    pixels = 2*size + 1      ## Double and plus 1 to have an odd-sized kernel
    kernel = np.ones((pixels,pixels),np.uint8)/(pixels*pixels)   ## Pixel of extension I get

    if erosions is not False:
        print('erosion')
        erosions = int(erosions)
        erosion_kernel = np.ones((3,3), np.uint8)/9                     ## Design an odd-sized erosion kernel
        image = cv2.erode(image, erosion_kernel, iterations=erosions)  ## How many erosion do you expect
        image = np.where(image > 0, 255, image)                       ## Any gray-clored pixel becomes white (smoothing)
        # Error-handler to prevent entire foreground annihilation
        if cv2.countNonZero(image) == 0:
            print("ERROR: foreground has been entirely eroded")
            sys.exit()

    dilation  = cv2.dilate(image, kernel, iterations = dilations)

    dilation  = np.where(dilation == 255, 127, dilation) 	## WHITE to GRAY
    remake    = np.where(dilation != 127, 0, dilation)		## Smoothing
    remake    = np.where(image > 127, 200, dilation)		## mark the tumor inside GRAY

    remake    = np.where(remake < 127, 0, remake)		## Embelishmentcv2.NORM_MINMAX, dtype=cv2.CV_32F
    remake    = np.where(remake > 200, 0, remake)		## Embelishment
    remake    = np.where(remake == 200, 255, remake)		## GRAY to WHITE

    #############################################
    # Ensures only three pixel values available #
    # TODO: Optimization with Cython            #
    #############################################    
    for i in range(0,row):
        for j in range (0,col):
            if (remake[i,j] != 0 and remake[i,j] != 255):
                remake[i,j] = 127

    new_name = '{}px_'.format(size) + name
    print(cv2.imwrite(os.path.join(path, new_name) , remake))
    return new_name

def grad_pixel(map, i, j):
    """
    This function calculates the gradient of a pixel in a map
    Inputs [3]: a map (image), pixel's row index, pixel's column index
    Output    : gradient of the pixel
    """
    h,w = map.shape
    grad_x, grad_y = 0, 0
    if i==0:
        grad_x = map[i+1,j]-map[i,j]
    elif i == h-1:
        grad_x = map[i,j]-map[i-1,j]
    else :
        grad_x = (map[i+1,j]-map[i-1,j])/2
    
    if j==0:
        grad_y = map[i,j+1]-map[i,j]
    elif j == w-1:
        grad_y = map[i,j]-map[i,j-1]
    else :
        grad_y = (map[i,j+1]-map[i,j-1])/2

    return np.array([grad_x, grad_y])


def alpha_gradient(alpha_map):
    """
    This function calculates the gradient of an alpha map
    Inputs [1]: alpha map
    Output    : gradient of the alpha map
    """
    h,w = alpha_map.shape
    grad = np.zeros((h,w,2))
    for i in range(h):
        for j in range(w):
            grad[i,j] = grad_pixel(alpha_map, i, j)

    return grad


def alpha_map(img, out_name, size, dilations = 1, erosions = False):
    """
    This function creates an alpha map based on the input image
    Inputs [3]: an image, name of the image, size of the dilation
    Output    : an alpha map
    """
    cv_img = cv2.normalize(img,None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    trimaps_path = '/home/msouda/Workspace/Blur_detection_classification/trimaps/'
    size    = size; # how many pixel extension do you want to dilate
    number  = 1;  # numbering purpose 
    new_name = trimap(cv_img, out_name, size, trimaps_path, dilations=dilations, erosions = erosions)

    image_file = img/255
    trimap_file = imread(trimaps_path + new_name)/255

    n,p = img.shape

    alpha = estimate_alpha_cf(image_file.reshape((n,p,1)), trimap_file)

    plt.imshow(alpha)
    plt.show()

    alpha_grad = alpha_gradient(alpha)

    plt.imshow((np.abs(alpha_grad[:,:,0])), cmap='Greys')
    plt.show()
    plt.imshow(alpha_grad[:,:,1], cmap='Greys')
    plt.show()

    plt.imshow(alpha_grad.mean(axis = -1), cmap='Greys')

    plt.show()

    return alpha, alpha_grad
