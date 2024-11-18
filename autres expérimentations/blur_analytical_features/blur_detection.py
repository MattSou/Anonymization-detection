import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_blur_degree(image_file, sv_num=10):
    """
    Get the blur degree of an image using Singular Value Decomposition
    Args:
        image_file: input image file
        sv_num: number of singular values to consider
    Returns:
        top_sv/total_sv
    """
    img = cv2.imread(image_file,cv2.IMREAD_GRAYSCALE)
    u, s, v = np.linalg.svd(img)
    top_sv = np.sum(s[0:sv_num])
    total_sv = np.sum(s)
    return top_sv/total_sv


def get_blur_map(img, win_size=10, sv_num=3):
    """
    Get the blur map of an image using Singular Value Decomposition
    Args:
        img: input image
        win_size: size of the window
        sv_num: number of singular values to consider
    Returns:
        blur_map
    """
    new_img = np.zeros((img.shape[0]+win_size*2, img.shape[1]+win_size*2))
    for i in range(new_img.shape[0]):
        for j in range(new_img.shape[1]):
            if i<win_size:
                p = win_size-i
            elif i>img.shape[0]+win_size-1:
                p = img.shape[0]*2-i
            else:
                p = i-win_size
            if j<win_size:
                q = win_size-j
            elif j>img.shape[1]+win_size-1:
                q = img.shape[1]*2-j
            else:
                q = j-win_size
            #print p,q, i, j
            new_img[i,j] = img[p,q]

    #cv2.imwrite('test.jpg', new_img)
    #cv2.imwrite('testin.jpg', img)
    blur_map = np.zeros((img.shape[0], img.shape[1]))
    max_sv = 0
    min_sv = 1
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            block = new_img[i:i+win_size*2, j:j+win_size*2]
            u, s, v = np.linalg.svd(block)
            top_sv = np.sum(s[0:sv_num])
            total_sv = np.sum(s)
            sv_degree = top_sv/total_sv
            if max_sv < sv_degree:
                max_sv = sv_degree
            if min_sv > sv_degree:
                min_sv = sv_degree
            blur_map[i, j] = sv_degree
    #cv2.imwrite('blurmap.jpg', (1 - blur_map) * 255)

    blur_map = (blur_map-min_sv)/(max_sv-min_sv)
    #cv2.imwrite('blurmap_norm.jpg', (1-blur_map)*255)
    return blur_map


def get_blur_map2(img, win_size=10, sv_num=3):
    """
    Get the blur map of an image using Singular Value Decomposition
    Args:
        img: input image
        win_size: size of the window
        sv_num: number of singular values to consider
    Returns:
        blur_map
    """
    img_padded = np.pad(img, win_size, mode='reflect')  # Ajoute un rembourrage réfléchissant pour éviter les problèmes de bord

    blur_map = np.zeros_like(img, dtype=np.float64)
    max_sv = 0
    min_sv = 1

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            block = img_padded[i:i+win_size*2, j:j+win_size*2]
            u, s, v = np.linalg.svd(block)
            top_sv = np.sum(s[:sv_num])
            total_sv = np.sum(s)
            sv_degree = top_sv / total_sv
            blur_map[i, j] = sv_degree

            max_sv = max(max_sv, sv_degree)
            min_sv = min(min_sv, sv_degree)

    blur_map = (blur_map - min_sv) / (max_sv - min_sv)
    return blur_map


def imshow_cv2_img(img):
    """
    Display a cv2 image using matplotlib
    Args:
        img: input image
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)


def get_motion_blur_kernel(x, y, thickness=1, ksize=21):
    """ Obtains Motion Blur Kernel
        Inputs:
            x - horizontal direction of blur
            y - vertical direction of blur
            thickness - thickness of blur kernel line
            ksize - size of blur kernel
        Outputs:
            blur_kernel
        """
    blur_kernel = np.zeros((ksize, ksize))
    c = int(ksize/2)

    blur_kernel = np.zeros((ksize, ksize))
    blur_kernel = cv2.line(blur_kernel, (c+x,c+y), (c,c), (255,), thickness)
    return blur_kernel/blur_kernel.sum()


def add_motion_blur(img, k_size, theta, thickness = 5):
    b = get_motion_blur_kernel(int(k_size/2*np.cos(theta)), int(k_size/2*np.sin(theta)), thickness=4, ksize = k_size)
    plt.imshow(b)
    plt.title('Blur kernel')
    plt.show()

    imshow_cv2_img(img)
    plt.title('Sharp image')
    plt.show()
    blurred = cv2.filter2D(img, ddepth=-1, kernel=b)

    imshow_cv2_img(blurred)
    plt.title('Blurred image')
    plt.show()

    return blurred

def add_circular_blur(img, k_size):
    circle_filter = cv2.circle(np.zeros((k_size+1, k_size+1)), center=(k_size//2, k_size//2), radius=k_size//2, color=(255,), thickness=-1)
    circle_filter = circle_filter/circle_filter.sum()
    plt.imshow(circle_filter)
    plt.title('Blur kernel')
    plt.show()

    imshow_cv2_img(img)
    plt.title('Sharp image')
    plt.show()
    blurred = cv2.filter2D(img, ddepth=-1, kernel=circle_filter)

    imshow_cv2_img(blurred)
    plt.title('Blurred image')
    plt.show()

    return blurred