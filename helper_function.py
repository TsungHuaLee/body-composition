import numpy as np
import pydicom
import matplotlib.pyplot as plt
import cv2
import math

"""
    for generate L3 train data
"""

def img_norm(img, new_max = 255, new_min = 0):
    """
    Args:
      img: An numpy array.
      new_max: Max range after normalization.
      new_min: Min range after normalization.
    Returns:
      normalized image
    """
    normed_img = (img - img.min()) * ((new_max - new_min) / (img.max() - img.min())) + new_min
    return normed_img

def crop_center(img, cropped_height = -1, cropped_width = -1):
    """
    Args:
      img: An 2D numpy array with (height, width)
      cropped_height: int, -1 do nothing
      cropped_width: int, -1 do nothing
    Returns:
      cropped image
    """
    (height, width) = img.shape

    if(cropped_width == -1):
        start_width = 0
        cropped_width = width
    else:
        start_width = width//2-(cropped_width//2)
    if(cropped_height == -1):
        start_height = 0
        cropped_height = height
    else:
        start_height = height//2-(cropped_height//2)

    cropped_img = img[start_height:start_height+cropped_height, start_width:start_width+cropped_width]
    return cropped_img

def resize_wo_interpolation(img, resized_height = -1, resized_width = -1):
    """
    Args:
      img: An 2D numpy array with (height, width)
      resized_height: int, -1 do nothing
      resized_width: int, -1 do nothing
    Returns:
      resized image
    """
    (height, width) = img.shape
    if(resized_width == -1):
        resized_image = img
    elif(width > resized_width and resized_width != -1):
        resized_image = crop_center(img, -1, resized_width)
    else:
        diff_w = resized_width - width
        resized_image = np.pad(img, ((0, 0), (int(diff_w/2), diff_w - int(diff_w/2))), 'constant', constant_values = (0,0))

    return resized_image

def preprocess(img):
    """
    Args:
      img: An 2D numpy array with (height, width)
    Returns:
      resized image
    """
    threshold = np.where(img < 3000, img, 0)
    normed = img_norm(threshold, 255, 0)
    return normed


def visualize_L3(img, line_height = [0], name = ""):
    """
    Args:
      img: An 2D numpy array with (height, width)
      line height: list, height idx which you want to display
      name: str, plt name
    Returns:
      resized image
    """
    plt.figure(figsize=(8,8))
    plt.title("{}\tshape:{}x{}".format(name, img.shape[0], img.shape[1]))
    plt.imshow(img, cmap="gray")
        
    for height in line_height:
        plt.plot([0, img.shape[1]], [height, height])
        
        
def gaussian_heatmap_value(yx = (0, 0), center_yx = (0, 0), sigma = 3.14, C = 1e3):
    fraction = -1 * (math.pow(yx[0] - center_yx[0], 2) + math.pow(yx[1] - center_yx[1], 2))
    denominator = 2 * math.pow(sigma, 2) + 1e-9
    guassian = C * math.exp(fraction/denominator)
    return guassian

def guassian_heatmap_gt(img_shape = (0, 0), center_yx = (0, 0), **kwargs):
    """
    Args:
      img_shape: tuple, image shape of 2d image
      center_yx: tuple, center of guassian heatmap
      sigma: float, width of heatmap
      C: int, scaling constant
    Returns:
      guassian heatmap
    """
    guassian_map = np.zeros(img_shape)
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            guassian_map[i][j] = gaussian_heatmap_value(yx = (i,j), center_yx = center_yx, **kwargs)
    return guassian_map

def exponential_heatmap_value(yx = (0, 0), center_yx = (0, 0), sigma = 3.14, C = 1e3):
    alpha = math.sqrt(math.log(2, 10)/2)/sigma
    power = -1 * alpha * (abs(yx[0] - center_yx[0]) + abs(yx[1] - center_yx[1]))
    guassian = C * math.exp(power)
    return guassian

def exponential_heatmap_gt(img_shape = (0, 0), center_yx = (0, 0), **kwargs):
    """
    Args:
      img_shape: tuple, image shape of 2d image
      center_yx: tuple, center of guassian heatmap
      sigma: float, width of heatmap
      C: int, scaling constant
    Returns:
      guassian heatmap
    """
    exponential_map = np.zeros(img_shape)
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            exponential_map[i][j] = exponential_heatmap_value(yx = (i,j), center_yx = center_yx, **kwargs)
    return exponential_map