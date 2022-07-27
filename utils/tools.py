import os
import re
import cv2
import sys
import yaml
import math
import numpy as np

from pprint import pprint as pp


def load_config(default_config: str, silent=False):
    """
    Multi-use function. Exposes command-line to provide alternative configurations to training scripts.
    Also works as a stand-alone configuration importer in notebooks and scripts.
    """
    if len(sys.argv) == 1:
        # Then the file being called is the only argument so return the default configuration
        config_file = default_config
    elif len(sys.argv) == 2:
        # Then a specific configuration file has been used so load it
        config_file = str(sys.argv[1])
    elif all([len(sys.argv) == 3, sys.argv[1] == '-f']):
        config_file = default_config
    else:
        print(sys.argv)
        raise ValueError('CLI only accepts 0 args (default) or 1 arg (path/to/config).')

    with open(str(config_file), 'r') as f:
        y = yaml.full_load(f)
        if not silent:
            print('Experimental parameters\n------')
            pp(y)
        return y


def check_directory(directory: str):
    if not os.path.isdir(directory):
        print(f"Creating directory {directory}")
        os.makedirs(directory)


def show_image(image, image_2=None):
    if image_2 is not None:
        image = np.concatenate((image, image_2), axis=1)

    cv2.imshow('Image', image)

    key = cv2.waitKey(0)

    if key == 27:  # Check for ESC key press
        return -1
    else:
        return 0


def save_image(image, 
               image_2=None, 
               directory='./data', 
               file_name='default_name.jpg'):

    if image_2 is not None:
        image = np.concatenate((image, image_2), axis=1)
    
    file_name = os.path.join(directory, file_name)
    cv2.imwrite(file_name, image)


def scan_horizon_files(directory: str, campaign_2: bool = False):

    horizons = []

    if campaign_2:
        for _, _, files in os.walk(directory):
            horizons = [''.join(re.split(r'(\d+)', f)[1:-1]) for f in files]
    else:
        for _, _, files in os.walk(directory):
            horizons = [f.split('.')[0].split('_')[1] for f in files]

    return horizons


def fill_horizon_line(image):

    copy_image = image.copy()
    copy_image = cv2.flip(image, 0)

    horizon_line = np.argmax(copy_image, axis=0)
    
    height = image.shape[0]
    width = image.shape[1]

    for col in range(width):
        corrected_horizon = horizon_line[col][0] - height - 1
        image[0:corrected_horizon, col] = (255, 255, 255)

    return cv2.flip(image, 0)


def fill_horizon_line_top_down(image):

    horizon_line = np.argmax(image, axis=0)
    
    height = image.shape[0]
    width = image.shape[1]

    for col in range(width):
        image[0:horizon_line[col][0], col] = (0, 0, 0)
        image[horizon_line[col][0] + 1:height, col] = (255, 255, 255)

    return image


def generate_depth_map(point_cloud: np.ndarray, 
                       normalize: bool = False):

    depth_estimate = np.zeros((point_cloud.shape[0], point_cloud.shape[1]))
    point_cloud = np.nan_to_num(point_cloud)

    for w in range(point_cloud.shape[0]):
        for h in range(point_cloud.shape[1]):
            value = point_cloud[w][h]

            if not np.isnan(value.any()):
                depth_estimate[w][h] = math.sqrt(value[0] * value[0] 
                                                 + value[1] * value[1]
                                                 + value[2] * value[2])
            else:
                depth_estimate[w][h] = 0

    if normalize:
        normalized_depth = (depth_estimate - np.min(depth_estimate)) / (np.max(depth_estimate) - np.min(depth_estimate)) * 255

        return normalized_depth.astype(np.uint8)

    else:
        return depth_estimate


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def get_images_from_directory(filepath: str):
    paths = []

    for entry in os.scandir(filepath):
        if (entry.path.endswith(".jpg") or entry.path.endswith(".jpeg")
                or entry.path.endswith(".png")) and entry.is_file():
            paths.append(entry.path)

    paths.sort(key=natural_keys)
    return paths
