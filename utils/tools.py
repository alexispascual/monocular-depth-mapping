import os
import cv2
import numpy as np
from pprint import pprint as pp

def check_directory(directory: str):
    if not os.path.isdir(directory):
        print(f"Creating directory {directory}")
        os.makedirs(directory)

def show_image(image, image_2=None):
    if image_2 is not None:
        image = np.concatenate((image, image_2), axis=1)

    cv2.imshow('Image', image)

    key = cv2.waitKey(0)

    if key == 27: # Check for ESC key press
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

def scan_horizon_files(directory: str):

    horizons = []

    for root, dirs, files in os.walk(directory):
        horizons = [f.split('.')[0].split('_')[1] for f in files]

    return horizons

def fill_horizon_line(image):

    pp(image)
    pp(image.shape)
    horizon_line = np.argmax(image, axis=0)

    pp(horizon_line)
    pp(horizon_line.shape)

    height = image.shape[0]
    width = image.shape[1]

    for col in range(width):
        image[0:horizon_line[col][0], col] = (255, 255, 255)

    return image