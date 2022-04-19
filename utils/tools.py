import os
import cv2

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

    for root, dirs, files in os.walk(data_directory):
        horizons = [f.split('.')[0].split('_')[1] for f in files]

    return horizons