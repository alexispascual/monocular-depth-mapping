import os
import cv2
import numpy as np
import math

from pprint import pprint as pp
from tqdm import tqdm

def main():
    data_directory = './downloads/moon_yard'
    show_image = False
    save_image = True

    for root, dirs, files in os.walk(data_directory):
        for directory in tqdm(dirs):
            image_file = os.path.join(data_directory, directory, f'zed_image_left_{directory}.jpg')
            image = cv2.imread(image_file)
            image = cv2.resize(image, (0, 0), None, .5, .5)

            point_cloud = np.load(os.path.join(data_directory, directory, f'point_cloud_{directory}.npy'))
            normalized_depth_map = generate_depth_map(point_cloud)
            normalized_depth_map = cv2.resize(normalized_depth_map, (0, 0), None, .5, .5)
            normalized_depth_map = cv2.cvtColor(normalized_depth_map, cv2.COLOR_GRAY2BGR)

            concat_images = np.concatenate((image, normalized_depth_map), axis = 1)

            if show_image:
                cv2.imshow('Image', concat_images)

                key = cv2.waitKey(0)

                if key==27: # Check for ESC key press
                    break
                else:
                    continue

            if save_image:
                file_name = os.path.join(data_directory, directory, f'side_by_side_view_{directory}.jpg')
                cv2.imwrite(file_name, concat_images)

        break

def generate_depth_map(point_cloud):

    depth_estimate = np.zeros((point_cloud.shape[0], point_cloud.shape[1]))
    point_cloud = np.nan_to_num(point_cloud)

    for w in range(point_cloud.shape[0]):
        for h in range(point_cloud.shape[1]):
            value = point_cloud[w][h]

            if not np.isnan(value.any()):
                depth_estimate[w][h] = math.sqrt(value[0] * value[0] + 
                                                 value[1] * value[1] + 
                                                 value[2] * value[2])
            else:
                depth_estimate[w][h] = 0

    normalized_depth = (depth_estimate - np.min(depth_estimate)) / (np.max(depth_estimate) - np.min(depth_estimate))  * 255

    return normalized_depth.astype(np.uint8)

def create_video():
    data_directory = './downloads/moon_yard'

    fourcc = cv2.VideoWriter_fourcc(*'X264')
    video_writer = cv2.VideoWriter(filename='depth_map_husky.mp4', 
                                   fourcc=fourcc, 
                                   fps=5, 
                                   frameSize=(1920, 540))

    for root, dirs, files in os.walk(data_directory):
        for directory in tqdm(dirs):
            image_file = os.path.join(data_directory, directory, f'side_by_side_view_{directory}.jpg')

            if os.path.exists(image_file):
                image = cv2.imread(image_file)

                video_writer.write(image)

        
        break

    video_writer.release()

if __name__ == '__main__':
    create_video()