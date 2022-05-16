import os
import cv2

import numpy as np

from tqdm import tqdm
from utils import tools, view_depth

def main():

    config_file = './config/depth_viewer_config.yaml'

    config = tools.load_config(config_file)

    directories = config['directories']

    data_directory = directories['data_directory']
    horizon_directory = directories['horizon_directory']
    
    ret = 0

    for root, dirs, files in os.walk(data_directory):
        for directory in tqdm(dirs):
            tqdm.write("Reading data...")

            image_file = os.path.join(data_directory, directory, f'zed_image_left_{directory}.jpg')
            image = cv2.imread(image_file)

            depth_file = os.path.join(data_directory, directory, f'depth_map_{directory}.npy')
            depth_map = np.load(depth_file)

            point_cloud_file = os.path.join(data_directory, directory, f'point_cloud_{directory}.npy')
            point_cloud = np.load(point_cloud_file)
            tqdm.write("Calculating depth from point cloud...")
            depth_estimate = tools.generate_depth_map(point_cloud)

            ret = view_depth.display_depth(image, depth_map, depth_estimate)

            if ret == -1:
                break

        break
if __name__ == '__main__':
    main()