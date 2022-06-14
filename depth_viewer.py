import os
import cv2

from tqdm import tqdm
from utils import tools, view_depth


def main():

    config_file = './config/depth_viewer_config.yaml'

    config = tools.load_config(config_file)

    directories = config['directories']

    data_directory = directories['data_directory']
    # horizon_directory = directories['horizon_directory']
    
    ret = 0

    for _, dirs, _ in os.walk(data_directory):
        for directory in tqdm(dirs):
            tqdm.write("Reading data...")

            image_file = os.path.join(data_directory, directory, f'zed_image_left_{directory}.jpg')
            depth_file = os.path.join(data_directory, directory, f'depth_map_{directory}.npy')
            point_cloud_file = os.path.join(data_directory, directory, f'point_cloud_{directory}.npy')

            image = cv2.imread(image_file)

            ret = view_depth.display_depth(image, depth_file, point_cloud_file)

            if ret == -1:
                break
        break


if __name__ == '__main__':
    main()
