import os
import cv2
import numpy as np
import math

from pprint import pprint as pp
from tqdm import tqdm
from utils.create_mask import draw_mask
from utils.tools import check_directory, save_image, show_image

def main():
    data_directory = './data/moon_yard'
    edges_directory = './data/edges'
    horizon_directory = './data/horizons'
    normalized_depth_directory = './data/normalized_depth'

    show_images = True
    save_images = False
    draw_mask_flag = True 
    detect_edges = True
    generate_depth_maps = False
    ret = 0

    check_directory(edges_directory)
    check_directory(normalized_depth_directory)
    check_directory(horizon_directory)

    for root, dirs, files in os.walk(data_directory):
        for directory in tqdm(dirs):
            tqdm.write("Reading image...")
            image_file = os.path.join(data_directory, directory, f'zed_image_left_{directory}.jpg')
            image = cv2.imread(image_file)
            image = cv2.resize(image, (0, 0), None, .5, .5)

            if generate_depth_maps:
                tqdm.write("Calculating depth from point cloud...") 
                point_cloud = np.load(os.path.join(data_directory, directory, f'point_cloud_{directory}.npy'))
                normalized_depth_map = generate_depth_map(point_cloud)
                normalized_depth_map = cv2.resize(normalized_depth_map, (0, 0), None, .5, .5)
                normalized_depth_map = cv2.cvtColor(normalized_depth_map, cv2.COLOR_GRAY2BGR)

                if show_images:
                    ret = show_image(image, normalized_depth_map)

                if save_images:
                    save_image(image,
                               normalized_depth_map, 
                               normalized_depth_directory, 
                               f'normalized_depth_{directory}.jpg')

            if detect_edges:
                tqdm.write("Detecting edges...")

                edges = detect_edge(image)
                
                if show_images:
                    ret = show_image(image, edges)

                    if draw_mask_flag:
                        draw_mask(edges)
                        save_image(edges, 
                                   directory=horizon_directory, 
                                   file_name=f'horizon_{directory}.jpg')

                if save_images:
                    save_image(image,
                               edges, 
                               edges_directory, 
                               f'edges_{directory}.jpg')

            if ret == -1:
                break

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
    data_directory = './data/moon_yard'

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

def detect_edge(image):
    image = cv2.GaussianBlur(image,(5,5), 0)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image = cv2.Canny(image, 100, 200)

    kernel = np.ones((5,5), np.uint8)
    image  = cv2.dilate(image, kernel)

    kernel = np.ones((3,3), np.uint8)
    image  = cv2.erode(image, kernel)

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image
    
if __name__ == '__main__':
    main()