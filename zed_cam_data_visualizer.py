import os
import cv2
import numpy as np
import glob

from tqdm import tqdm
from PIL import Image

from utils import tools
from utils.create_mask import draw_mask


def main():

    default_config = './config/zed_cam_viewer_config.yaml'

    config = tools.load_config(default_config)

    directories = config['directories']
    flags = config['flags']
    filenames = config['filenames']

    data_directory = directories['data_directory']
    edges_directory = directories['edges_directory']
    horizon_directory = directories['horizon_directory']
    normalized_depth_directory = directories['normalized_depth_directory']
    video_directory = directories['video_directory']

    video_file_name = filenames['video_file_name']

    show_images = flags['show_images']
    save_images = flags['save_images']
    show_depth = flags['show_depth']
    draw_mask_flag = flags['draw_mask_flag'] 
    show_edges = flags['show_edges']
    depth_from_point_cloud = flags['depth_from_point_cloud']
    create_video_flag = flags['create_video_flag']
    ret = 0

    tools.check_directory(edges_directory)
    tools.check_directory(normalized_depth_directory)
    tools.check_directory(horizon_directory)

    horizon_files = tools.scan_horizon_files(horizon_directory)

    if create_video_flag:
        create_video(video_directory, video_file_name)

    for _, dirs, _ in os.walk(data_directory):
        for directory in tqdm(dirs):

            image_file = os.path.join(data_directory, directory, f'zed_image_left_{directory}.jpg')

            if not os.path.isfile(image_file):
                image_file = os.path.join(data_directory, directory, f'zed_image_left_{directory}.png')

            image = cv2.imread(image_file)
            # image = cv2.resize(image, (0, 0), None, .5, .5)
            image_2 = None
            save_dir = data_directory
            save_file_name = f'image_2{directory}.jpg'

            if show_depth:
                normalized_depth_map = None

                if depth_from_point_cloud:
                    tqdm.write("Calculating depth from point cloud...") 
                    point_cloud = np.load(os.path.join(data_directory, directory, f'point_cloud_{directory}.npy'))
                    normalized_depth_map = tools.generate_depth_map(point_cloud, normalize=True)
                    # normalized_depth_map = cv2.resize(normalized_depth_map, (0, 0), None, .5, .5)
                    normalized_depth_map = cv2.cvtColor(normalized_depth_map, cv2.COLOR_GRAY2BGR)

                else:
                    tqdm.write("Loading depth map from file...")
                    depth_file = os.path.join(data_directory, directory, f'depth_map_{directory}.npy')

                    if os.path.isfile(depth_file):
                        depth_map = np.load(depth_file)
                        normalized_depth_map = 255 * (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))
                        normalized_depth_map = cv2.cvtColor(normalized_depth_map, cv2.COLOR_GRAY2BGR).astype(np.uint8)

                    else:
                        depth_file = os.path.join(data_directory, directory, f'depth_map_{directory}.tiff')
                        depth_map = np.array(Image.open(depth_file))
                        depth_map = np.nan_to_num(depth_map, posinf=20.0)
                        normalized_depth_map = 255 * (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))
                        normalized_depth_map = cv2.cvtColor(normalized_depth_map, cv2.COLOR_GRAY2BGR).astype(np.uint8)

                image_2 = normalized_depth_map
                save_dir = normalized_depth_directory
                save_file_name = f'normalized_depth_{directory}.jpg'

            if show_edges:

                edges = detect_edge(image)

                if draw_mask_flag:

                    ret = tools.show_image(image, edges)

                    if directory not in horizon_files:
                        draw_mask(edges, 
                                  horizon_directory, 
                                  directory)

                image_2 = edges
                save_dir = edges_directory
                save_file_name = f'edges_{directory}.jpg'

            if show_images:
                ret = tools.show_image(image, image_2)

            if save_images:
                tools.save_image(image,
                                 image_2, 
                                 save_dir, 
                                 save_file_name)

            if ret == -1:
                break

        break


def create_video(data_directory: str, filename: str):

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(filename=filename, 
                                   fourcc=fourcc, 
                                   fps=5, 
                                   frameSize=(1920, 540))
    file_paths = glob.glob(os.path.join(data_directory, "*.jpg"))
    
    for file in tqdm(file_paths):
        image = cv2.imread(file)
        if image is not None:
            video_writer.write(image)

    video_writer.release()
    # for _, _, files in os.walk(data_directory):
    #     for file in tqdm(files):
    #         image_file = os.path.join(data_directory, f'side_by_side_view_{directory}.jpg')

    #         if os.path.exists(image_file):
    #             image = cv2.imread(image_file)

    #             video_writer.write(image)
    #     break

    # video_writer.release()


def detect_edge(image):
    image = cv2.GaussianBlur(image, (5, 5), 0)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image = cv2.Canny(image, 100, 200)

    kernel = np.ones((5, 5), np.uint8)
    image = cv2.dilate(image, kernel)

    kernel = np.ones((3, 3), np.uint8)
    image = cv2.erode(image, kernel)

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image
    

if __name__ == '__main__':
    main()
