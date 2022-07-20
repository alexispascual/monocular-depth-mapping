import os
import shutil
import re

from tqdm import tqdm

from utils.tools import check_directory


def format_campaign_1():
    root_folder = 'D:/PhD/Depth Estimation/Data/campaign_1/data'
    dest_folder = 'D:/PhD/Depth Estimation/Data/campaign_1/images'

    check_directory(dest_folder)

    for root, dirs, files in os.walk(root_folder):
        for directory in dirs:
            image_file = os.path.join(root_folder, directory, f'zed_image_left_{directory}.jpg')

            shutil.copyfile(image_file, os.path.join(dest_folder, f'zed_image_left_{directory}.jpg'))


def format_campaign_2():
    root_folder = 'D:/PhD/Depth Estimation/Analogue/analogue_july12/data'
    depth_folder = 'D:/PhD/Depth Estimation/Analogue/analogue_july12/zed-depth'
    image_folder = 'D:/PhD/Depth Estimation/Analogue/analogue_july12/zed-left'

    depth_files = []
    image_files = []

    for root, dirs, files in os.walk(depth_folder):
        depth_files = [f for f in files]

    for root, dirs, files in os.walk(image_folder):
        image_files = [f for f in files]

    for depth, image in tqdm(zip(depth_files, image_files)):
        name = ''.join(re.split(r'(\d+)', depth)[1:-1])
        path = os.path.join(root_folder, name)
        os.makedirs(path)

        shutil.copyfile(os.path.join(image_folder, image), os.path.join(path, f'zed_image_left_{name}.png'))
        shutil.copyfile(os.path.join(depth_folder, depth), os.path.join(path, f'depth_map_{name}.tiff'))
        

if __name__ == '__main__':
    format_campaign_1()
    print("========= Done! =========")
