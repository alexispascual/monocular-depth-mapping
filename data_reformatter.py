import os
import shutil
from pprint import pp
import re
from tqdm import tqdm


def main():
    root_folder = 'D:/PhD/Depth Estimation/Analogue/analogue_july12/data'
    depth_folder = 'D:/PhD/Depth Estimation/Analogue/analogue_july12/zed-depth'
    image_folder = 'D:/PhD/Depth Estimation/Analogue/analogue_july12/zed-left'

    depth_files = []
    image_files = []

    for root, dirs, files in os.walk(depth_folder):
        depth_files = [f for f in files]

    for root, dirs, files in os.walk(image_folder):
        image_files = [f for f in files]

    pp(depth_files)
    pp(image_files)

    for depth, image in tqdm(zip(depth_files, image_files)):
        name = ''.join(re.split(r'(\d+)', depth)[1:-1])
        path = os.path.join(root_folder, name)
        os.makedirs(path)

        shutil.copyfile(os.path.join(image_folder, image), os.path.join(path, f'zed_image_left_{name}.png'))
        shutil.copyfile(os.path.join(depth_folder, depth), os.path.join(path, f'depth_map_{name}.tiff'))
        

if __name__ == '__main__':
    main()
