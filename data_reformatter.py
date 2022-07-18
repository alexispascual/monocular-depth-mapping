import os
from pprint import pp
import re


def main():
    depth_folder = 'D:/PhD/Depth Estimation/Analogue/analogue_july12/zed-depth'
    image_folder = 'D:/PhD/Depth Estimation/Analogue/analogue_july12/zed-left'

    depth_files = []
    image_files = []

    for root, dirs, files in os.walk(depth_folder):
        depth_files = [''.join(re.split(r'(\d+)', f)[1:-1]) for f in files]

    for root, dirs, files in os.walk(image_folder):
        image_files = [''.join(re.split(r'(\d+)', f)[1:-1]) for f in files]

    pp(depth_files)
    pp(image_files)


if __name__ == '__main__':
    main()
