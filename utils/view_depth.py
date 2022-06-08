from typing import Union
import cv2
import numpy as np

from tqdm import tqdm
from utils import tools

image = None
depth_map = np.zeros((1080, 1920))
depth_map_2 = np.zeros((1080, 1920))
depth_loaded = False
depth_position = (10, 10)
depth_estimate_position = (10, 20)


def write_depth(event, x, y, flags, params):

    if event == cv2.EVENT_LBUTTONUP:

        if depth_loaded:
            depth = depth_map[y, x]
            depth_2 = depth_map_2[y, x]

            write_image_text(f"{depth = :.2f}", depth_position)
            write_image_text(f"{depth_2 = :.2f}", depth_estimate_position)
        
        else:
            write_image_text("Load depth file!", depth_position)
            write_image_text("Press 'r'", depth_estimate_position)

    elif event == cv2.EVENT_LBUTTONDOWN:
        clear_image_text()


def display_depth(_image, 
                  _depth_map: Union[str, np.ndarray], 
                  _depth_map_2: Union[str, np.ndarray]):
    global image, depth_map, depth_map_2, depth_loaded

    image = _image
    depth_loaded = False

    cv2.namedWindow('Depth')
    cv2.setMouseCallback('Depth', write_depth)

    while(1):
        cv2.imshow('Depth', image)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            cv2.destroyAllWindows()
            return -1

        elif key == ord('r'):
            if type(_depth_map) == str:
                depth_map = np.load(_depth_map)
                point_cloud = np.load(_depth_map_2)

                tqdm.write("Calculating depth from point cloud...") 
                depth_map_2 = tools.generate_depth_map(point_cloud)

            else:
                depth_map = _depth_map
                depth_map_2 = _depth_map_2

            clear_image_text()
            write_image_text("Done!", depth_position)
            depth_loaded = True

        elif key == ord(' '):
            return 0


def clear_image_text():
    cv2.rectangle(image,
                  (0, 0),
                  (205, 25),
                  (0, 0, 0),
                  -1)


def write_image_text(text: str, position):
    cv2.putText(image, 
                text,
                position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0))
