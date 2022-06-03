import cv2
import numpy as np

from tqdm import tqdm
from utils import tools

image = None
depth_map = np.zeros((1080, 1920))
depth_from_point_cloud = np.zeros((1080, 1920))
depth_loaded = False
depth_position = (10, 10)
depth_estimate_position = (10, 20)


def write_depth(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONUP:

        if depth_loaded:
            depth = depth_map[y, x]
            depth_estimate = depth_from_point_cloud[y, x]

            write_image_text(f"{depth = :.2f}", depth_position)
            write_image_text(f"{depth_estimate = :.2f}", depth_estimate_position)
        
        else:
            write_image_text("Load depth file!", depth_position)
            write_image_text("Press 'r'", depth_estimate_position)

    elif event == cv2.EVENT_LBUTTONDOWN:
        clear_image_text()


def display_depth(_image, depth_map_file, point_cloud_file):
    global image, depth_map, depth_from_point_cloud, depth_loaded

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
            if type(depth_map_file) == str:
                depth_map = np.load(depth_map_file)
                point_cloud = np.load(point_cloud_file)

                tqdm.write("Calculating depth from point cloud...") 
                depth_from_point_cloud = tools.generate_depth_map(point_cloud)

            else:
                depth_map = depth_map_file
                point_cloud = point_cloud_file

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
