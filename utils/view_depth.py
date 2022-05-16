import os
import cv2
import numpy as np

image = None
depth_map = None
depth_from_point_cloud = None
depth_position = (10, 10)
depth_estimate_position = (10, 20)

def write_depth(event, x, y, flags, param):

    depth = depth_map[y, x]
    depth_estimate = depth_from_point_cloud[y, x]

    if event == cv2.EVENT_LBUTTONUP:

        cv2.putText(image, 
                    f"{depth = :.2f}",
                    depth_position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0))

        cv2.putText(image, 
                    f"{depth_estimate = :.2f}",
                    depth_estimate_position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0))

    elif event == cv2.EVENT_LBUTTONDOWN:
       
        cv2.rectangle(image,
                      (0,0),
                      (205, 25),
                      (0, 0, 0),
                      -1)

def display_depth(_image, _depth_map, _depth_from_point_cloud):
    global image, depth_map, depth_from_point_cloud

    image = _image
    depth_map = _depth_map
    depth_from_point_cloud = _depth_from_point_cloud

    cv2.namedWindow('Depth')
    cv2.setMouseCallback('Depth', write_depth)

    while(1):
        cv2.imshow('Depth', image)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            cv2.destroyAllWindows()
            return -1



