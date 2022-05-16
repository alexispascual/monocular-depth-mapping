import os
import cv2

image = None
depth_map = None
depth_from_point_cloud = None
depth_position = (10, 50)
depth_estimate_position = (10, 70)

def display_depth(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONUP:
        depth = depth_map[x, y]
        depth_estimate = depth_from_point_cloud[x, y]

        cv2.putText(image, 
                    f"Depth = {depth}",
                    depth_position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0))

        cv2.putText(image, 
                    f"Depth Estimate = {depth_estimate}",
                    depth_estimate_position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0))

def view_depth(_image, _depth_map, _depth_from_point_cloud):
    global image, depth_map, depth_from_point_cloud

    image = _image
    depth_map = _depth_map
    depth_from_point_cloud = _depth_from_point_cloud

    cv2.namedWindow('Depth')
    cv2.setMouseCallback('Depth', display_depth)

    while(1):
        cv2.imshow('Depth', image)

