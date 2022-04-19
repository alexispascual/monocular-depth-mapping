import numpy as np
import cv2

drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1
image = None

def draw_pixels(event,x,y,flags,param):
    global ix,iy,drawing,mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv2.rectangle(image,(ix,iy),(x,y),(0,0,0),-1)
            else:
                cv2.circle(image,(x,y),5,(0,0,0),-1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv2.rectangle(image,(ix,iy),(x,y),(0,0,0),-1)
        else:
            cv2.circle(image,(x,y),5,(0,0,0),-1)

def draw_mask(edges):
	global image, mode
	image = edges

	cv2.namedWindow('Edges')
	cv2.setMouseCallback('Edges',draw_pixels)

	while(1):
	    cv2.imshow('Edges',image)
	    key = cv2.waitKey(1) & 0xFF
	    if key == ord('m'):
	        mode = not mode
	    elif key == 27:
	        break