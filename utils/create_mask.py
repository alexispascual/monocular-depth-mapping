import cv2

from utils.tools import save_image, fill_horizon_line, fill_horizon_line_top_down

drawing = False  # true if mouse is pressed
mode = True  # if True, draw rectangle. Press 'm' to toggle to curve
colour_flag = True
colour = (0, 0, 0)
ix, iy = -1, -1
image = None


def draw_pixels(event, x, y, flags, param):

    global ix, iy, drawing, mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            if mode:
                cv2.rectangle(image, (ix, iy), (x, y), colour, -1)
            else:
                cv2.circle(image, (x, y), 2, colour, -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode:
            cv2.rectangle(image, (ix, iy), (x, y), colour, -1)
        else:
            cv2.circle(image, (x, y), 2, colour, -1)


def draw_mask(edges, horizon_directory: str, directory: str):

    global image, mode, colour, colour_flag
    image = edges

    cv2.namedWindow('Edges')
    cv2.setMouseCallback('Edges', draw_pixels)

    while(1):
        cv2.imshow('Edges', image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('a'):
            mode = not mode
        elif key == ord('d'):
            if colour_flag:
                colour = (255, 255, 255)
            else:
                colour = (0, 0, 0)
        
            colour_flag = not colour_flag

        elif key == ord('f'):
            image = fill_horizon_line(image)

        elif key == ord('g'):
            image = fill_horizon_line_top_down(image)

        elif key == ord('q') or key == 27:
            break

        elif key == ord('s'):
            save_image(image, 
                       directory=horizon_directory, 
                       file_name=f'horizon_{directory}.jpg')
