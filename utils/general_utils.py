import cv2 as cv
import numpy as np


def get_crop_heuristics(first_frame_shape):
    # percentages to crop the input image. See Figure 3 in paper.
    crops_y = int(first_frame_shape[0] - np.ceil((first_frame_shape[0] * 0.12037)))
    crops_x1 = int(np.ceil((first_frame_shape[1] * 0.3125)))
    crops_x2 = int(first_frame_shape[1] - np.ceil((first_frame_shape[1] * 0.3457)))

    return crops_y, crops_x1, crops_x2


def crop_and_resize(frame, crops_y, crops_x1, crops_x2):
    frame = frame[0:crops_y, crops_x1:crops_x2]
    return cv.resize(frame, (int((crops_x2 - crops_x1) / 2), int(crops_y / 2)))