import cv2 as cv
import numpy as np


def get_crop_heuristics(frame):
    """
    Crop frame after the heuristics described in the paper (see Figure 3)
    :param frame: NumPy array of shape (H, W, 3)
    :return: Tuple: Value for cropping the image
    """
    frame_shape = frame.shape
    crops_y = int(frame_shape[0] - np.ceil((frame_shape[0] * 0.12037)))
    crops_x1 = int(np.ceil((frame_shape[1] * 0.3125)))
    crops_x2 = int(frame_shape[1] - np.ceil((frame_shape[1] * 0.3457)))
    return crops_y, crops_x1, crops_x2


def crop_and_resize(frame, crops):
    frame = frame[0:crops[0], crops[1]:crops[2]]
    return cv.resize(frame, (int((crops[2] - crops[1]) / 2), int(crops[0] / 2)))
