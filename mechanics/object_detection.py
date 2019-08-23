import cv2 as cv
import numpy as np
import gesture_detection as gesture


def find_yellow_object(frame, hand_positions):
    range_yellow = [(17, 130, 80), (37, 255, 255)]
    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    frame_threshold_yellow = cv.inRange(frame_hsv, range_yellow[0], range_yellow[1])

    for hand in hand_positions:
        if hand is not None:
            x1, y1, x2, y2 = hand[0], hand[1], hand[0] + hand[2], hand[1] + hand[3]
            frame_threshold_yellow[y1:y2, x1:x2] = 0

    kernel2 = np.ones((3, 3), np.uint8)
    dilation_yellow = cv.dilate(frame_threshold_yellow, kernel2, iterations=1)

    cnt = gesture.get_biggest_contours(dilation_yellow, 10)
    if cnt is not None:
        x, y, w, h = cv.boundingRect(cnt[0])
        return (x, y, x + w, y + h), "yellow", 1

    return None


def find_green_object(frame, hand_positions):
    range_green = [(40, 78, 26), (90, 255, 255)]
    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    frame_threshold_green = cv.inRange(frame_hsv, range_green[0], range_green[1])

    for hand in hand_positions:
        if hand is not None:
            x1, y1, x2, y2 = hand[0], hand[1], hand[0] + hand[2], hand[1] + hand[3]
            frame_threshold_green[y1:y2, x1:x2] = 0

    kernel1 = np.ones((3, 3), np.uint8)
    dilation_green = cv.dilate(frame_threshold_green, kernel1, iterations=1)

    cnt = gesture.get_biggest_contours(dilation_green, 10)
    if cnt is not None:
        x, y, w, h = cv.boundingRect(cnt[0])
        return (x, y, x + w, y + h), "green", 2

    return None


def find_red_object(frame, hand_positions):
    range_red_1 = [(0, 145, 175), (10, 255, 255)]
    range_red_2 = [(130, 50, 10), (180, 255, 255)]
    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    frame_threshold_red1 = cv.inRange(frame_hsv, range_red_1[0], range_red_1[1])
    frame_threshold_red2 = cv.inRange(frame_hsv, range_red_2[0], range_red_2[1])
    frame_threshold_red = cv.bitwise_or(frame_threshold_red1, frame_threshold_red2)

    for hand in hand_positions:
        if hand is not None:
            x1, y1, x2, y2 = hand[0], hand[1], hand[0] + hand[2], hand[1] + hand[3]
            frame_threshold_red[y1:y2, x1:x2] = 0

    kernel1 = np.ones((3, 3), np.uint8)
    kernel2 = np.ones((5, 5), np.uint8)
    erosion = cv.erode(frame_threshold_red, kernel1, iterations=1)
    dilation_red = cv.dilate(erosion, kernel2, iterations=4)

    cnt = gesture.get_biggest_contours(dilation_red, 10)
    if cnt is not None:
        x, y, w, h = cv.boundingRect(cnt[0])
        return (x, y, x + w, y + h), "red", 0

    return None


def detect_objects(frame, hand_positions_t0):
    bbs = [None, None, None]

    bb_red = find_red_object(frame, hand_positions_t0)
    if bb_red:
        bbs[0] = bb_red

    bb_yellow = find_yellow_object(frame, hand_positions_t0)
    if bb_yellow:
        bbs[1] = bb_yellow

    bb_green = find_green_object(frame, hand_positions_t0)
    if bb_green:
        bbs[2] = bb_green

    return bbs


def check_for_ambiguity(frame, union_of_gwr_nodes, hand_positions_t0):
    x1, y1, x2, y2 = union_of_gwr_nodes
    shape = frame.shape
    black_frame = np.zeros(shape, np.uint8)
    black_frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2]

    detected_objects = detect_objects(black_frame, hand_positions_t0)
    objecs_not_none = [d_object for d_object in detected_objects if d_object is not None]

    if len(objecs_not_none) >= 2:
        return True, objecs_not_none, black_frame
    else:
        return False, objecs_not_none, black_frame
