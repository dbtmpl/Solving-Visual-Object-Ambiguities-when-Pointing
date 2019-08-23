import cv2 as cv
import numpy as np
from mechanics import gesture_detection as gesture


def detection_step(frame, skin_binary, hand_positions_t0, tracking_state):
    opening = cv.morphologyEx(skin_binary, cv.MORPH_OPEN, np.ones((2, 2), np.uint8))

    # Detect one or two hands
    # One
    cnts = gesture.get_biggest_contours(opening, 40)
    # Two
    # cnts = gesture.get_biggest_two_contours(opening, 40)

    if cnts is not None:

        if len(cnts) == 1:
            x1, y1, w1, h1 = cv.boundingRect(cnts[0])
            # cv.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (50, 50, 255), 2)
            hand_positions_t0[0] = (x1, y1, w1, h1)
            hand_positions_t0[1] = None
            tracking_state = "One"

        elif len(cnts) == 2:
            # cv.drawContours(frame, [cnt], 0, (255, 50, 50), 2)
            cnt_1, cnt_2 = cnts
            x1, y1, w1, h1 = cv.boundingRect(cnt_1)
            x2, y2, w2, h2 = cv.boundingRect(cnt_2)
            # cv.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (50, 50, 255), 2)
            # cv.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (50, 50, 255), 2)
            hand_positions_t0[0] = (x1, y1, w1, h1)
            hand_positions_t0[1] = (x2, y2, w2, h2)
            tracking_state = "Two"

        return frame, hand_positions_t0, tracking_state
    else:
        hand_positions_t0 = [None, None]
        tracking_state = "None"
        return frame, hand_positions_t0, tracking_state


def search_new_hand_cnts(skin_binary, mask_1, mask_2, mask_result, hand_positions_t0, tracking_state):
    if tracking_state == "One":
        x_0, y_0, w_0, h_0 = hand_positions_t0[0]

        padding = 30
        mask_1 = gesture.calc_tracking_roi(skin_binary, mask_1, padding, (x_0, y_0, w_0, h_0))
        skin_binary = gesture.calc_tracking_roi(mask_2, skin_binary, padding, (x_0, y_0, w_0, h_0))

        mask_result = cv.bitwise_or(mask_1, skin_binary)

    elif tracking_state == "Two":

        x_0, y_0, w_0, h_0 = hand_positions_t0[0]
        x_1, y_1, w_1, h_1 = hand_positions_t0[1]

        # create search window bigger than bounding box around the hand
        padding = 30

        mask_1 = gesture.calc_tracking_roi(skin_binary, mask_1, padding, (x_0, y_0, w_0, h_0))
        mask_2 = gesture.calc_tracking_roi(skin_binary, mask_2, padding, (x_1, y_1, w_1, h_1))

        mask_result = cv.bitwise_or(mask_1, mask_2)

    blur = cv.GaussianBlur(mask_result, (11, 11), 0)
    ret, thresh = cv.threshold(blur, 100, 255, cv.THRESH_BINARY)

    # Detect one or two hands
    # One
    cnts = gesture.get_biggest_contours(thresh, 40)
    # Two
    # cnts = gesture.get_biggest_two_contours(thresh, 40)

    return cnts, skin_binary, thresh, [mask_result, blur]


def tracking_one_hand(frame, cnts, thresh, hand_positions_t0, object_bb, mean_aver, gwr, pointing_estimation):
    hand = cnts[0]

    frame2 = frame.copy()

    # get new Bounding box
    x, y, w, h = cv.boundingRect(hand)
    # update tracking
    cv.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
    hand_positions_t0[0] = (x, y, w, h)
    hand_positions_t0[1] = None
    tracking_state = "One"

    # Calculate Centroid of hand
    icx, icy = gesture.get_contour_centroid(hand)
    # cv.circle(frame, (icx, icy), 2, [255, 255, 255], -1)

    fingertips, p1_p2_points, defects, hull = gesture.get_fingertips(hand, icy)
    hand_data = [[fingertips, p1_p2_points, (icx, icy), defects, hand]]

    # gesture.draw_convex_defects(frame, hand_data)

    if fingertips.shape[0] == 1:
        p1_p2_points = p1_p2_points[0]
        p1 = p1_p2_points[0]
        p2 = p1_p2_points[1]
        p3 = p1_p2_points[2]
        fingertip = fingertips[0]

        frame, defect_count = gesture.check_defects_for_pointing(frame, hand, defects, fingertips[0], (icx, icy))

        if defect_count <= 2:
            if pointing_estimation:
                frame = gesture.calc_and_clip_pointing_array(frame, p3, fingertip, object_bb)
            else:
                frame, frame2 = gesture.predict_gwr_pointing(frame, frame2, hand, fingertip, (icx, icy), gwr, hand_positions_t0)

    return frame, frame2, tracking_state, thresh


def tracking_two_hands(frame, cnts, thresh, hand_positions_t0, object_bb, mean_aver, gwr, pointing_estimation):
    hand_1 = cnts[0]
    hand_2 = cnts[1]

    frame2 = frame.copy()

    # Calculate new Bounding boxes
    x1, y1, w1, h1 = cv.boundingRect(hand_1)
    x2, y2, w2, h2 = cv.boundingRect(hand_2)
    cv.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (50, 50, 255), 2)
    cv.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (50, 50, 255), 2)
    hand_positions_t0[0] = (x1, y1, w1, h1)
    hand_positions_t0[1] = (x2, y2, w2, h2)
    tracking_state = "Two"

    # Calculate Centroid of hands
    icx_1, icy_1 = gesture.get_contour_centroid(hand_1)
    icx_2, icy_2 = gesture.get_contour_centroid(hand_2)
    # cv.circle(frame, (icx_1, icy_1), 2, [255, 255, 255], -1)
    # cv.circle(frame, (icx_2, icy_2), 2, [255, 255, 255], -1)

    fingertips_1, p1_p2_points_1, defects_1, hull_1 = gesture.get_fingertips(hand_1, icy_1)
    fingertips_2, p1_p2_points_2, defects_2, hull_2 = gesture.get_fingertips(hand_2, icy_2)

    hand_data = [[fingertips_1, p1_p2_points_1, (icx_1, icy_1), defects_1, hand_1],
                 [fingertips_2, p1_p2_points_2, (icx_2, icy_2), defects_2, hand_2]]

    for hand_d in hand_data:
        if hand_d[0].shape[0] == 1:

            p1_p2_points = hand_d[1][0]
            p1 = p1_p2_points[0]
            p2 = p1_p2_points[1]
            p3 = p1_p2_points[2]
            fingertip = hand_d[0][0]

            frame, defect_count = gesture.check_defects_for_pointing(frame, hand_d[4], hand_d[3], fingertip, hand_d[2])

            if defect_count <= 2:
                if pointing_estimation:
                    frame = gesture.calc_and_clip_pointing_array(frame, p3, fingertip, object_bb)
                else:
                    frame, frame2 = gesture.predict_gwr_pointing(frame, frame2, hand_d[4], fingertip, hand_d[2], gwr, hand_positions_t0)

    return frame, frame2, tracking_state, thresh
