import cv2 as cv

from engine import object_detection as obj_d


def detect_objects_in_frame(frame, hand_positions):
    object_bb = []
    detected_objects = obj_d.detect_objects(frame, hand_positions)
    for i, d_object in enumerate(detected_objects):
        if d_object is not None:
            bb, color_str, color_int = d_object
            cv.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2, cv.LINE_AA)
            object_bb.append(
                [color_str, (bb[0], bb[1], bb[2] - bb[0], bb[3] - bb[1]), bb])

    return object_bb