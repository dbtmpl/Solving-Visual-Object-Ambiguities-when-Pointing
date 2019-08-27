import cv2 as cv
import numpy as np

import mechanics.gesture_detection as gesture
import mechanics.state_model as state
import mechanics.object_detection as obj_d
from mechanics.gwr_interface import GWRInterface


def run_demo():
    # Bayesian Hist => for detecting the hand
    skin_prob_crcb_link = "resources/skin_color_segmentation/saved_histograms/skin_probabilities_crcb.npy"
    thresh_crcb = 10
    skin_prob_binary_crcb = gesture.get_lab_skin_hist(thresh_crcb, skin_prob_crcb_link)

    # Load video
    cap = cv.VideoCapture('resources/test_videos/amb1_o3_r1_m.webm')

    # !!! Which approach !!!
    # True: Computer Vision approach, False: GWR
    # pointing_estimation = False
    pointing_estimation = False

    # three states are possible: State 0: None, State 1: One, State 2: Two
    tracking_state = "None"
    hand_positions_t0 = [None, None]
    frame_counter = 0

    # take first frame and get shape
    ret, first_frame = cap.read()
    first_frame_shape = first_frame.shape

    # percentages of empirical cropped image
    crops_y = int(first_frame_shape[0] - np.ceil((first_frame_shape[0] * 0.12037)))
    crops_x1 = int(np.ceil((first_frame_shape[1] * 0.3125)))
    crops_x2 = int(first_frame_shape[1] - np.ceil((first_frame_shape[1] * 0.3457)))

    # print(int(first_frame_shape[1] / 2))
    # print(first_frame_shape[0] / 2)

    # initialize gwr for predicting
    link_data_gwr = "results/gwr_based_approach/gwr_models_and_results/crossval_90_50e/0"
    gwr = GWRInterface(link_data_gwr, first_frame.shape)

    # for debugging purposes, saves binary images
    prep_steps = []

    # 2D list: stores the last 5 values of hand features for calculating the mean
    # 1. fingertip, 2. p3, 3. gwr_angle
    mean_aver = [[], [], []]

    # while not pressing esc
    while cv.waitKey(30) & 0xFF != ord('q'):
        ret, frame = cap.read()
        # print(frame_counter)

        if ret:
            # cropping and resizing of 4K Video
            complete_frame = cv.resize(frame, (int(first_frame_shape[1] / 2), int(first_frame_shape[0] / 2)))
            frame = frame[0:crops_y, crops_x1:crops_x2]
            frame = cv.resize(frame, (int((crops_x2 - crops_x1) / 2), int(crops_y / 2)))

            # convert to ycrcb
            frame_ycrcb = cv.cvtColor(frame, cv.COLOR_BGR2YCrCb)

            # detect objects in scene for computer vision approach
            object_bb = []
            if pointing_estimation:
                detected_objects = obj_d.detect_objects(frame, hand_positions_t0)
                for i, d_object in enumerate(detected_objects):
                    if d_object is not None:
                        bb, color_str, color_int = d_object
                        cv.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2, cv.LINE_AA)
                        object_bb.append(
                            ["The " + color_str + " Object", (bb[0], bb[1], bb[2] - bb[0], bb[3] - bb[1]), bb])

            # Get initial skin binary
            skin_binary = gesture.apply_skin_hist2d(frame_ycrcb, skin_prob_binary_crcb)

            # create frames beforehand for debugging
            mask_result = np.zeros(skin_binary.shape, np.uint8)
            blur = np.zeros(skin_binary.shape, np.uint8)
            thresh = np.zeros(skin_binary.shape, np.uint8)
            mask_1 = np.zeros(skin_binary.shape, np.uint8)
            mask_2 = np.zeros(skin_binary.shape, np.uint8)

            if tracking_state == "One" or tracking_state == "Two":

                cnts, skin_binary, thresh, prep_steps = state.search_new_hand_cnts(skin_binary, mask_1, mask_2,
                                                                                   mask_result,
                                                                                   hand_positions_t0, tracking_state)

                if cnts is not None:

                    if len(cnts) == 1:

                        frame, frame2, tracking_state, thresh = state.tracking_one_hand(frame, cnts, thresh,
                                                                                        hand_positions_t0,
                                                                                        object_bb, mean_aver, gwr,
                                                                                        pointing_estimation)

                        frame_counter += 1

                    elif len(cnts) == 2:

                        frame, frame2, tracking_state, thresh = state.tracking_two_hands(frame, cnts, thresh,
                                                                                         hand_positions_t0,
                                                                                         object_bb, mean_aver, gwr,
                                                                                         pointing_estimation)

                        frame_counter += 1

                else:
                    tracking_state = "None"
                    frame_counter += 1

            else:
                # when tracking is lost

                frame, hand_positions_t0, tracking_state = state.detection_step(frame, skin_binary, hand_positions_t0,
                                                                                tracking_state)
                frame_counter += 1

            # for saving frames for visualizations
            k = cv.waitKey(10)
            if k == 0x63 or k == 0x43:
                print
                'capturing!'
                # cv.imwrite("resources/for_thesis/" + str(counter) + "complete_frame.jpg", complete_frame)
                # cv.imwrite("resources/for_thesis/" + str(counter) + "orig_frame.jpg", orig_frame)
                cv.imwrite("resources/for_thesis/" + str(frame_counter) + "frame.jpg", frame)
                # cv.imwrite("resources/for_thesis/" + str(frame_counter) + "frame2.jpg", frame2)
                # cv.imwrite("resources/for_thesis/" + str(frame_counter) + "masked_result.jpg", prep_steps[0])
                # cv.imwrite("resources/for_thesis/" + str(frame_counter) + "blur.jpg", prep_steps[1])
                # cv.imwrite("resources/for_thesis/" + str(frame_counter) + "thresh.jpg", thresh)
                # cv.imwrite("resources/for_thesis/" + str(frame_counter) + "frame_2.jpg", cluster_frame)

            frame_counter += 1

            # Show
            cv.imshow("original", frame)
            cv.imshow("mask results", thresh)
            # cv.imshow("blur", prep_steps[1])
            cv.imshow("original binary", skin_binary)

        else:
            cap.release()
            cv.destroyAllWindows()


if __name__ == '__main__':
    run_demo()
