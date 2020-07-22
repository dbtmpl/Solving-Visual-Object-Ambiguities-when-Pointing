import argparse
import cv2 as cv
import numpy as np

import engine.gesture_detection as gesture
import engine.state_model as state
from engine.gwr_interface import GWRInterface
from utils.general_utils import get_crop_heuristics, crop_and_resize
from utils.detection_utils import detect_objects_in_frame


def run_demo(ARGS):
    skin_prob_binary_crcb = gesture.get_skin_histogram(path=ARGS.skin_model)

    # Load video
    cap = cv.VideoCapture(ARGS.demo_video)

    pointing_estimation = ARGS.use_pointing_array

    # three states are possible: State 0: None, State 1: One, State 2: Two
    tracking_state, hand_positions_t0, frame_counter = "None", [None, None], 0

    # take first frame and get shape
    ret, first_frame = cap.read()
    first_frame_shape = first_frame.shape

    # Get crop parameters for frame crop
    crops_y, crops_x1, crops_x2 = get_crop_heuristics(first_frame_shape)
    first_frame = crop_and_resize(first_frame, crops_y, crops_x1, crops_x2)

    # initialize gwr for predicting
    gwr = GWRInterface(ARGS.gwr_model, ARGS.skin_model, first_frame.shape)

    # 2D list: stores the last 5 values of hand features for calculating the mean
    # 1. fingertip, 2. p3, 3. gwr_angle
    mean_aver = [[], [], []]

    object_bb = []

    # while not pressing esc
    while cv.waitKey(30) & 0xFF != ord('q'):
        ret, frame = cap.read()
        if ret:
            # cropping and resizing of 4K Video
            frame = crop_and_resize(frame, crops_y, crops_x1, crops_x2)

            # convert to ycrcb
            frame_ycrcb = cv.cvtColor(frame, cv.COLOR_BGR2YCrCb)

            if pointing_estimation:
                object_bb = detect_objects_in_frame(frame, hand_positions_t0)

            # Get initial skin binary
            skin_binary = gesture.apply_skin_hist2d(frame_ycrcb, skin_prob_binary_crcb)

            # create frames beforehand for debugging
            mask_result = np.zeros(skin_binary.shape, np.uint8)
            mask_1 = np.zeros(skin_binary.shape, np.uint8)
            mask_2 = np.zeros(skin_binary.shape, np.uint8)

            if tracking_state == "One" or tracking_state == "Two":

                cnts, skin_binary, thresh, prep_steps = state.search_new_hand_cnts(
                    skin_binary, mask_1, mask_2, mask_result, hand_positions_t0, tracking_state
                )

                if cnts is not None:

                    if len(cnts) == 1:
                        frame, frame2, tracking_state, thresh = state.tracking_one_hand(
                            frame, cnts, thresh, hand_positions_t0, object_bb, mean_aver, gwr, pointing_estimation
                        )
                        frame_counter += 1

                    elif len(cnts) == 2:
                        frame, frame2, tracking_state, thresh = state.tracking_two_hands(
                            frame, cnts, thresh, hand_positions_t0, object_bb, mean_aver, gwr, pointing_estimation
                        )
                        frame_counter += 1

                else:
                    tracking_state = "None"
                    frame_counter += 1

            else:
                # when tracking is lost
                frame, hand_positions_t0, tracking_state = state.detection_step(
                    frame, skin_binary, hand_positions_t0, tracking_state
                )
                frame_counter += 1

            # for saving frames for visualizations
            k = cv.waitKey(10)
            if k == ord('p'):
                cv.imwrite(f"frame_{frame_counter}.jpg", frame)
            frame_counter += 1

            # Show
            cv.imshow("original", frame)

        else:
            cap.release()
            cv.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="General setup")

    # General
    parser.add_argument(
        '--gwr-model',
        type=str,
        default="resources/gwr_models/model_demo",
        help="Path to the GWR model (Not used when using pointing array for prediction)"
    )
    parser.add_argument(
        '--skin-model',
        type=str,
        default="resources/skin_color_segmentation/saved_histograms/skin_probabilities_crcb.npy",
        help="Path to the skin-color model used for hand detection."
    )
    parser.add_argument(
        '--demo-video',
        type=str,
        default="resources/test_videos/amb1_o3_r1_m.webm",
        help="Path to the demo video."
    )
    parser.add_argument(
        '--use-pointing-array',
        action='store_true',
        help="If set, the pointing array approach is used. By default, the GWR network is used."
    )

    ARGS = parser.parse_args()
    run_demo(ARGS)
