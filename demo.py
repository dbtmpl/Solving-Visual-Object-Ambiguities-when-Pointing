import argparse
import cv2 as cv
import numpy as np

import engine.gesture_detection as gesture
import engine.state_model as state
from engine.gwr_interface import GWRInterface
from utils.general_utils import get_crop_heuristics, crop_and_resize
from utils.detection_utils import detect_objects_in_frame


class GestureDemo:

    def __init__(self, ARGS):
        self.skin_model = gesture.get_skin_histogram(path=ARGS.skin_model)

        # Load video
        self.cap = cv.VideoCapture(ARGS.demo_video)

        # If true uses the pointing array technique
        self.pointing_estimation = ARGS.use_pointing_array

        # three states are possible: State 0: None, State 1: One, State 2: Two
        self.tracking_state = 'None'
        self.hand_positions_t0 = [None, None]

        # take first frame and get shape
        _, frame = self.cap.read()
        self._crops = get_crop_heuristics(frame)
        frame = crop_and_resize(frame, self._crops)

        # initialize gwr for predicting
        self.gwr = GWRInterface(ARGS.gwr_model, ARGS.skin_model, frame.shape)

        # Current bounding boxes in the scene
        self.object_bb = []

    def process_frame(self, frame):
        # cropping and resizing current frame
        frame_cropped = crop_and_resize(frame, self._crops)
        frame_ycrcb = cv.cvtColor(frame_cropped, cv.COLOR_BGR2YCrCb)
        # Get initial skin binary
        skin_binary = gesture.apply_skin_hist2d(frame_ycrcb, self.skin_model)

        return skin_binary, frame_cropped

    def run_iteration(self, frame):
        skin_binary, frame = self.process_frame(frame)
        if self.pointing_estimation:
            self.object_bb = detect_objects_in_frame(frame, self.hand_positions_t0)

        if self.tracking_state in ["One", "Two"]:

            cnts, thresh = state.search_new_hand_cnts(skin_binary, self.hand_positions_t0, self.tracking_state)

            if cnts is not None:
                if len(cnts) == 1:
                    frame, frame2, self.tracking_state, thresh = state.tracking_one_hand(
                        frame, cnts, thresh, self.hand_positions_t0, self.object_bb, self.gwr,
                        self.pointing_estimation
                    )

                elif len(cnts) == 2:
                    frame, frame2, self.tracking_state, thresh = state.tracking_two_hands(
                        frame, cnts, thresh, self.hand_positions_t0, self.object_bb, self.gwr,
                        self.pointing_estimation
                    )

            else:
                self.tracking_state = "None"

        else:
            # when tracking is lost
            frame, self.hand_positions_t0, self.tracking_state = state.detection_step(
                frame, skin_binary, self.hand_positions_t0, self.tracking_state
            )

        return frame

    def run_demo(self):
        frame_counter = 0
        # while not pressing esc
        while cv.waitKey(30) & 0xFF != ord('q'):
            ret, frame = self.cap.read()

            if not ret:
                self.cap.release()
                cv.destroyAllWindows()
                return False

            frame = self.run_iteration(frame)

            # Saving frames for visualizations
            k = cv.waitKey(10)
            if k == ord('p'):
                cv.imwrite(f"frame_{frame_counter}.jpg", frame)

            # Show
            cv.imshow("HRI Scene", frame)
            frame_counter += 1


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
    gesture_demo = GestureDemo(ARGS)
    gesture_demo.run_demo()
