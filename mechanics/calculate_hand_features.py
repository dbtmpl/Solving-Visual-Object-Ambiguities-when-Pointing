import gesture_detection as gesture
import cv2 as cv


# Get hand features for GWR feature vector
class GWR_Hand_Features(object):

    def __init__(self):

        skin_prob_crcb_link = "resources/skin_color_data/histograms/skin_probabilities_crcb.npy"
        thresh_crcb = 20
        self.skin_prob_binary_crcb = gesture.get_lab_skin_hist(thresh_crcb, skin_prob_crcb_link)

    def handle_angle_list(self, angle, list):
        if len(list) < 5:
            list.append(angle)
            angle = sum(list) / len(list)
            return angle, list
        else:
            _ = list.pop(0)
            list.append(angle)
            angle = sum(list) / len(list)
            return angle, list

    def get_cs_pointing_data(self, frame):
        frame_ycrcb = cv.cvtColor(frame, cv.COLOR_BGR2YCrCb)
        skin_binary = gesture.apply_skin_hist2d(frame_ycrcb, self.skin_prob_binary_crcb)

        blur = cv.GaussianBlur(skin_binary, (11, 11), 0)
        ret, thresh = cv.threshold(blur, 50, 255, cv.THRESH_BINARY)

        # Detect one or two hands
        # One
        cnts = gesture.get_biggest_contours(thresh, 40)
        # Two
        # cnts = gesture.get_biggest_two_contours(thresh, 40)
        if cnts:
            hand = cnts[0]

            icx, icy = gesture.get_contour_centroid(hand)

            fingertips, p1_p2_points, defects, hull = gesture.get_fingertips(hand, icy)

            if fingertips.shape[0] == 1:
                fingertip = fingertips[0]
                p1_p2_points = p1_p2_points[0]
                p3 = p1_p2_points[2]

                return fingertip, p3

            else:
                return None, None
        return None, None

    def get_gwr_data(self, frame, last_five_angles):
        frame_ycrcb = cv.cvtColor(frame, cv.COLOR_BGR2YCrCb)
        skin_binary = gesture.apply_skin_hist2d(frame_ycrcb, self.skin_prob_binary_crcb)

        blur = cv.GaussianBlur(skin_binary, (11, 11), 0)
        ret, thresh = cv.threshold(blur, 50, 255, cv.THRESH_BINARY)

        # Detect one or two hands
        # One
        cnts = gesture.get_biggest_contours(thresh, 40)
        # Two
        # cnts = gesture.get_biggest_two_contours(thresh, 40)
        if cnts:
            hand = cnts[0]

            icx, icy = gesture.get_contour_centroid(hand)

            fingertips, p1_p2_points, defects, hull = gesture.get_fingertips(hand, icy)

            if fingertips.shape[0] == 1:
                fingertip = fingertips[0]

                fitted_line = gesture.calculate_contour_line(frame, hand)

                angle = gesture.get_hand_features_for_GWR(frame, fingertip, fitted_line)

                mean_angle, last_five_angles = self.handle_angle_list(angle, last_five_angles)

                return mean_angle, icx, icy, f, last_five_angles

            return None, None, None, None, last_five_angles
        return None, None, None, None, last_five_angles
