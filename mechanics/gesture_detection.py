import cv2 as cv
import numpy as np
import scipy.cluster.hierarchy as hcluster
import object_detection as obj_d


def apply_skin_hist_3d(frame, skin_prob):
    x, y, z = cv.split(frame)
    B = skin_prob[x.ravel(), y.ravel(), z.ravel()]
    skin_area_orig = B.reshape(frame.shape[:2])
    return np.uint8(skin_area_orig)


def apply_skin_hist2d(frame, skin_prob):
    x, y, z = cv.split(frame)
    B = skin_prob[y.ravel(), z.ravel()]
    skin_area_orig = B.reshape(frame.shape[:2])
    return np.uint8(skin_area_orig)


def get_lab_skin_hist(thresh, link):
    skin_prob = np.load(link)
    print(np.min(skin_prob))
    print(np.max(skin_prob))
    print(skin_prob.shape)
    return np.where(skin_prob > thresh, 255, 0)


# used for detecting the objects as well
def get_biggest_contours(img, noise_thresh):
    im, contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_KCOS)

    # filter noise
    contours = filter(lambda cnt_x: len(cnt_x) > noise_thresh, contours)

    if contours:
        contours.sort(key=len, reverse=True)
        return [contours[0]]

    return None


# For detecting two hands, however in the scenes where just one hand is present, the red object might get confused with
# a hand
def get_biggest_two_contours(img, noise_thresh):
    im, contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_KCOS)

    if contours:
        contours.sort(key=len, reverse=True)

    # filter noise
    contours = filter(lambda cnt_x: len(cnt_x) > noise_thresh, contours)

    if contours:
        if len(contours) >= 2:
            return [contours[0], contours[1]]

        elif len(contours) == 1:
            return [contours[0]]

    else:
        return None


def get_contour_centroid(cnt):
    m = cv.moments(cnt)
    return int(m['m10'] / m['m00']), int(m['m01'] / m['m00'])


def draw_convex_hull(hull_points, frame):
    for point in hull_points:
        point = point.flatten()
        cv.circle(frame, (point[0], point[1]), 2, [255, 255, 255], -1)

    return frame


def draw_hull_with_defects(right_defects, frame):
    for i in range(len(right_defects)):
        sp, pe, pd, ld = right_defects[i]

        cv.line(frame, sp, pe, [255, 100, 255], 2)
        cv.line(frame, sp, pd, [25, 100, 255], 2)
        cv.line(frame, pe, pd, [25, 100, 255], 2)
        cv.circle(frame, pd, 2, [0, 0, 255], -1)
        cv.circle(frame, sp, 2, [0, 255, 0], -1)
        cv.circle(frame, pe, 2, [255, 0, 0], -1)
        cv.putText(frame, str(i), pd, cv.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 255], 1)

    return frame


def draw_fingertips(frame, hands):
    for hand in hands:
        for i in range(hand[0].shape[0]):
            cv.circle(frame, tuple(hand[0][i]), 2, [0, 255, 0], -1)
            cv.circle(frame, tuple(hand[1][i][2]), 2, [0, 255, 0], -1)
            # cv.line(frame, tuple(hand[1][i][2]), tuple(hand[0][i]), (255, 0, 0), 1)
            return frame


def draw_convex_defects(frame, hands):
    for hand in hands:
        defects = hand[3]
        for defect in defects:
            cs = hand[4][defect[0][0]].flatten()
            ce = hand[4][defect[0][1]].flatten()
            cf = hand[4][defect[0][2]].flatten()
            cv.circle(frame, tuple(cs), 2, [0, 255, 0], -1)
            cv.circle(frame, tuple(ce), 2, [0, 255, 0], -1)
            cv.circle(frame, tuple(cf), 2, [0, 255, 0], -1)
            cv.line(frame, tuple(cs), tuple(cf), (0, 0, 255), 2)
            cv.line(frame, tuple(ce), tuple(cf), (0, 0, 255), 2)

    return frame


def calc_and_draw_circles(frame, hands, centroids):
    ras = []
    rbs = []

    for i, hand in enumerate(hands):
        # min inscribed circle
        ra = int(cv.pointPolygonTest(hand, centroids[i], True))
        ras.append(ra)

        if ra > 0:
            cv.circle(frame, centroids[i], ra, [255, 255, 255], 1)

        # min enclosing circle
        (ecx, ecy), rb = cv.minEnclosingCircle(hand)
        center = (int(ecx), int(ecy))
        rb = int(rb)
        rbs.append(rb)

        if rb > 0:
            cv.circle(frame, center, rb, [255, 255, 255], 2)

    return frame, ras, rbs


def filter_defects_and_hull(cnt):
    # indices of hull points
    hull = cv.convexHull(cnt, returnPoints=False)

    # get defects
    defects = cv.convexityDefects(cnt, hull)

    if defects is not None:
        # convert defects to floats
        defects[:, 0, 3] = defects[:, 0, 3] / 256.0

        # set and apply empirically threshold to filter small defects (noise)
        depth_thesh = 4.00

        # Filter defects by threshold value
        filtered_defects = defects[np.where(defects[:, 0, 3] > depth_thesh)]

        # filtered_hull = filer_hull_defects(filtered_defects)

        return hull, filtered_defects

    else:
        return np.asarray([]), np.asarray([])


def filer_hull_defects(filtered_defects):
    filtered_hull = []

    for defect in filtered_defects:
        filtered_hull.append(defect[0][0])
        filtered_hull.append(defect[0][1])
    hull_set = set(filtered_hull)
    filtered_hull = list(hull_set)
    filtered_hull.sort(key=int)
    for first, second in zip(filtered_hull, filtered_hull[1:]):
        filtered_hull.append(int((second - first) / 2) + first)
    filtered_hull.sort(key=int)
    filtered_hull = np.asarray(filtered_hull)
    np.reshape(filtered_hull, (filtered_hull.shape[0], 1))

    return filtered_hull


def filter_defects_by_angle(filtered_defects, cnt, icy):
    right_defects = []

    for i in range(filtered_defects.shape[0]):
        sp, pe, pd, ld = filtered_defects[i, 0]
        sp = tuple(cnt[sp][0])
        pe = tuple(cnt[pe][0])
        pd = tuple(cnt[pd][0])
        if (140 > get_angle(sp, pe, pd) > 110) and pd[1] > icy:
            right_defects.append((sp, pe, pd, ld, get_angle(sp, pe, pd)))


def get_fingertips(cnt, centroid_y):
    # filters defects and hullpoints => defects are returned in case they are needed for preprocessing
    filtered_hull, filtered_defects = filter_defects_and_hull(cnt)

    # a possible indicator for pointing gesture => the angle the defects form with hull points

    # get fingertip candidates by getting every contour point which is also part of convex hull, additionally for each
    # such point get the contour points left and right along the hull => for calculating the angle
    if filtered_hull.shape[0] > 0:

        # distance of fingertip candidate to contour point
        distance = 12
        fingertip_angle = 40
        # for each hull point, get contour points +- some empirically defined distance
        fingertip_candidates = get_contour_points(cnt, filtered_hull.flatten(), distance)

        # calculate the angles between the fingertip candidate and the corresponding points left and right along the
        # contour (defined distance) if angle smaller than 60 degrees => considered fingertip
        points_on_fingertip, related_p1_p2_points = process_fingertip_candidates(fingertip_candidates, fingertip_angle)

        # shape of points_on_fingertip (nx2), shape of p1_p2_points(nx2x2) for each fingertip point 2 tuples
        # from process_fingertip_candidates we got fingertip candidates, but some fingertip candidates will be on the
        # same fingertip. therefore we cluster the fingertips together with their corresponding curvature points
        fingertips, p1_p2_points = cluster_fingertip_candidates(points_on_fingertip, related_p1_p2_points)

        return fingertips, p1_p2_points, filtered_defects, filtered_hull

    else:
        return np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([])


def get_contour_points(cnt, hull, distance):
    v_get_x_range = np.vectorize(get_x_range)

    cnt = np.reshape(cnt, (cnt.shape[0], 2))
    candidate_indices = v_get_x_range(hull, cnt.shape[0] - 1, distance)

    return cnt[candidate_indices[0]], cnt[candidate_indices[1]], cnt[candidate_indices[2]]


def process_fingertip_candidates(candidates, angle_thresh):
    fingertip_candidates = []
    related_p_points = []

    for i in range(candidates[0].shape[0]):
        p1 = candidates[0][i]
        p2 = candidates[1][i]
        p0 = candidates[2][i]

        angle = get_angle(p1, p2, p0)

        if angle < angle_thresh:
            fingertip_candidates.append(p0)
            related_p_points.append((p1, p2))

    return np.asarray(fingertip_candidates), np.asarray(related_p_points)


def cluster_fingertip_candidates(points_on_fingertip, related_curvature_points):
    fingertips = []
    curve_points = []

    if not points_on_fingertip.size == 0:
        if points_on_fingertip.size > 2:
            clusters = hcluster.fclusterdata(points_on_fingertip, 20, criterion="distance")

            for cluster in list(set(clusters)):
                # for each cluster: get the indices of the corresponding fingertip and curvature tuples
                fingertip_candidates = points_on_fingertip[np.where(clusters == cluster)]
                curve_point_candidates = related_curvature_points[np.where(clusters == cluster)]

                # calculate centroid fingertip candidates of current cluster => calc final fingertip
                fingertip = calc_centroid_of_points(fingertip_candidates)
                fingertips.append(fingertip)

                # Calculates centroid of all curve points minus the fingertip candidate
                curve_point_minus = calc_centroid_of_points(curve_point_candidates[:, 0])
                # Calculates centroid of all curve points plus the fingertip candidate
                curve_point_plus = calc_centroid_of_points(curve_point_candidates[:, 1])
                curve_point_center = calc_centroid_of_points(np.asarray([curve_point_minus, curve_point_plus]))

                curve_points.append([curve_point_minus, curve_point_plus, curve_point_center])

        else:

            # reshape related curvature to nx2 with related_curvature_points[0]
            curve_point_center = calc_centroid_of_points(related_curvature_points[0])
            # tuple minus fingertip
            curve_point_minus = tuple(related_curvature_points[0][0])
            # tuple plus fingertip
            curve_point_plus = tuple(related_curvature_points[0][1])

            curve_points.append([curve_point_minus, curve_point_plus, curve_point_center])
            fingertips.append(points_on_fingertip[0])

    return np.asarray(fingertips), np.asarray(curve_points)


# cnt and p1 shape => (n, 1, 2), good_old and good_new => (n, 2)
def calc_optical_flow(frame, pre_frame, next_frame, cnt, params, threshold, mask):
    motion = False

    p1, st, err = cv.calcOpticalFlowPyrLK(pre_frame, next_frame, np.float32(cnt), None, **params)

    good_new = p1[st == 1]
    good_old = cnt[st == 1]

    if good_new.size > 0:
        # filter optical flow
        f_new = np.zeros((0, 2), dtype=np.int32)
        f_old = np.zeros((0, 2), dtype=np.int32)

        for n, o in zip(good_new, good_old):
            if np.linalg.norm(n - o) > threshold:
                f_new = np.append(f_new, np.reshape(n, (1, 2)), axis=0)
                f_old = np.append(f_old, np.reshape(o, (1, 2)), axis=0)

        f_new = np.int32(f_new)

        # draw bounding box around flow
        x, y, w, h = cv.boundingRect(np.reshape(f_new, (f_new.shape[0], 1, 2)))
        # if bounding box (0, 0, 0, 0) => no motion > threshold detected
        if (x, y, w, h) == (0, 0, 0, 0):
            motion = False
        else:
            motion = True

        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), 2)

        # draw optical flow lines
        for f_n, f_o in zip(f_new, f_old):
            mask = cv.line(mask, tuple(f_n), tuple(f_o), (255, 0, 0), 1)
            frame = cv.line(frame, tuple(f_n), tuple(f_o), (255, 0, 0), 1)

        return frame, mask, motion

    return frame, mask, motion


def get_angle(sp, pe, pd):
    ba = np.subtract(sp, pd)
    bc = np.subtract(pe, pd)
    # linalg.norm => eucleadian norm => np.sqrt(a1^2 + a2^2 + ... + aN^2)
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def get_x_range(x, cnt_length, distance):
    return max(0, x - distance), min(cnt_length, x + distance), x


def calc_centroid_of_points(points):
    # length is the number of tuples/points, shape of matrix: length x 2
    length = points.shape[0]
    sum_x = np.sum(points[:, 0])
    sum_y = np.sum(points[:, 1])
    return sum_x / length, sum_y / length


def calc_tracking_roi(source, target, padding, bounds):
    x_0, y_0, w_0, h_0 = bounds

    target[max(0, y_0 - padding): min(y_0 + h_0 + padding, 600),
    max(0, x_0 - padding): min(x_0 + w_0 + padding, 800)] = source[
                                                            max(0, y_0 - padding): min(y_0 + h_0 + padding,
                                                                                       600),
                                                            max(0, x_0 - padding): min(x_0 + w_0 + padding,
                                                                                       800)]

    return target


# calcs line from two points and returns a, b, c from formula: ax + by = c
def calc_line(p1, p2):
    a = (p1[1] - p2[1])
    b = (p2[0] - p1[0])
    c = (p1[0] * p2[1] - p2[0] * p1[1])
    return a, b, -c


# l1 and l2 each represent a line by containing the params a(l[0]), b(l[1]), -c[l2]
# The intersection can be found by calculating determinants d, dx and dy
def intersection(l1, l2):
    d = l1[0] * l2[1] - l1[1] * l2[0]
    dx = l1[2] * l2[1] - l1[1] * l2[2]
    dy = l1[0] * l2[2] - l1[2] * l2[0]
    if d != 0:
        x = dx / d
        y = dy / d
        return x, y
    else:
        return False


def calc_and_clip_pointing_array(frame, p_3, fingertip, object_bb):
    l1 = calc_line(tuple(p_3), tuple(fingertip))
    l2 = calc_line((0, frame.shape[0]), (frame.shape[1], frame.shape[0]))
    inters = intersection(l1, l2)

    if inters:
        cv.line(frame, tuple(p_3), inters, (0, 0, 255), 2)
        cv.circle(frame, inters, 4, [255, 255, 255], -1)

        gwr_angle = get_angle([0, fingertip[1]], inters, fingertip)
        # cv.putText(frame, "The angle is: " + str(gwr_angle), (40, 20), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1,
        #            cv.LINE_AA)

        probabilities = []
        for box in object_bb:
            retval, p1, p2 = cv.clipLine(box[1], tuple(p_3), inters)
            if retval:
                cv.circle(frame, p1, 4, [255, 255, 255], -1)
                cv.circle(frame, p2, 4, [255, 255, 255], -1)

                prob = calc_poiting_probability(box[2], p1, p2)
                probabilities.append((prob, box[0]))

            if probabilities:
                probabilities = sorted(probabilities, key=lambda x: x[1])
                detection_text = "Pointing towards: " + probabilities[0][1] + " (" + str(
                    round(probabilities[0][0], 2)) + ")"
                cv.putText(frame, detection_text, (10, 400), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv.LINE_AA)

                return frame

        return frame
    else:
        return frame


# first approach => cut of line clip smaller / bigger half => value between 0 - 1
# box => (b[0], b[1]), (b[2], b[3])
def calc_poiting_probability(box, p1, p2):
    box_1 = np.array([[box[0], box[1]], p1, p2, [box[0], box[3]]], np.int32)
    box_1 = box_1.reshape((-1, 1, 2))
    area_1 = cv.contourArea(box_1)

    box_2 = np.array([p1, [box[2], box[1]], [box[2], box[3]], p2], np.int32)
    box_2 = box_2.reshape((-1, 1, 2))
    area_2 = cv.contourArea(box_2)

    if area_1 > area_2:
        return area_2 / area_1

    elif area_1 == area_2:
        return 1.0

    else:
        return area_1 / area_2


def check_defects_for_pointing(frame, cnt, defects, fingertip, hand_centroid):
    radius = np.linalg.norm(fingertip - hand_centroid)
    center = fingertip
    defect_count = 0
    # cv.circle(frame, tuple(center), int(radius), [255, 255, 255])

    for defect in defects:
        defect_point = cnt[defect[0][2]].flatten()
        # cv.circle(frame, tuple(defect_point), 2, [0, 255, 0], -1)

        if np.square(defect_point[1] - center[1]) + np.square(defect_point[0] - center[0]) < np.square(radius):
            defect_count += 1
            # cv.circle(frame, tuple(defect_point), 2, [0, 0, 255], -1)

    return frame, defect_count


def visualize_poiting_direction_gwr(frame, fingertip, fitted_line):
    left_to_fingertip = calc_line((0, fingertip[1]), tuple(fingertip))
    pointing_direction = calc_line(fitted_line[0], fitted_line[1])
    bottom_line = calc_line((0, frame.shape[0]), (frame.shape[1], frame.shape[0]))

    intersec_fingertip = intersection(left_to_fingertip, pointing_direction)
    intersec_bottom = intersection(pointing_direction, bottom_line)

    cv.line(frame, (0, fingertip[1]), (frame.shape[1], fingertip[1]), (0, 255, 255), 2)
    cv.line(frame, (0, fingertip[1]), intersec_fingertip, (0, 0, 255), 2)
    cv.line(frame, intersec_fingertip, intersec_bottom, (255, 255, 255), 2)

    return frame


def get_hand_features_for_GWR(frame, fingertip, fitted_line):
    # calc lines for fitted line of hand shape
    left_to_fingertip = calc_line((0, fingertip[1]), tuple(fingertip))
    pointing_direction = calc_line(fitted_line[0], fitted_line[1])
    bottom_line = calc_line((0, frame.shape[0]), (frame.shape[1], frame.shape[0]))

    intersec_fingertip = intersection(left_to_fingertip, pointing_direction)
    intersec_bottom = intersection(pointing_direction, bottom_line)

    angle = get_angle((0, fingertip[1]), intersec_bottom, intersec_fingertip)

    return angle


def calculate_contour_line(frame, hand):
    rows, cols = frame.shape[:2]
    [vx, vy, x, y] = cv.fitLine(hand, cv.DIST_L2, 0, 0.01, 0.01)
    lefty = int((-x * vy / vx) + y)
    righty = int(((cols - x) * vy / vx) + y)
    return [(0, lefty), (cols - 1, righty)]


def predict_gwr_pointing(frame, frame2, hand_contour, fingertip, hand_centroid, gwr, hand_positions_t0):
    fitted_line = calculate_contour_line(frame, hand_contour)
    angle = get_hand_features_for_GWR(frame, fingertip, fitted_line)

    observation = [angle, hand_centroid[0], hand_centroid[1], fingertip[0], fingertip[1]]

    norm_observation = gwr.normalize_live_observation(observation)

    gwr_prediction = gwr.predict_live(norm_observation)
    if gwr_prediction[0] is not None:
        bmu, union_of_best, activation, x_best_labels = gwr_prediction

        # for box in x_best_labels:
        #     cv.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 255, 255), 2)

        # cv.rectangle(frame, (bmu[0], bmu[1]), (bmu[2], bmu[3]), (0, 0, 255), 2)

        # after predicting the pointing position we check whether there is an ambiguity
        ambiguity, bounding_boxes, b_frame = obj_d.check_for_ambiguity(frame, union_of_best, hand_positions_t0)

        if ambiguity:
            cv.rectangle(frame, (bmu[0], bmu[1]), (bmu[2], bmu[3]), (0, 0, 255), 2)
            cv.rectangle(frame, (union_of_best[0], union_of_best[1]), (union_of_best[2], union_of_best[3]),
                         (0, 255, 255), 2)
            # object_bb: 0: "object_color", 1: bb_wh, 2: bb])
            for d_object in bounding_boxes:
                bb = d_object[0]
                cv.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (255, 255, 255), 2)
                iou = gwr.calc_iou(bb, bmu)
                if iou > .5:
                    detection_text = "Pointing towards: " + d_object[1] + " (" + str(iou) + ")"
                    cv.putText(frame, detection_text, (10, 400), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1,
                               cv.LINE_AA)

            return frame, frame2
        else:
            if len(bounding_boxes) > 0:
                # cv.rectangle(frame2, (union_of_best[0], union_of_best[1]), (union_of_best[2], union_of_best[3]),
                #              (0, 255, 255), 2)
                d_object = bounding_boxes[0]
                detection_text = "Pointing towards: " + d_object[1]
                cv.putText(frame, detection_text, (10, 400), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
                return frame, frame2

            else:
                detection_text = "No pointing without a target"
                cv.putText(frame, detection_text, (10, 400), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
                return frame, frame2
    else:
        cv.putText(frame, "ACTIVATION LOW " + str(gwr_prediction[1]), (10, 400), cv.FONT_HERSHEY_SIMPLEX, 1,
                   (255, 255, 255), 2, cv.LINE_AA)
        return frame, frame2
