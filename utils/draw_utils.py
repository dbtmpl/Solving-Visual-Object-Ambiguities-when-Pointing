import cv2 as cv


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


def draw_convex_hull(hull_points, frame):
    """
    Draws a convex hull onto a given frame
    :param hull_points:
    :param frame:
    :return:
    """
    for point in hull_points:
        point = point.flatten()
        cv.circle(frame, (point[0], point[1]), 2, [255, 255, 255], -1)

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