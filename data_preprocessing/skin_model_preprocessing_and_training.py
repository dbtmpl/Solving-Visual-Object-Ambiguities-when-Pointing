import cv2 as cv
import numpy as np
import glob
import matplotlib.pyplot as plt


def load_original_img():
    return [cv.imread(skin_image) for skin_image in glob.glob("../resources/skin_color_data/original/*.jpg")]


def load_ground_truth():
    return [cv.imread(gt) for gt in glob.glob("../resources/skin_color_data/ground_truth/*.jpg")]


def get_originals_and_gt():
    ground_truth = load_ground_truth()
    ground_truth_preprocessed = preprocess_groundtruth(ground_truth)
    originals = load_original_img()

    return originals, ground_truth_preprocessed


def preprocess_groundtruth(ground_truth):
    prep_gt = []
    for gt in ground_truth:
        gt = cv.cvtColor(gt, cv.COLOR_BGR2GRAY)
        ret2, thresh = cv.threshold(gt, 5, 255, cv.THRESH_BINARY)
        prep_gt.append(thresh)

    return prep_gt


def save_groundtruth(groundtruth):
    for i, gt in enumerate(groundtruth):
        cv.imwrite("resources/lab_settings_data/preprocessed_gt/gt_" + str(i) + ".jpg", gt)


def save_trainingpairs(pairs):
    for i, pair in enumerate(pairs):
        cv.imwrite("resources/lab_settings_data/preprocessed_gt/gt_" + str(i) + ".jpg", pair[0])
        cv.imwrite("resources/lab_settings_data/preprocessed_gt/ori_" + str(i) + ".jpg", pair[1])


def extract_skin_data(ground_truth, originals):
    skin_data = np.zeros((0, 3), dtype=np.uint8)
    for i, gt in enumerate(ground_truth):
        img = originals[i]
        skin = img[np.where(gt == 255)]
        skin_data = np.append(skin_data, skin, axis=0)

    np.save("../resources/skin_color_data/pixel_data/skin_pixels", np.uint8(skin_data))


def extract_noskin_data(ground_truth, originals):
    noskin_data = np.zeros((0, 3), dtype=np.uint8)
    for i, gt in enumerate(ground_truth):
        img = originals[i]
        no_skin = img[np.where(gt == 0)]
        noskin_data = np.append(noskin_data, no_skin, axis=0)

    np.save("../resources/skin_color_data/pixel_data/no_skin_pixels", np.uint8(noskin_data))


def setup_skin_color_data():
    originals, ground_truth = get_originals_and_gt()

    extract_skin_data(ground_truth, originals)
    extract_noskin_data(ground_truth, originals)
    print("Skin color data saved")


def get_skin_color_data():
    skin_data = np.load("../resources/skin_color_data/pixel_data/skin_pixels.npy")
    no_skin_data = np.load("../resources/skin_color_data/pixel_data/no_skin_pixels.npy")

    return skin_data, no_skin_data


def get_ycrcb_skin_data(skin, no_skin):
    ycrcb_skin = cv.cvtColor(np.reshape(skin, (skin.shape[0], 1, 3)), cv.COLOR_BGR2YCrCb)
    ycrcb_no_skin = cv.cvtColor(np.reshape(no_skin, (no_skin.shape[0], 1, 3)), cv.COLOR_BGR2YCrCb)

    return np.reshape(ycrcb_skin, (ycrcb_skin.shape[0], 3)), np.reshape(ycrcb_no_skin, (ycrcb_no_skin.shape[0], 3))


def get_crcb_hist(skin, no_skin):
    ycrcb_skin = cv.cvtColor(np.reshape(skin, (skin.shape[0], 1, 3)), cv.COLOR_BGR2YCrCb)
    ycrcb_no_skin = cv.cvtColor(np.reshape(no_skin, (no_skin.shape[0], 1, 3)), cv.COLOR_BGR2YCrCb)

    crcb_skin_hist = cv.calcHist([ycrcb_skin], [1, 2], None, [256, 256], [0, 256, 0, 256])
    crcb_no_skin_hist = cv.calcHist([ycrcb_no_skin], [1, 2], None, [256, 256], [0, 256, 0, 256])

    np.save("../resources/skin_color_data/histograms/cr_cb_skin", np.float64(crcb_skin_hist))
    np.save("../resources/skin_color_data/histograms/cr_cb_no_skin", np.float64(crcb_no_skin_hist))

    return np.float64(crcb_skin_hist), np.float64(crcb_no_skin_hist)


def calc_3d_data_hist(bgr_data):
    hist, edges = np.histogramdd(bgr_data, bins=(256, 256, 256))

    return hist, edges


def calc_bayesian_hist(hist_skin, hist_no_skin):
    # A survey of skin-color modeling and detection methods (2.2.2. Histogram model with naive bayes classifier)

    Ts = np.sum(hist_skin)
    Tn = np.sum(hist_no_skin)

    p_c_skin = np.divide(hist_skin, Ts)
    p_c_noskin = np.divide(hist_no_skin, Tn)

    np.save("../resources/skin_color_data/histograms/cr_cb_skin_prob", np.float64(p_c_skin))
    np.save("../resources/skin_color_data/histograms/cr_cb_no_skin_prob", np.float64(p_c_noskin))

    # Prevent nan type and lost information
    p_c_skin[np.where(p_c_skin == 0.0)] = np.power(10, -100.0)
    p_c_noskin[np.where(p_c_noskin == 0.0)] = np.power(10, -100.0)
    skin_prob = np.divide(p_c_skin, p_c_noskin)

    return skin_prob


def train_bgr_3d_hist():
    skin, no_skin = get_skin_color_data()

    skin_hist, skin_edges = calc_3d_data_hist(skin)
    no_skin_hist, no_skin_edges = calc_3d_data_hist(no_skin)

    skin_prob = calc_bayesian_hist(skin_hist, no_skin_hist)

    np.save("../resources/skin_color_data/histograms/skin_probabilities_bgr", np.float64(skin_prob))
    print("success")


def train_ycrcb_3d_hist():
    skin, no_skin = get_skin_color_data()

    ycrcb_skin, ycrcb_no_skin = get_ycrcb_skin_data(skin, no_skin)

    skin_hist, skin_edges = calc_3d_data_hist(ycrcb_skin)
    no_skin_hist, no_skin_edges = calc_3d_data_hist(ycrcb_no_skin)

    skin_prob = calc_bayesian_hist(skin_hist, no_skin_hist)

    np.save("../resources/lab_settings_data/pixel_data/skin_probabilities_ycrcb", np.float64(skin_prob))
    print("success")


def train_crcb_2d_hist():
    setup_skin_color_data()
    skin, no_skin = get_skin_color_data()

    crcb_skin_hist, crcb_no_skin_hist = get_crcb_hist(skin, no_skin)
    skin_prob = calc_bayesian_hist(crcb_skin_hist, crcb_no_skin_hist)
    np.save("resources/skin_color_data/histograms/skin_probabilities_crcb", np.float64(skin_prob))
    print("success")


def plot_2d_hist(hist, name):
    plt.matshow(hist, interpolation='None')
    plt.colorbar()
    plt.savefig("../resources/skin_color_data/histograms/figures/" + name + ".png")
    plt.show()


def normalize_histogram(hist_old):
    hist_old = np.array(hist_old).astype('float')
    hist = np.copy(hist_old)
    size = hist.shape

    hist_min = np.amin(hist_old)
    hist_max = np.amax(hist_old)

    for i in range(0, size[0] - 1):
        for j in range(0, size[1] - 1):
            hist[j, i] = (hist_old[j, i] - hist_min) / (hist_max - hist_min)
    return hist


if __name__ == "__main__":
    train_crcb_2d_hist()
    skin_prob_crcb_link = "resources/skin_color_data/histograms/skin_probabilities_crcb.npy"
    skin_prob_binary_crcb = gesture.get_skin_histogram(skin_prob_crcb_link)
