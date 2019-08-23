import cv2 as cv
import numpy as np
import csv
from mechanics import object_detection as obj_d, gesture_detection as gesture, state_model as state


def calc_raw_confusion_data(data, side_data, dimensions):
    # Bayesian Hist
    skin_prob_crcb_link = "resources/skin_color_data/histograms/skin_probabilities_crcb.npy"
    thresh_crcb = 10
    skin_prob_binary_crcb = gesture.get_lab_skin_hist(thresh_crcb, skin_prob_crcb_link)

    sample_count = data.shape[0]

    # List of lists: 0: overall, 1-4 respective ambiguity class
    # In list 0. TP, 1. FP, 2. FN, 3. misses, 4. no intersections 5. total count
    confusion_values = [[0, 0, 0, 0, 0, sample_count], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0]]

    pointing_qualities_tp = []
    pointing_qualities_fp = []

    for iteration in range(0, sample_count):
        print(iteration)
        observation = data[iteration]
        c_side_data = side_data[iteration]
        filename = c_side_data[0]
        ambiguity_class = int(filename.split("_")[0][-1])
        # label in this case => 0: red, 1: yellow, 2: green
        target_color = int(c_side_data[1])
        confusion_values[ambiguity_class][5] += 1
        corres_image = cv.imread("resources/current_training/bb_test/all_obj_permutations/" + filename + ".jpg")

        frame_ycrcb = cv.cvtColor(corres_image, cv.COLOR_BGR2YCrCb)
        skin_binary = gesture.apply_skin_hist2d(frame_ycrcb, skin_prob_binary_crcb)
        frame, hand_positions_t0, tracking_state = state.detection_step(corres_image, skin_binary, [None, None], "None")

        object_bb = []
        detected_objects = obj_d.detect_objects(corres_image, hand_positions_t0)
        for i, d_object in enumerate(detected_objects):
            if d_object is not None:
                bb, color_str, color_int = d_object
                object_bb.append([color_int, (bb[0], bb[1], bb[2] - bb[0], bb[3] - bb[1]), bb])

        fingertip = observation[0], observation[1]
        p3 = observation[2], observation[3]

        hit_target, (pointing_quality, p_color) = calculate_pointing(fingertip, p3, dimensions, object_bb)
        if hit_target is not None:
            if hit_target:
                if pointing_quality > 0.2:
                    if target_color == p_color:
                        # TP
                        confusion_values[0][0] += 1
                        confusion_values[ambiguity_class][0] += 1
                        pointing_qualities_tp.append(pointing_quality)
                    else:
                        # add FP
                        confusion_values[0][1] += 1
                        confusion_values[ambiguity_class][1] += 1
                        pointing_qualities_fp.append(pointing_quality)
                        # add FN
                        confusion_values[0][2] += 1
                        confusion_values[ambiguity_class][2] += 1
                else:
                    # add FN
                    confusion_values[0][2] += 1
                    confusion_values[ambiguity_class][2] += 1
                    # add Miss
                    confusion_values[0][3] += 1
                    confusion_values[ambiguity_class][3] += 1
            else:
                # add FN
                confusion_values[0][2] += 1
                confusion_values[ambiguity_class][2] += 1
                # add Miss
                confusion_values[0][3] += 1
                confusion_values[ambiguity_class][3] += 1
        else:
            # add FN
            confusion_values[0][4] += 1
            confusion_values[ambiguity_class][4] += 1

    return confusion_values, [pointing_qualities_tp, pointing_qualities_fp]


def calculate_results(confusion_values, pointing_quality):
    mean_pq_tp = sum(pointing_quality[0]) / float(len(pointing_quality[0]))
    if len(pointing_quality[1]) > 0:
        mean_pq_fp = sum(pointing_quality[1]) / float(len(pointing_quality[1]))
    else:
        mean_pq_fp = 0
    total_mean = sum(pointing_quality[0] + pointing_quality[1]) / float(len(pointing_quality[0] + pointing_quality[1]))

    pointing_qualities = [mean_pq_tp, mean_pq_fp, total_mean]

    # list of means of each fold
    # 0: overall, 1- 4 ambiguity classes

    # 0: TP, 1: FP. 2: FN, 3. misses_count
    tp_fp_fn_m = [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]]

    # 0. Precision, 1, Recall, 2. F1-score, 3: Misses
    p_r_f_m = [[0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0]]

    # In list 0. TP, 1. FP, 2. FN, 3. misses, 4. no intersections 5. total count
    for amb_c, amb_class in enumerate(confusion_values):
        count = amb_class[-1]
        tp_count, fp_count, fn_count, misses_count, no_inters = amb_class[:-1]
        count -= no_inters
        print(tp_count)
        print(fn_count)
        print(tp_count + fn_count)
        precision = float(tp_count) / (tp_count + fp_count)
        recall = float(tp_count) / (tp_count + fn_count)
        print(recall)
        f1_score = 2 * ((precision * recall) / (precision + recall))
        misses = misses_count / float(count)

        tp_fp_fn_m[amb_c][0] = int(tp_count)
        tp_fp_fn_m[amb_c][1] = int(fp_count)
        tp_fp_fn_m[amb_c][2] = int(fn_count)
        tp_fp_fn_m[amb_c][3] = int(misses_count)
        tp_fp_fn_m[amb_c][4] = int(count)
        p_r_f_m[amb_c][0] = precision
        p_r_f_m[amb_c][1] = recall
        p_r_f_m[amb_c][2] = f1_score
        p_r_f_m[amb_c][3] = misses

    return tp_fp_fn_m, p_r_f_m, pointing_qualities


def save_results(results1, results2, pq, link):
    print(results2)
    np.save(link + "/results/cs_tp_fp_fn_m", np.int32(results1))
    np.save(link + "/results/cs_p_r_f_m", np.float32(results2))
    np.save(link + "/results/cs_pointing_quality", np.float64(pq))


def save_raw_data(results, pq, link):
    np.save(link + "/results/raw/raw_data", np.float64(results))
    with open(link + "/results/raw/raw_pq.csv", "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(pq)


def print_outcome(pointing_quality, result_values):
    # in sublists0: TP, 1: FP. 2: FN, 3. Precision, 4, Recall, 5. F1-score, 6: Misses
    print("Pointing qualities:")
    print("TP Pointing qualities:")
    print(pointing_quality[0])
    print("")
    print("FP Pointing qualities:")
    print(pointing_quality[1])
    print("")
    print("OVERALL Pointing qualities:")
    print(pointing_quality[2])
    print("")

    for i, results in enumerate(result_values):
        print("ambiguity class: " + str(i))

        print("TP: ")
        print(results[0])
        print("")
        print("FP: ")
        print(results[1])
        print("")
        print("FN: ")
        print(results[2])
        print("")
        print("Misses count")
        print(results[3])
        # print("")
        # print("Precision")
        # print(results[4])
        # print("")
        # print("Recall")
        # print(results[5])
        # print("")
        # print("F1_score")
        # print(results[6])
        # print("")
        # print("Misses")
        # print(results[7])
        # print(" ")
        # print(" ")


def results_to_latex_tabular(resuls):
    print " \\\\\n".join([" & ".join(map(str, line)) for line in resuls])


def calculate_pointing(fingertip, p3, dimensions, d_objects):
    l1 = gesture.calc_line(tuple(p3), tuple(fingertip))
    l2 = gesture.calc_line((0, dimensions[0]), (dimensions[1], dimensions[0]))
    inters = gesture.intersection(l1, l2)

    if inters:
        # check pointing to target object
        p3_int = (int(p3[0]), int(p3[1]))
        inters_int = (int(inters[0]), int(inters[1]))

        # check pointing quality to objects
        pointing_qualities_color = []
        for d_obj in d_objects:
            color, bb_wh, bb = d_obj[0], d_obj[1], d_obj[2]
            retval, intersection1, intersection2 = cv.clipLine(bb_wh, tuple(p3_int), inters_int)
            if retval:
                pointing_quality = gesture.calc_poiting_probability(bb, intersection1, intersection2)
                pointing_qualities_color.append([pointing_quality, color])

        if len(pointing_qualities_color) > 0:
            best_pq = max(pointing_qualities_color, key=lambda x: x[1])
            return True, best_pq
        else:
            return False, [None, None]

    else:
        return None, [None, None]


def convert_bb_to_pxv(bb, dimensions):
    return [int(bb[0] * dimensions[1]), int(bb[1] * dimensions[0]),
            int(bb[2] * dimensions[1]), int(bb[3] * dimensions[0])]


if __name__ == "__main__":
    link_data = "resources/cs_pointing_data/15_07_fingertip_p3/noise_filtered"
    data = np.load(link_data + "/data.npy")
    labels = np.load(link_data + "/labels.npy")
    side_data = list(csv.reader(open(link_data + "/side_data.csv", "rU"), delimiter=','))
    print(data.shape)

    frame_dimensions = (950, 700)
    # confusion_values, pq = calc_raw_confusion_data(data, side_data, frame_dimensions)
    # save_raw_data(confusion_values, pq, link_data)

    confusion_values = np.load(link_data + "/results/raw/raw_data.npy")
    pq = list(csv.reader(open(link_data + "/results/raw/raw_pq.csv", "rU"), delimiter=','))
    for i, p in enumerate(pq):
        for j, q in enumerate(p):
            p[j] = float(p[j])

    tp_fp_fn_m, p_r_f_m, mean_pointing_qualities = calculate_results(confusion_values, pq)
    save_results(tp_fp_fn_m, p_r_f_m, mean_pointing_qualities, link_data)

    tp_fp = np.load(link_data + "/results/cs_tp_fp_fn_m.npy")
    prfm = np.load(link_data + "/results/cs_p_r_f_m.npy")

    print_outcome(mean_pointing_qualities, prfm)
    results_to_latex_tabular(tp_fp)
    results_to_latex_tabular((np.around(prfm, decimals=4)*100))
