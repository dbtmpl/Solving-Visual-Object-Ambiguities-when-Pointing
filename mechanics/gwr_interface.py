import cv2 as cv
import numpy as np
import math
import ast
import csv
import matplotlib.pyplot as plt

from mechanics import object_detection as obj_d, gesture_detection as gesture, state_model as state


class GWRInterface(object):
    def __init__(self, link_to_data, frame_dimensions):
        gwr = self.get_trained_gwr(link_to_data)
        self.weights, self.edges, self.labels, self.error = gwr
        print(self.weights.shape)

        self.number_nodes = self.weights.shape[0]
        self.number_edges = self.get_no_edges()
        self.aT = .5

        self.x_b_l_value = int(np.ceil(self.number_nodes * 0.01) + 5)

        self.frame_dimensions = frame_dimensions
        print("HERE SET FRAME DIMS")
        print(self.frame_dimensions)

        skin_prob_crcb_link = "resources/skin_color_segmentation/saved_histograms/skin_probabilities_crcb.npy"
        thresh_crcb = 10
        self.skin_prob_binary_crcb = gesture.get_lab_skin_hist(thresh_crcb, skin_prob_crcb_link)

    def get_trained_gwr(self, link_to_data):
        gwr_weights = np.load(link_to_data + "/weights.npy")
        gwr_edges = np.load(link_to_data + "/edges.npy")
        gwr_labels = np.load(link_to_data + "/labels.npy")
        error_count = np.load(link_to_data + "/error_count.npy")

        return gwr_weights, gwr_edges, gwr_labels, error_count

    def predict_live(self, gwr_input):
        distances = np.zeros(self.number_nodes)

        for i in range(0, self.number_nodes):
            distances[i] = self.metric_distance(self.weights[i], gwr_input)

        first_index = distances.argmin()
        first_distance = distances.min()
        activation = math.exp(-first_distance)

        if activation > self.aT:
            p_label = self.labels[first_index]

            # here activate neighborhood approach or take 4 best nodes
            # neighborhood_labels = self.check_neighborhood(first_index)
            x_best_labels = self.get_x_best_labels(distances, 15)
            union_of_best = self.union_bounding_boxes(x_best_labels)

            return self.convert_bb_to_pxv(p_label), union_of_best, activation, x_best_labels
        else:
            return None, activation

    def get_x_best_labels(self, distances, x):
        x_best_nodes = distances.argsort()[1:x]

        fbl = []
        for i in x_best_nodes:
            fbl.append(self.convert_bb_to_pxv(self.labels[i]))

        return fbl

    # gets neighbors bmu and returns its lables
    def check_neighborhood(self, first_index):
        neighborhood_labels = []
        neighborhood_bmu = np.nonzero(self.edges[first_index])
        for z in range(0, len(neighborhood_bmu[0])):
            neighbor_index = neighborhood_bmu[0][z]
            neighborhood_labels.append(self.convert_bb_to_pxv(self.labels[neighbor_index]))

        return neighborhood_labels

    def union_bounding_boxes(self, bounding_boxes):
        min_x1 = []
        min_y1 = []
        max_x2 = []
        max_y2 = []
        for box in bounding_boxes:
            min_x1.append(box[0])
            min_y1.append(box[1])
            max_x2.append(box[2])
            max_y2.append(box[3])

        x1 = min(min_x1)
        y1 = min(min_y1)
        x2 = max(max_x2)
        y2 = max(max_y2)

        return np.asarray([x1, y1, x2, y2])

    def metric_distance(self, x, y):
        # Euclidean distance
        # return np.sqrt(numpy.sum((x-y)**2))
        return np.linalg.norm(x - y)

    def convert_bb_to_pxv(self, bb):
        return [int(bb[0] * self.frame_dimensions[1]), int(bb[1] * self.frame_dimensions[0]),
                int(bb[2] * self.frame_dimensions[1]), int(bb[3] * self.frame_dimensions[0])]

    def normalize_with_fixed_min_max(self, data_set, dimensions):
        n_data_set = np.copy(data_set)
        size = data_set.shape
        min_max_values = [(0, 180), (0, dimensions[1]), (0, dimensions[0]), (0, dimensions[1]), (0, dimensions[0])]
        for i in range(0, size[1]):
            maxColumn = min_max_values[i][1]
            minColumn = min_max_values[i][0]
            for j in range(0, size[0]):
                n_data_set[j, i] = (data_set[j, i] - minColumn) / (maxColumn - minColumn)

        return n_data_set

    # normalizes the data on basis of their min_max values as not reference distribution is available.
    def normalize_live_observation(self, observation):
        norm_observation = []
        min_max_values = [(74.20, 114.01), (86.0, 543.0), (216.0, 375.0),
                          (59.0, 610.0), (384.0, 619.0)]

        for value in zip(observation, min_max_values):
            norm_observation.append((value[0] - value[1][0]) / float(value[1][1] - value[1][0]))

        return np.asarray(norm_observation, np.float64)

    def calc_iou(self, detected_obj, gwr_prediction):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(detected_obj[0], gwr_prediction[0])
        yA = max(detected_obj[1], gwr_prediction[1])
        xB = min(detected_obj[2], gwr_prediction[2])
        yB = min(detected_obj[3], gwr_prediction[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both the prediction and ground-truth rectangles
        area_bb1 = (detected_obj[2] - detected_obj[0] + 1) * (detected_obj[3] - detected_obj[1] + 1)
        area_bb2 = (gwr_prediction[2] - gwr_prediction[0] + 1) * (gwr_prediction[3] - gwr_prediction[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(area_bb1 + area_bb2 - interArea)

        # return the intersection over union value
        return iou

    # first evaluate general GWR performance and see which hyperparameters are best
    # then introduce the detect ambiguity metric and evaluate approach on that
    # generally incorporate cross validation?

    def quantitatively_evaluate_gwr(self, testing_data):
        test_data, test_labels, side_data = testing_data

        samples = test_data.shape[0]
        number_nodes = self.weights.shape[0]
        distances = np.zeros(number_nodes)
        activations = np.zeros(samples)

        intersection_over_union = []

        # List of lists: 0: overall, 1-4 respective ambiguity class
        # In list 0. TP, 1. FP, 2. FN, 3. misses, 4. total count
        confusion_values = [[0, 0, 0, 0, samples], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]]

        for iteration in range(0, samples):
            # print(iteration)
            observation = test_data[iteration]
            obs_label = test_labels[iteration]
            filename = side_data[iteration][0]
            ambiguity_class = int(filename.split("_")[0][-1])
            confusion_values[ambiguity_class][4] += 1

            # saves outcome for visualizations
            outcome = ""

            for i in range(0, number_nodes):
                distances[i] = self.metric_distance(self.weights[i], observation)

            c_side_data = side_data[iteration]
            # test_bbs: 1: prediction, target bb, if available other false positive bbs

            first_index = distances.argmin()
            first_distance = distances.min()
            activations[iteration] = math.exp(-first_distance)
            p_label = self.labels[first_index]

            pre_label_int = self.convert_bb_to_pxv(p_label)

            gt_label_int = self.convert_bb_to_pxv(obs_label)
            fbbs = ast.literal_eval(c_side_data[3])
            false_bbs = []
            if fbbs:
                for bb in fbbs:
                    bb = np.asarray(bb, np.float64).flatten()
                    bb = self.convert_bb_to_pxv(bb)
                    false_bbs.append(bb)

            target_iou = self.calc_iou(gt_label_int, pre_label_int)
            intersection_over_union.append(target_iou)
            detected = False
            if target_iou > 0.5:
                detected = True
                confusion_values[0][0] += 1
                confusion_values[ambiguity_class][0] += 1
                outcome = "TP_"
                # self.visualize_pointings(gt_label_int, pre_label_int, target_iou, iteration, filename, color)
            else:
                confusion_values[0][2] += 1
                confusion_values[ambiguity_class][2] += 1
                outcome = "FN_"

            for bb in false_bbs:
                false_iou = self.calc_iou(bb, pre_label_int)
                if false_iou > .5:
                    detected = True
                    confusion_values[0][1] += 1
                    confusion_values[ambiguity_class][1] += 1
                    outcome = outcome + "FP_"

            if not detected:
                # add Miss
                confusion_values[0][3] += 1
                confusion_values[ambiguity_class][3] += 1

        return np.asarray(intersection_over_union), np.asarray(confusion_values)

    def evaluate_gwr_with_amb_detection(self, testing_data):
        test_data, test_labels, side_data = testing_data

        samples = test_data.shape[0]
        number_nodes = self.weights.shape[0]
        distances = np.zeros(number_nodes)
        activations = np.zeros(samples)

        intersection_over_union = []

        # List of lists: 0: overall, 1-4 respective ambiguity class
        # In list 0. TP, 1. FP, 2. FN, 3. misses, 4. correct_amb 5. false noise 6. total count
        confusion_values = [[0, 0, 0, 0, 0, 0, samples],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0]]

        for iteration in range(0, samples):
            observation = test_data[iteration]
            c_side_data = side_data[iteration]
            filename = c_side_data[0]
            ambiguity_class = int(filename.split("_")[0][-1])
            # label in this case => 0: red, 1: yellow, 2: green
            color = int(c_side_data[1])
            confusion_values[ambiguity_class][6] += 1
            corres_image = cv.imread("resources/current_training/bb_test/all_obj_permutations/" + filename + ".jpg")

            frame_ycrcb = cv.cvtColor(corres_image, cv.COLOR_BGR2YCrCb)
            skin_binary = gesture.apply_skin_hist2d(frame_ycrcb, self.skin_prob_binary_crcb)
            frame, hand_positions_t0, tracking_state = state.detection_step(corres_image, skin_binary, [None, None],
                                                                            "None")
            # saves outcome for visualizations
            outcome = ""

            for i in range(0, number_nodes):
                distances[i] = self.metric_distance(self.weights[i], observation)

            # test_bbs: 1: prediction, target bb, if available other false positive bbs

            first_index = distances.argmin()
            first_distance = distances.min()
            activations[iteration] = math.exp(-first_distance)

            if activations[iteration] > self.aT:
                p_label = self.labels[first_index]

                # here activate neighborhood approach or take 4 best nodes
                # neighborhood_labels = self.check_neighborhood(first_index)
                x_best_labels = self.get_x_best_labels(distances, self.x_b_l_value)
                union_of_best = self.union_bounding_boxes(x_best_labels)

                bmu = self.convert_bb_to_pxv(p_label)

                ambiguity, bounding_boxes, b_frame = obj_d.check_for_ambiguity(corres_image, union_of_best,
                                                                               hand_positions_t0)

                # Here infer which of the detected object is the target and get target and false bbs
                # compare gt and detected positions
                iou_target = []
                if ambiguity:
                    # object_bb: 0: "object_color", 1: bb_wh, 2: bb])
                    detected = False
                    for d_object in bounding_boxes:
                        if d_object[2] == color:
                            iou_target.append(self.calc_iou(d_object[0], bmu))

                        iou = self.calc_iou(d_object[0], bmu)
                        if iou > .5:
                            detected = True
                            if d_object[2] == color:
                                # add TP
                                confusion_values[0][0] += 1
                                confusion_values[ambiguity_class][0] += 1
                                intersection_over_union.append(iou)
                                outcome = "TP_"
                            else:
                                # add FP
                                confusion_values[0][1] += 1
                                confusion_values[ambiguity_class][1] += 1
                                # add FN
                                confusion_values[0][2] += 1
                                confusion_values[ambiguity_class][2] += 1
                                outcome = "FP_"
                                outcome += "FN_"
                                if len(iou_target) > 0:
                                    intersection_over_union.append(iou_target[0])
                                # self.visualize_pointings(d_object[0], bmu, iou, outcome, filename,
                                #                          color, d_object[2], iteration)

                    if not detected:
                        # add FN
                        confusion_values[0][2] += 1
                        confusion_values[ambiguity_class][2] += 1
                        # add Miss
                        confusion_values[0][3] += 1
                        confusion_values[ambiguity_class][3] += 1
                        outcome = "FN_Miss"
                        # self.visualize_pointings(bounding_boxes, bmu, 0.0, outcome, filename,
                        #                          color, None, iteration)

                    # add correct amb
                    if ambiguity_class > 2:
                        confusion_values[0][4] += 1
                        confusion_values[ambiguity_class][4] += 1

                else:
                    if len(bounding_boxes) > 0:
                        d_object = bounding_boxes[0]
                        if d_object[2] == color:
                            iou_target.append(self.calc_iou(d_object[0], bmu))
                            confusion_values[0][0] += 1
                            confusion_values[ambiguity_class][0] += 1
                            intersection_over_union.append(self.calc_iou(d_object[0], bmu))
                            outcome = "TP_"
                            # self.visualize_pointings(d_object[0], union_of_best, 0.0, outcome, filename,
                            #                          color, d_object[2], iteration)

                        else:
                            # add FP
                            confusion_values[0][1] += 1
                            confusion_values[ambiguity_class][1] += 1
                            # add FN
                            confusion_values[0][2] += 1
                            confusion_values[ambiguity_class][2] += 1
                            outcome = "FP_FN_"

                            # self.visualize_pointings(d_object[0], union_of_best, 0.0, outcome, filename,
                            #                          color, d_object[2], iteration)

                    else:
                        # add FN
                        confusion_values[0][2] += 1
                        confusion_values[ambiguity_class][2] += 1
                        # add Miss
                        confusion_values[0][3] += 1
                        confusion_values[ambiguity_class][3] += 1
                        outcome = "FN_Miss"
                        # self.visualize_pointings(bounding_boxes, union_of_best, 0.0, outcome, filename,
                        #                          color, None, iteration)

                    # add correct amb
                    if ambiguity_class < 3:
                        confusion_values[0][4] += 1
                        confusion_values[ambiguity_class][4] += 1

            else:
                # add false amb detec
                confusion_values[0][5] += 1
                confusion_values[ambiguity_class][5] += 1

        return np.asarray(intersection_over_union), np.asarray(confusion_values)

    def visualize_pointings(self, gt, prediction, iou, outcome, filename, t_color, p_color, iteration):
        test_image = cv.imread("resources/current_training/bb_test/all_obj_permutations/" + filename + ".jpg")

        if p_color is not None:
            cv.rectangle(test_image, (int(gt[0]), int(gt[1])),
                         (int(gt[2]), int(gt[3])),
                         (0, 0, 255), 2)
            cv.rectangle(test_image, (int(prediction[0]), int(prediction[1])),
                         (int(prediction[2]), int(prediction[3])),
                         (0, 255, 255), 2)
        else:
            cv.rectangle(test_image, (int(prediction[0]), int(prediction[1])),
                         (int(prediction[2]), int(prediction[3])),
                         (0, 255, 255), 2)
            for bb in gt:
                cv.rectangle(test_image, (int(bb[0][0]), int(bb[0][1])),
                             (int(bb[0][2]), int(bb[0][3])),
                             (255, 255, 255), 2)

        cv.putText(test_image, str(iou) + "T: " + str(t_color) + " P: " + str(p_color), (100, 100),
                   cv.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 255], 1)

        cv.imwrite("resources/current_training/bb_test/gwr_test/" + str(iteration) + str(outcome) + filename + ".jpg",
                   test_image)

    # find a way to visualize the 2D structure of the network
    def visualize_gwr(self):
        pass

    def get_no_edges(self):
        no_edges = 0

        u_edges = np.triu(self.edges)
        for i in range(0, u_edges.shape[0]):
            neighbors = np.nonzero(u_edges[i])
            no_edges += len(neighbors[0])

        return no_edges

    def visualize_error(self, title):
        plt.plot(self.error)
        plt.title(title, fontsize=24)
        plt.ylabel('AQE', fontsize=18)
        plt.show()


if __name__ == "__main__":
    # link_data_gwr = "resources/current_training/gwr_data/12_07_hand_shape/gwr/live_test_gwr"
    link_data_gwr = "resources/current_training/gwr_data/12_07_hand_shape/trained_gwr/classic_normalized/cross_90/0"
    link_data_td = "resources/current_training/gwr_data/12_07_hand_shape/noise_filtered/normalized/training_test"
    # link_data_td = "resources/current_training/gwr_data/12_07_hand_shape/noise_filtered/training_test"

    frame_shape = (950, 700)
    gwr = GWRInterface(link_data_gwr, frame_shape)

    test_data = np.load(link_data_td + "/test_data.npy")
    test_labels = np.load(link_data_td + "/test_labels.npy")
    side_data = list(csv.reader(open(link_data_td + "/side_data_test.csv", "rU"), delimiter=','))

    # test_data = gwr.normalize_with_fixed_min_max(test_data, frame_shape)

    # ious, confusion_values = gwr.quantitatively_evaluate_gwr((test_data, test_labels, side_data))
    ious, confusion_values = gwr.evaluate_gwr_with_amb_detection((test_data, test_labels, side_data))
