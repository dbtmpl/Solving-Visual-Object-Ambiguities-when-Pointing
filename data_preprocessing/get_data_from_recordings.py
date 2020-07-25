import ast
import csv
import glob

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from engine.calculate_hand_features import GWR_Hand_Features
from utils.general_utils import get_crop_heuristics, crop_and_resize


class RecordGWRData(object):
    def __init__(self):
        skin_model_path = "resources/skin_color_data/histograms/skin_probabilities_crcb.npy"
        self.gwr_hf = GWR_Hand_Features(skin_model_path)

        self.data = self.load_video_data()
        self.data_labels = self.load_labels()
        self.crops = self.get_crops()

    @staticmethod
    def load_video_data():
        # import video files with glob in same order
        one_object = [cv.VideoCapture(pointings_one) for pointings_one in
                      sorted(glob.glob("resources/current_training/video_data/one object/*.webm"))]
        two_objects = [cv.VideoCapture(pointings_two) for pointings_two in
                       sorted(glob.glob("resources/current_training/video_data/two objects/*/*.webm"))]
        three_objects = [cv.VideoCapture(pointings_three) for pointings_three in
                         sorted(glob.glob("resources/current_training/video_data/three objects/*/*.webm"))]

        data = one_object + two_objects + three_objects

        return data[:50]

    @staticmethod
    def load_labels():
        reader = csv.reader(open("resources/current_training/labeled_data.csv", "rU"), delimiter=',')
        data_labels = list(reader)[2:]
        for row in data_labels:
            row[1] = ast.literal_eval(row[1])
            row[2] = ast.literal_eval(row[2])
            row[3] = ast.literal_eval(row[3])

        return data_labels[:50]

    def get_crops(self):
        cap = self.data[0]
        _, frame_t1 = cap.read()
        return get_crop_heuristics(frame_t1)

    def get_x_frames(self):
        save_link = "resources/current_training/bb_test/all_obj_permutations/"
        for i, cap in enumerate(self.data):
            row = self.data_labels[i]
            filename = row[0]
            cap.set(cv.CAP_PROP_POS_FRAMES, 40)
            ret, frame = cap.read()
            if ret:
                frame = crop_and_resize(frame, self.crops)
                cv.imwrite(save_link + filename + ".jpg", frame)

    def check_data(self):
        # for each video take first frame, draw the corresponding bounding boxes and save picutures in folder
        # to check if bounding boxes are correct
        save_link = "resources/current_training/bb_test/test_data_labelling/"
        for i, cap in enumerate(self.data):
            row = self.data_labels[i]
            bbs = row[1:]
            for j, bb in enumerate(bbs):
                if bb is not None:
                    box = bb[0]
                    start_pointing = bb[1]
                    end_pointing = bb[2]

                    cap.set(cv.CAP_PROP_POS_FRAMES, start_pointing - 1)
                    ret, frame = cap.read()

                    frame = crop_and_resize(frame, self.crops)

                    if j == 0:
                        cv.rectangle(frame, box[0], box[1], (0, 0, 255), 3)
                        cv.putText(frame, "RED", (20, 100), cv.FONT_HERSHEY_SIMPLEX, 3.0, (255, 0, 0),
                                   lineType=cv.LINE_AA)
                    elif j == 1:
                        cv.rectangle(frame, box[0], box[1], (0, 255, 255), 3)
                        cv.putText(frame, "YELLOW", (20, 100), cv.FONT_HERSHEY_SIMPLEX, 3.0, (255, 0, 0),
                                   lineType=cv.LINE_AA)
                    else:
                        cv.rectangle(frame, box[0], box[1], (0, 255, 0), 3)
                        cv.putText(frame, "GREEN", (20, 100), cv.FONT_HERSHEY_SIMPLEX, 3.0, (255, 0, 0),
                                   lineType=cv.LINE_AA)
                    cv.imwrite(save_link + row[0] + "_" + str(j) + "_start.jpg", frame)

                    cap.set(cv.CAP_PROP_POS_FRAMES, end_pointing - 1)
                    ret, frame = cap.read()

                    frame = frame[0:self.crops[0], self.crops[1]:self.crops[2]]
                    frame = cv.resize(frame, ((self.crops[2] - self.crops[1]) / 2, self.crops[0] / 2))

                    if j == 0:
                        cv.rectangle(frame, box[0], box[1], (0, 0, 255), 3)
                        cv.putText(frame, "RED", (20, 100), cv.FONT_HERSHEY_SIMPLEX, 3.0, (255, 0, 0),
                                   lineType=cv.LINE_AA)
                    elif j == 1:
                        cv.rectangle(frame, box[0], box[1], (0, 255, 255), 3)
                        cv.putText(frame, "YELLOW", (20, 100), cv.FONT_HERSHEY_SIMPLEX, 3.0, (255, 0, 0),
                                   lineType=cv.LINE_AA)
                    else:
                        cv.rectangle(frame, box[0], box[1], (0, 255, 0), 3)
                        cv.putText(frame, "GREEN", (20, 100), cv.FONT_HERSHEY_SIMPLEX, 3.0, (255, 0, 0),
                                   lineType=cv.LINE_AA)
                    cv.imwrite(save_link + row[0] + "_" + str(j) + "_end.jpg", frame)

    def get_false_bbs(self, pointings, target_bb):
        false_bbs = []
        for pointing in pointings:
            if pointing is not None:
                if not np.array_equal(np.asarray(pointing[0], np.float64).flatten(), target_bb):
                    false_bbs.append(pointing[0])
        return false_bbs

    def convert_bb_to_percentages(self, pointings):
        for pointing in pointings:
            if pointing is not None:
                bb = pointing[0]
                bb[0] = (bb[0][0] / 700.0, bb[0][1] / 950.0)
                bb[1] = (bb[1][0] / 700.0, bb[1][1] / 950.0)

        return pointings

    def create_cs_pointing_data(self, link_to_files):
        cs_pointing_data = np.zeros((0, 4), dtype=np.float64)
        cs_pointing_labels = np.zeros((0, 4), dtype=np.float64)
        side_data = []

        for i, cap in enumerate(self.data):
            row = self.data_labels[i]
            pointings = self.convert_bb_to_percentages(row[1:])
            filename = row[0]
            for j, pointing in enumerate(pointings):
                if pointing is not None:
                    target_bb = np.asarray(pointing[0], np.float64).flatten()
                    false_bbs = self.get_false_bbs(pointings, target_bb)
                    start_pointing = pointing[1] + 2
                    end_pointing = pointing[2] - 1
                    cap.set(cv.CAP_PROP_POS_FRAMES, start_pointing)
                    while start_pointing <= end_pointing:
                        ret, frame = cap.read()
                        if ret:
                            frame = crop_and_resize(frame, self.crops)

                            # capture gwr feature vector
                            fingertip, p3 = self.gwr_hf.get_cs_pointing_data(frame)

                            if fingertip is not None:
                                observation = np.asarray([[fingertip[0], fingertip[1], p3[0], p3[1]]])
                                cs_pointing_data = np.append(cs_pointing_data, observation, axis=0)

                                label = np.asarray([target_bb], np.float64)
                                cs_pointing_labels = np.append(cs_pointing_labels, label, axis=0)

                                side_data.append([filename, j, start_pointing, false_bbs])

                                # save file name for next training and color of object pointed at?

                            start_pointing += 1

        np.save(link_to_files + "/data2", np.float64(cs_pointing_data))
        np.save(link_to_files + "/labels2", np.float64(cs_pointing_labels))

        with open(link_to_files + "/side_data2.csv", "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerows(side_data)

        print("success")

    def create_gwr_data(self, link_to_files):
        gwr_features = np.zeros((0, 5), dtype=np.float64)
        gwr_labels = np.zeros((0, 4), dtype=np.float64)
        side_data = []

        for i, cap in enumerate(self.data):
            row = self.data_labels[i]
            pointings = self.convert_bb_to_percentages(row[1:])
            filename = row[0]
            for j, pointing in enumerate(pointings):
                if pointing is not None:
                    last_five_angles = []
                    target_bb = np.asarray(pointing[0], np.float64).flatten()
                    false_bbs = self.get_false_bbs(pointings, target_bb)
                    start_pointing = pointing[1] + 2
                    end_pointing = pointing[2] - 1
                    cap.set(cv.CAP_PROP_POS_FRAMES, start_pointing)
                    while start_pointing <= end_pointing:
                        ret, frame = cap.read()
                        if ret:
                            frame = crop_and_resize(frame, self.crops)

                            # capture gwr feature vector
                            angle, icx, icy, f, last_five_angles = self.gwr_hf.get_gwr_data(frame, last_five_angles)

                            if angle is not None:
                                observation = np.asarray([[angle, icx, icy, f[0], f[1]]])

                                gwr_features = np.append(gwr_features, observation, axis=0)

                                label = np.asarray([target_bb], np.float64)
                                gwr_labels = np.append(gwr_labels, label, axis=0)

                                side_data.append([filename, j, start_pointing, false_bbs])

                                # save file name for next training and color of object pointed at?

                            start_pointing += 1

        np.save(link_to_files + "/data2", np.float64(gwr_features))
        np.save(link_to_files + "/labels2", np.float64(gwr_labels))

        with open(link_to_files + "/side_data2.csv", "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerows(side_data)
        #
        print("success")

    def concat_data(self, link_to_files, data1, data2, labels1, labels2, side_data1, side_data2):

        data = np.append(data1, data2, axis=0)
        labels = np.append(labels1, labels2, axis=0)

        np.save(link_to_files + "/data", np.float64(data))
        np.save(link_to_files + "/labels", np.float64(labels))

        side_data = side_data1 + side_data2

        with open(link_to_files + "/side_data.csv", "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerows(side_data)

    def split_training_test_data(self, data_set, labels, side_data):
        idx = np.random.permutation(data_set.shape[0])
        side_data_np = np.asarray(side_data)
        d, l, fc = data_set[idx], labels[idx], side_data_np[idx]

        training_data = d[:4914]
        training_labels = l[:4914]
        fc_training = fc[:4914]
        training = training_data, training_labels, fc_training

        test_data = d[4914:]
        test_labels = l[4914:]
        fc_test = fc[4914:]
        test = test_data, test_labels, fc_test

        return training, test

    def filter_noise_z_score(self, data, labels, side_data, threshold, dim):

        mean_y = np.mean(data[:, dim])
        stdev_y = np.std(data[:, dim])
        z_scores = [(observation - mean_y) / stdev_y for observation in data[:, dim]]
        x = np.where(np.abs(z_scores) < threshold)
        d = data[x[0]]
        la = labels[x[0]]
        sd = side_data[x[0]]

        return d, la, sd

    def filter_noise(self, data, labels, side_data, threshold, dim, operator):
        if operator:
            x = np.where(data[:, dim] < threshold)
            tr = data[x[0]]
            la = labels[x[0]]
            sd = side_data[x[0]]
        else:
            x = np.where(data[:, dim] > threshold)
            tr = data[x[0]]
            la = labels[x[0]]
            sd = side_data[x[0]]
        return tr, la, sd

    def normalize_data(self, data):
        size = data.shape
        # Data normalization
        oDataSet = np.copy(data)
        for i in range(0, size[1]):
            maxColumn = max(data[:, i])
            minColumn = min(data[:, i])
            for j in range(0, size[0]):
                oDataSet[j, i] = (data[j, i] - minColumn) / (maxColumn - minColumn)

        return oDataSet

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

    def visualize_data(self, data_set, link_data, name):
        x, y, z, a, b = data_set[:, 0], data_set[:, 1], data_set[:, 2], data_set[:, 3], data_set[:, 4]

        # colors = {0: 'b', 1: 'y', 2: 'r'}

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax = fig.add_subplot(111)

        f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5)
        # f, (ax1, ax2, ax3, ax4) = plt.subplots(4)

        ax1.set_title('Pointing Angle')
        # ax1.set_title('Fingertip Position X')
        ax1.plot(x, np.zeros_like(x) + 0, 'o')
        ax1.yaxis.set_visible(False)

        ax2.set_title('Hand Centroid X')
        # ax2.set_title('Fingertip Position Y')
        ax2.plot(y, np.zeros_like(y) + 0, 'o')
        ax2.yaxis.set_visible(False)

        ax3.set_title('Hand Centroid Y')
        # ax3.set_title("$\psi_3$ Position X")
        ax3.plot(z, np.zeros_like(z) + 0, 'o')
        ax3.yaxis.set_visible(False)

        ax4.set_title('Fingertip Position X')
        # ax4.set_title("$\psi_3$ Position Y")
        ax4.plot(a, np.zeros_like(a) + 0, 'o')
        ax4.yaxis.set_visible(False)

        ax5.set_title('Fingertip Position Y')
        ax5.plot(b, np.zeros_like(b) + 0, 'o')
        ax5.yaxis.set_visible(False)

        # plt.savefig(link_data + "/visualizations/" + name + ".png")
        plt.show()


if __name__ == "__main__":
    setup = RecordGWRData()
    setup.get_x_frames()

    link_data = "resources/cs_pointing_data/15_07_fingertip_p3"
    link_data2 = "resources/current_training/gwr_data/12_07_hand_shape"
    link_data_td = "resources/current_training/gwr_data/12_07_hand_shape/training_test"
    link_data_gwr = "resources/current_training/gwr_data/12_07_hand_shape/gwr"

    data1 = np.load(link_data + "/raw_data/data.npy")
    data2 = np.load(link_data + "/noise_filtered/data.npy")
    data3 = np.load(link_data2 + "/data.npy")
    data4 = np.load(link_data2 + "/noise_filtered/data.npy")
    labels = np.load(link_data + "/noise_filtered/labels.npy")
    side_data = list(csv.reader(open(link_data + "/noise_filtered/side_data.csv", "rU"), delimiter=','))
    # setup.normalize_data(data)
