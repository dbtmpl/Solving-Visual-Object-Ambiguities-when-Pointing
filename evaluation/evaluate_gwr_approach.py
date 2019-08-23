import csv
import numpy as np
import cPickle

from sklearn.model_selection import KFold
from mechanics.gwr_interface import GWRInterface
from gwr_german.agwr_class import AssociativeGWR


def train_gwr(tdata, tlabels, insertion_thresh, save_link):
    initNeurons = 1  # Weight initialization (0: random, 1: sequential)
    numberOfEpochs = 50  # Number of training epochs
    learningRateBMU = 0.1  # Learning rate of the best-matching unit (BMU)
    learningRateNeighbors = 0.01  # Learning rate of the BMU's topological neighbors
    print(insertion_thresh)

    myAGWR = AssociativeGWR(tdata, tlabels, initNeurons, numberOfEpochs, insertion_thresh, learningRateBMU,
                            learningRateNeighbors)

    weights, edges, labels, error_count = myAGWR.trainAGWR(tdata, tlabels)

    np.save(save_link + "/weights", np.float64(weights))
    np.save(save_link + "/edges", np.float64(edges))
    np.save(save_link + "/labels", np.float64(labels))
    np.save(save_link + "/error_count", np.float64(error_count))

    gwr_pickle = open(save_link + "/pickle_save" + '.network', 'w')
    gwr_pickle.write(cPickle.dumps(myAGWR.__dict__))
    gwr_pickle.close()


# Trains and evaluates gwrs
def train_evaluate_gwr_k_fold(packed_data, hyper_params):
    data, labels, side_data = packed_data

    kf = KFold(n_splits=3, random_state=13, shuffle=True)
    for i in range(0, 3):
        print(i)
        q_ious_list = []
        q_confusion_values_list = []

        a_ious_list = []
        a_confusion_values_list = []

        for j, indexes in enumerate(kf.split(data)):
            print(j)
            train_index, test_index = indexes
            train_data = data[train_index]
            train_labels = labels[train_index]

            train_gwr(train_data, train_labels, hyper_params[0][i], hyper_params[1][i] + str(j))

            test_data = data[test_index]
            test_labels = labels[test_index]
            test_side_data = side_data[test_index]

            # we distinguish between evaluating the GWR with the annotated ground-truth bounding boxes (q) and with the
            # object detection including the ambiguity detection (a)
            gwr = GWRInterface(hyper_params[1][i] + str(j), (950, 700))
            q_ious, q_confusion_value = gwr.quantitatively_evaluate_gwr((test_data, test_labels, test_side_data))
            q_ious_list.append(q_ious)
            q_confusion_values_list.append(q_confusion_value)
            save_results(q_ious, q_confusion_value, hyper_params[1][i] + str(j) + "/_q")

            a_ious, a_confusion_value = gwr.evaluate_gwr_with_amb_detection((test_data, test_labels, test_side_data))
            print(a_confusion_value)
            a_ious_list.append(a_ious)
            a_confusion_values_list.append(a_confusion_value)
            save_results(a_ious, a_confusion_value, hyper_params[1][i] + str(j) + "/_a")

        q_iou_mean_kfold, q_confusion_values_mean_kfold, raw_values_q = calc_mean_of_k_fold(q_ious_list,
                                                                                            q_confusion_values_list,
                                                                                            False)

        a_iou_mean_kfold, a_confusion_values_mean_kfold, raw_values_a = calc_mean_of_k_fold(a_ious_list,
                                                                                            a_confusion_values_list,
                                                                                            True)

        save_overall(q_iou_mean_kfold, q_confusion_values_mean_kfold, raw_values_q, "/_q", hyper_params[1][i])
        save_overall(a_iou_mean_kfold, a_confusion_values_mean_kfold, raw_values_a, "/_a", hyper_params[1][i])

        # gwr.print_outcome(a_iou_mean_kfold, a_confusion_values_mean_k_fold)
        # gwr.print_outcome(a_ious, a_confusion_value)

        print(a_iou_mean_kfold)
        print(a_confusion_values_mean_kfold)


# Takes trained GWRs and evaluates them
def evaluate_trained_networks(packed_data, hyper_params):
    data, labels, side_data = packed_data

    kf = KFold(n_splits=3, random_state=13, shuffle=True)
    for i in range(0, 3):
        print(i)
        q_ious_list = []
        q_confusion_values_list = []

        a_ious_list = []
        a_confusion_values_list = []

        for j, indexes in enumerate(kf.split(data)):
            print(j)
            train_index, test_index = indexes

            test_data = data[test_index]
            test_labels = labels[test_index]
            test_side_data = side_data[test_index]

            # we distinguish between evaluating the GWR with the annotated ground-truth bounding boxes (q) and with the
            # object detection including the ambiguity detection (a)
            gwr = GWRInterface(hyper_params[1][i] + str(j), (950, 700))
            q_ious, q_confusion_value = gwr.quantitatively_evaluate_gwr((test_data, test_labels, test_side_data))
            save_results(q_ious, q_confusion_value, hyper_params[1][i] + str(j) + "/_q")

            q_ious_list.append(q_ious)
            q_confusion_values_list.append(q_confusion_value)

            a_ious, a_confusion_value = gwr.evaluate_gwr_with_amb_detection((test_data, test_labels, test_side_data))
            save_results(a_ious, a_confusion_value, hyper_params[1][i] + str(j) + "/_a")

            a_ious_list.append(a_ious)
            a_confusion_values_list.append(a_confusion_value)

        q_iou_mean_kfold, q_confusion_values_mean_kfold, raw_values_q = calc_mean_of_k_fold(q_ious_list,
                                                                                            q_confusion_values_list,
                                                                                            False)

        a_iou_mean_kfold, a_confusion_values_mean_kfold, raw_values_a = calc_mean_of_k_fold(a_ious_list,
                                                                                            a_confusion_values_list,
                                                                                            True)

        save_overall(q_iou_mean_kfold, q_confusion_values_mean_kfold, raw_values_q, "/_q", hyper_params[1][i])
        save_overall(a_iou_mean_kfold, a_confusion_values_mean_kfold, raw_values_a, "/_a", hyper_params[1][i])

        # gwr.print_outcome(a_iou_mean_kfold, a_confusion_values_mean_k_fold)
        # gwr.print_outcome(a_ious, a_confusion_value)

        print(a_iou_mean_kfold)
        print(a_confusion_values_mean_kfold)


# Calculates the mean of the cross validation folds. d_amb is a boolean to distinguish between cases (q vs. a).
def calc_mean_of_k_fold(ious, confusion_values_list, d_amb):
    # check saved data
    print(len(confusion_values_list))
    print(len(confusion_values_list[0]))

    means_iou = []
    for iou in ious:
        iou_mean = sum(iou) / len(iou)
        means_iou.append(iou_mean)
    overall_iou_mean = sum(means_iou) / len(means_iou)
    overall_iou_std = np.std(np.asarray(means_iou, np.float64))

    raw_values_overall = []
    if d_amb:
        # list of means of each fold
        # 0: overall, 1- 4 ambiguity classes
        # in sublists0: 0. Precision, 1, Recall, 2. F1-score, 3: Misses, 4: correct_amb 5. false n
        mean_confusion_lists = [[[], [], [], [], [], []],
                                [[], [], [], [], [], []],
                                [[], [], [], [], [], []],
                                [[], [], [], [], [], []],
                                [[], [], [], [], [], []]]

        for i, confusion_values in enumerate(confusion_values_list):
            print("A")
            print(confusion_values)
            for amb_c, amb_class in enumerate(confusion_values):
                count = amb_class[-1]
                tp_count, fp_count, fn_count, misses_count, correct_amb_count, f_noise = amb_class[:-1]

                raw_values = [i, amb_c, tp_count, fp_count, fn_count, misses_count, correct_amb_count, f_noise, count]
                raw_values_overall.append(raw_values)

                precision = float(tp_count) / (tp_count + fp_count)
                recall = float(tp_count) / (tp_count + fn_count)
                f1_score = 2 * ((precision * recall) / (precision + recall))
                misses = misses_count / float(count)
                correct_amb = correct_amb_count / float(count)
                false_noise = f_noise / float(count)

                mean_confusion_lists[amb_c][0].append(precision)
                mean_confusion_lists[amb_c][1].append(recall)
                mean_confusion_lists[amb_c][2].append(f1_score)
                mean_confusion_lists[amb_c][3].append(misses)
                mean_confusion_lists[amb_c][4].append(correct_amb)
                mean_confusion_lists[amb_c][5].append(false_noise)

        print(len(mean_confusion_lists[0][0]) == 3)

    else:
        # list of means of each fold
        # 0: overall, 1- 4 ambiguity classes
        # in sublists0: TP, 1: FP. 2: FN, 3. Precision, 4, Recall, 5. F1-score, 6: Misses
        mean_confusion_lists = [[[], [], [], []],
                                [[], [], [], []],
                                [[], [], [], []],
                                [[], [], [], []],
                                [[], [], [], []]]

        for i, confusion_values in enumerate(confusion_values_list):
            print("Q")
            print(confusion_values)
            for amb_c, amb_class in enumerate(confusion_values):
                print(len(amb_class) == 5)
                print(len(amb_class[:-1]) == 4)

                count = amb_class[-1]
                tp_count, fp_count, fn_count, misses_count = amb_class[:-1]
                raw_values = [i, amb_c, tp_count, fp_count, fn_count, misses_count, count]
                raw_values_overall.append(raw_values)

                precision = float(tp_count) / (tp_count + fp_count)
                recall = float(tp_count) / (tp_count + fn_count)
                f1_score = 2 * ((precision * recall) / (precision + recall))
                misses = misses_count / float(count)

                mean_confusion_lists[amb_c][0].append(precision)
                mean_confusion_lists[amb_c][1].append(recall)
                mean_confusion_lists[amb_c][2].append(f1_score)
                mean_confusion_lists[amb_c][3].append(misses)

    confusion_mean_of_folds = []

    for amb_class_f in mean_confusion_lists:
        save_list = []
        for i in range(0, len(amb_class_f)):
            print(len(amb_class_f[i]) == 3)
            mean = sum(amb_class_f[i]) / len(amb_class_f[i])
            std = np.std(np.asarray(amb_class_f[i], np.float64))
            save_list.append([mean, std])
        confusion_mean_of_folds.append(save_list)

    return [overall_iou_mean, overall_iou_std], confusion_mean_of_folds, raw_values_overall


# prints out latex tables
def load_and_print_results(hyper_params):
    q_a = [(0, "/_q"), (1, "/_a")]
    epochs = ["30 Epochs", "50 Epochs", "100 Epochs"]
    ats = ["$\\boldsymbol{a_T:} \\boldsymbol{0.85}$, ", "$\\boldsymbol{a_T:} \\boldsymbol{0.90}$, ",
           "$\\boldsymbol{a_T:} \\boldsymbol{0.95}$, "]

    q_a_text = ["\\textbf{Individual GWR Evaluation}", "\\textbf{Evaluate Prediction with GWR}"]
    table2_text = "Ambiguity and Noise Detection"
    label = ["-85-gwr-results", "-90-gwr-results", "-95-gwr-results"]

    at_info = [".85", ".90", ".95"]

    table_width = 0.75

    overall_tables = [["", "", ""], ["", "", ""]]
    overall_tables2 = ["", "", ""]
    for e in q_a:

        if e[0] == 0:
            for i in range(0, 3):
                table_act = "\\begin{center} \n \\resizebox{" + str(
                    table_width) + "\\textwidth}{!}{ \n" + "\\begin{tabular}{||c|c|c|c|c||} \n" + "\\hline \n" + "\\multicolumn{5}{|c|}{" + \
                            ats[i] + q_a_text[e[0]] + "} \\\\ \n \\hline"

                for j in range(0, 3):
                    link = hyper_params[j][1][i]
                    ious = np.load(link + "mean" + e[1] + "/ious.npy")
                    confusion_values_list = np.load(link + "mean" + e[1] + "/confusion_values.npy")
                    raw_values = np.load(link + "mean" + e[1] + "/raw_values.npy")
                    # print_outcome(ious, confusion_values_list, raw_values, e[1], epochs[j], at_info[i])
                    table, table2 = convert_to_latex(ious, confusion_values_list, e[0], epochs[j])
                    table_act += table

                table_act += "\\end{tabular}} \n \captionof{table}{This table shows the precision, recall, ...}\label{0" + \
                             label[i] + "} \n \end{center}"
                overall_tables[0][i] = table_act

        elif e[0] == 1:
            for i in range(0, 3):
                table_act = "\\begin{center} \n \\resizebox{" + str(
                    table_width) + "\\textwidth}{!}{ \n" + "\\begin{tabular}{||c|c|c|c|c||} \n" + "\\hline \n" + "\\multicolumn{5}{|c|}{" + \
                            ats[i] + q_a_text[e[0]] + "} \\\\ \n \\hline"

                table_act_2 = "\\begin{center} \n \\resizebox{" + "0.45" + "\\textwidth}{!}{ \n" + "\\begin{tabular}{||c|c|c|} \n" + "\\hline \n" + "\\multicolumn{3}{|c|}{" + \
                              ats[i] + table2_text + "} \\\\ \n \\hline"

                for j in range(0, 3):
                    link = hyper_params[j][1][i]
                    ious = np.load(link + "mean" + e[1] + "/ious.npy")
                    confusion_values_list = np.load(link + "mean" + e[1] + "/confusion_values.npy")
                    raw_values = np.load(link + "mean" + e[1] + "/raw_values.npy")
                    table, table2 = convert_to_latex(ious, confusion_values_list, e[0], epochs[j])
                    table_act += table
                    if len(table2) > 0:
                        table_act_2 += table2

                table_act += "\\end{tabular}} \n \captionof{table}{This table shows the precision, recall, ...}\label{1" + \
                             label[i] + "} \n \end{center}"
                overall_tables[1][i] = table_act

                table_act_2 += "\\end{tabular}} \n \captionof{table}{This table shows the precision, recall, ...}\label{1" + \
                               label[i] + "-am-n" + "} \n \end{center}"
                overall_tables2[i] = table_act_2

    print(overall_tables[0][0])
    print ""
    print(overall_tables[0][1])
    print ""
    print(overall_tables[0][2])
    print("")
    print(overall_tables[1][0])
    print ""
    print(overall_tables2[0])
    print ""
    print(overall_tables[1][1])
    print ""
    print(overall_tables2[1])
    print ""
    print(overall_tables[1][2])
    print ""
    print(overall_tables2[2])


def convert_to_latex(intersection_over_union, confusion_values, e, epoch):
    amb_c = ["$a_1$", "$a_2$", "$a_3$", "$a_4$"]

    iou = "$" + str(np.round(intersection_over_union[0], 3)) + " \pm " + str(
        np.round(intersection_over_union[1], 3)) + "$"

    if e == 0:

        table = "\\hline \n \\multicolumn{5}{|c|}{\\textit{" + epoch + "}, \\textit{IoU}:" + iou + "} \\\\ \n \hline \n"
        if epoch == "30 Epochs":
            table += "& \\textit{Precision (\%)} & \\textit{Recall (\%)} & \\textit{$F_1\\text{-}score$ (\%)} & \\textit{Misses (\%)} \\\\ \n \hline \n"

        for i, amb in enumerate(confusion_values[1:]):
            row = amb_c[i] + " & " + "$" + str(np.round(amb[0][0], 4)) + " \pm " + str(
                np.round(amb[0][1], 3)) + "$" + " & " + "$" + str(np.round(amb[1][0], 4)) + " \pm " + str(
                np.round(amb[1][1], 3)) + "$" + " & " + "$" + str(np.round(amb[2][0], 4)) + " \pm " + str(
                np.round(amb[2][1], 3)) + "$" + " & " + "$" + str(np.round(amb[3][0], 4)) + " \pm " + str(
                np.round(amb[3][1], 3)) + "$" + " \\\\ \n"

            table += row

        first_row = confusion_values[0]
        total = "\\textit{Total}" + " & " + "$" + str(np.round(first_row[0][0], 4)) + " \pm " + str(
            np.round(first_row[0][1], 3)) + "$" + " & " + "$" + str(np.round(first_row[1][0], 4)) + " \pm " + str(
            np.round(first_row[1][1], 3)) + "$" + " & " + "$" + str(np.round(first_row[2][0], 4)) + " \pm " + str(
            np.round(first_row[2][1], 3)) + "$" + " & " + "$" + str(np.round(first_row[3][0], 4)) + " \pm " + str(
            np.round(first_row[3][1], 3)) + "$" + " \\\\ \n \\hline \n"

        table += total

        return table, []

    if e == 1:

        table = "\\hline \n \\multicolumn{5}{|c|}{\\textit{" + epoch + "}, \\textit{IoU}:" + iou + "} \\\\ \n \hline \n"

        if epoch == "30 Epochs":
            table += "& \\textit{Precision (\%)} & \\textit{Recall (\%)} & \\textit{$F_1\\text{-}score$ (\%)} & \\textit{Misses (\%)} \\\\ \n \hline \n"

        for i, amb in enumerate(confusion_values[1:]):
            row = amb_c[i] + " & " + "$" + str(np.round(amb[0][0], 4)) + " \pm " + str(
                np.round(amb[0][1], 3)) + "$" + " & " + "$" + str(np.round(amb[1][0], 4)) + " \pm " + str(
                np.round(amb[1][1], 3)) + "$" + " & " + "$" + str(np.round(amb[2][0], 4)) + " \pm " + str(
                np.round(amb[2][1], 3)) + "$" + " & " + "$" + str(np.round(amb[3][0], 4)) + " \pm " + str(
                np.round(amb[3][1], 3)) + "$" + " \\\\ \n"
            table += row

        first_row = confusion_values[0]
        total = "\\textit{Total}" + " & " + "$" + str(np.round(first_row[0][0], 4)) + " \pm " + str(
            np.round(first_row[0][1], 3)) + "$" + " & " + "$" + str(np.round(first_row[1][0], 4)) + " \pm " + str(
            np.round(first_row[1][1], 3)) + "$" + " & " + "$" + str(np.round(first_row[2][0], 4)) + " \pm " + str(
            np.round(first_row[2][1], 3)) + "$" + " & " + "$" + str(np.round(first_row[3][0], 4)) + " \pm " + str(
            np.round(first_row[3][1], 3)) + "$" + " \\\\ \n \\hline \n"

        table += total

        table2 = "\\hline \n \\multicolumn{3}{|c|}{\\textit{" + epoch + "}} \\\\ \n \hline \n"
        if epoch == "30 Epochs":
            table2 += "& \\textit{CDA (\%)} & \\textit{FDN(\%)} \\\\ \n \hline \n"

        for i, amb in enumerate(confusion_values[1:]):
            row = amb_c[i] + " & " + "$" + str(np.round(amb[4][0], 4)) + " \pm " + str(
                np.round(amb[4][1], 3)) + "$" + " & " + "$" + str(np.round(amb[5][0], 4)) + " \pm " + str(
                np.round(amb[5][1], 3)) + "$" + " \\\\ \n"
            table2 += row

        first_row = confusion_values[0]
        total = "\\hline \n \\textit{Total}" + " & " + "$" + str(np.round(first_row[4][0], 4)) + " \pm " + str(
            np.round(first_row[4][1], 3)) + "$" + " & " + "$" + str(np.round(first_row[5][0], 4)) + " \pm " + str(
            np.round(first_row[5][1], 3)) + "$" + " \\\\ \n \\hline \n"

        table2 += total

        return table, table2


# old function for printing results while evaluating
def print_outcome(intersection_over_union, confusion_values, raw_values, e, epochs, at_info):
    # in sublists 0: TP, 1: FP. 2: FN, 3. Precision, 4, Recall, 5. F1-score, 6: Misses, (7. correct..)
    print("FINAL SCORE's of: " + e + " " + epochs + " " + at_info)

    print(intersection_over_union)

    print("OVERALL IOU:")
    print(intersection_over_union[0])
    print("std +- " + str(intersection_over_union[1]))
    print(" ")

    if e == "/_a":

        for i, results in enumerate(confusion_values):
            print("ambiguity class: " + str(i))

            print("TP")
            print(raw_values[i][2])
            print("")
            print("FP")
            print(raw_values[i][3])
            print("")
            print("FN")
            print(raw_values[i][4])
            print("")
            print("Misses")
            print(raw_values[i][5])

            print("Precision")
            print(results[0][0])
            print("std +- " + str(results[0][1]))
            print("")
            print("Recall")
            print(results[1][0])
            print("std +- " + str(results[1][1]))
            print("")
            print("F1_score")
            print(results[2][0])
            print("std +- " + str(results[2][1]))
            print("")
            print("Misses")
            print(results[3][0])
            print("std +- " + str(results[3][1]))
            print("")
            print("correct identified ambiguities")
            print(results[4][0])
            print("std +- " + str(results[4][1]))
            print(" ")
            print("falsely identified noise")
            print(results[5][0])
            print("std +- " + str(results[5][1]))
            print(" ")
            print(" ")

    else:

        for i, results in enumerate(confusion_values):
            print("ambiguity class: " + str(i))

            print("TP")
            print(raw_values[i][2])
            print("")
            print("FP")
            print(raw_values[i][3])
            print("")
            print("FN")
            print(raw_values[i][4])
            print("")
            print("Misses")
            print(raw_values[i][5])

            print("Precision")
            print(results[0][0])
            print("std +- " + str(results[0][1]))
            print("")
            print("Recall")
            print(results[1][0])
            print("std +- " + str(results[1][1]))
            print("")
            print("F1_score")
            print(results[2][0])
            print("std +- " + str(results[2][1]))
            print("")
            print("Misses")
            print(results[3][0])
            print("std +- " + str(results[3][1]))
            print(" ")
            print(" ")


# latex table for network details
def get_network_details(hyper_params):
    aTs = ["0.85", "0.90", "0.95"]
    epochs = ["30", "50", "100"]

    latex_table = ""
    for i in range(0, 3):
        m_row = "\hline \n \multirow{3}{3em}{" + aTs[i] + "}"
        for j in range(0, 3):
            gwr = GWRInterface(hyper_params[j][1][i] + str(0), (950, 700))
            m_row += "&" + epochs[j] + "&" + str(gwr.number_nodes) + " & " + str(gwr.number_edges) + " & " + str(
                np.round(gwr.error[-1], 4)) + " \\\\ \n"

            # if epochs[j] == "100":
            #     gwr.visualize_error("$a_T: $" + aTs[i])

            # title = r'$a_T$: ' + str(hyper_params[0][i]) + ', # Epochs: ' + str(epochs)
            # gwr.visualize_error(title)
        latex_table += m_row + "\hline"

    print(latex_table)


def save_results(ious, confusion_value, link):
    np.save(link + "/iou", np.float64(ious))
    np.save(link + "/confusion_value", np.float64(confusion_value))


def save_overall(ious_list, confusion_values_list, raw_values_overall, e_type, link):
    np.save(link + "mean" + e_type + "/ious", np.float64(ious_list))
    np.save(link + "mean" + e_type + "/confusion_values", np.float64(confusion_values_list))
    np.save(link + "mean" + e_type + "/raw_values", np.float64(raw_values_overall))


if __name__ == "__main__":
    link_data_td = "resources/current_training/gwr_data/12_07_hand_shape/noise_filtered"
    # l_gwr = "resources/current_training/gwr_data/12_07_hand_shape/trained_gwr/classic_normalized"
    l_gwr = "resources/current_training/gwr_data/12_07_hand_shape/trained_gwr/area_a_func"

    data = np.load(link_data_td + "/normalized/normalized_data.npy")
    labels = np.load(link_data_td + "/labels.npy")
    side_data = np.asarray(list(csv.reader(open(link_data_td + "/side_data.csv", "rU"), delimiter=',')))

    # # GWR hyperparams: 0: aT, 1: save_link
    # hyper_params = [[[.85, .90, .95], [l_gwr + "/cross_85_30e/", l_gwr + "/cross_90_30e/", l_gwr + "/cross_95_30e/"]],
    #                 [[.85, .90, .95], [l_gwr + "/cross_85_50e/", l_gwr + "/cross_90_50e/", l_gwr + "/cross_95_50e/"]],
    #                 [[.85, .90, .95],
    #                  [l_gwr + "/cross_85_100e/", l_gwr + "/cross_90_100e/", l_gwr + "/cross_95_100e/"]]]


    # 85/90/95 is the activation threshold, XXe is the number of epochs

    # GWR hyperparams: 0: aT, 1: save_link
    hyper_params = [[[.85, .90, .95], [l_gwr + "/cross_85_30e/", l_gwr + "/cross_90_30e/", l_gwr + "/cross_95_30e/"]],
                    [[.85, .90, .95], [l_gwr + "/cross_85_50e/", l_gwr + "/cross_90_50e/", l_gwr + "/cross_95_50e/"]],
                    [[.85, .90, .95],
                     [l_gwr + "/cross_85_100e/", l_gwr + "/cross_90_100e/", l_gwr + "/cross_95_100e/"]]]

    # for params in hyper_params:
    #     evaluate_trained_networks((data, labels, side_data), params)
    # train_evaluate_gwr_k_fold((data, labels, side_data), hyper_params2)

    load_and_print_results(hyper_params)
    # get_network_details(hyper_params)
