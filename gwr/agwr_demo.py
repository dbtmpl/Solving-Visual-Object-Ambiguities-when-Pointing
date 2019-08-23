# -*- coding: utf-8 -*-
"""
Demo with Associative GWR

@last-modified: 3 July 2018

@author: German I. Parisi (german.parisi@gmail.com)

Please cite this paper: Parisi, G.I., Weber, C., Wermter, S. (2015) Self-Organizing Neural Integration of Pose-Motion Features for Human Action Recognition. Frontiers in Neurorobotics, 9(3).
"""

import csv
from agwr_class import AssociativeGWR
import numpy as np
import matplotlib.pyplot as plt
import cPickle

# Main ########################################################################

def normalize_data(data):
    size = data.shape
    # Data normalization
    oDataSet = np.copy(data)
    for i in range(0, size[1]):
        print(i)
        maxColumn = max(data[:, i])
        minColumn = min(data[:, i])
        for j in range(0, size[0]):
            oDataSet[j, i] = (data[j, i] - minColumn) / (maxColumn - minColumn)

    return oDataSet


def normalize_with_fixed_min_max(data_set, dimensions):
    n_data_set = np.copy(data_set)
    size = data_set.shape
    min_max_values = [(0, 180), (0, dimensions[1]), (0, dimensions[0]), (0, dimensions[1]), (0, dimensions[0])]
    for i in range(0, size[1]):
        print(i)
        maxColumn = min_max_values[i][1]
        minColumn = min_max_values[i][0]
        for j in range(0, size[0]):
            n_data_set[j, i] = (data_set[j, i] - minColumn) / (maxColumn - minColumn)

    return n_data_set


if __name__ == "__main__":
    # Set working path
    # os.getcwd()
    dataFlag = 1  # Import dataset from file
    importFlag = 0  # Import saved network
    trainFlag = 1  # Train AGWR with imported dataset
    saveFlag = 1  # Save trained network to file
    testFlag = 0  # Compute classification accuracy
    plotFlag = 0  # Plot 2D map

    if (dataFlag):
        # Load data set
        link_data_td = "../resources/current_training/gwr_data/12_07_hand_shape/noise_filtered/normalized/training_test"
        # link_data_td = "../Thesis_deictics_with_NICO/resources/current_training/gwr_data/12_07_hand_shape/noise_filtered/normalized_for_live/training_test"
        # link_data_gwr = "../Thesis_deictics_with_NICO/resources/current_training/gwr_data/12_07_hand_shape/trained_gwr/classic_normalized/act_T_90"
        # link_data_gwr = "../Thesis_deictics_with_NICO/resources/current_training/gwr_data/12_07_hand_shape/trained_gwr/for_live_normalized/act_T_90"
        dataSet = np.load(link_data_td + "/training_data.npy")
        print(dataSet.dtype)
        print(dataSet)
        print(dataSet.shape)
        labelSet = np.load(link_data_td + "/training_labels.npy")
        size = dataSet.shape

        # Pre-process samples anXd labels
        dimension = 5

        oDataSet = dataSet

        # oDataSet = normalize_data(dataSet)
        # oDataSet = normalize_with_fixed_min_max(dataSet, (950, 700))
        # print(oDataSet)

    # if (importFlag):
    #     file = open("myAGWR" + '.network', 'r')
    #     dataPickle = file.read()
    #     file.close()
    #     myAGWR = AssociativeGWR()
    #     myAGWR.__dict__ = cPickle.loads(dataPickle)

    if (trainFlag):
        initNeurons = 1  # Weight initialization (0: random, 1: sequential)
        numberOfEpochs = 30  # Number of training epochs
        insertionThreshold = 0.90  # Activation threshold for node insertion
        learningRateBMU = 0.1  # Learning rate of the best-matching unit (BMU)
        learningRateNeighbors = 0.01  # Learning rate of the BMU's topological neighbors

        myAGWR = AssociativeGWR(oDataSet, labelSet, initNeurons, numberOfEpochs, insertionThreshold, learningRateBMU,
                                learningRateNeighbors)
        weights, edges, labels, error_count = myAGWR.trainAGWR(oDataSet, labelSet)

        np.save(link_data_gwr + "/weights", np.float64(weights))
        np.save(link_data_gwr + "/edges", np.float64(edges))
        np.save(link_data_gwr + "/labels", np.float64(labels))
        np.save(link_data_gwr + "/error_count", np.float64(error_count))

    if (saveFlag):
        file = open(link_data_gwr + "/pickle_save" + '.network', 'w')
        file.write(cPickle.dumps(myAGWR.__dict__))
        file.close()

    if (testFlag):
        bmus, blabels, activations = myAGWR.predictAGWR(oDataSet, myAGWR.weights, myAGWR.alabels)

        print "Test accuracy: " + str(myAGWR.computeAccuracy(labelSet, blabels))

    if (plotFlag):
        # Plot network
        # This just plots the first two dimensions of the weight vectors.
        # For better visualization, PCA over weight vectors must be performed.
        classLabels = 1
        ccc = ['black', 'blue', 'red']  # 'green','yellow','cyan','magenta','0.75','0.15','1'
        fig = plt.figure()
        for ni in range(len(myAGWR.weights)):
            plindex = np.argmax(myAGWR.alabels[ni])
            if (classLabels):
                plt.scatter(myAGWR.weights[ni, 0], myAGWR.weights[ni, 1], color=ccc[plindex])
            else:
                plt.scatter(myAGWR.weights[ni, 0], myAGWR.weights[ni, 1])
            for nj in range(len(myAGWR.weights)):
                if (myAGWR.edges[ni, nj] > 0):
                    plt.plot([myAGWR.weights[ni, 0], myAGWR.weights[nj, 0]],
                             [myAGWR.weights[ni, 1], myAGWR.weights[nj, 1]], 'gray', alpha=.3)
        plt.show()
