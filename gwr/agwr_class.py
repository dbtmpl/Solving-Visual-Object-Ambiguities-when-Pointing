# -*- coding: utf-8 -*-
"""
Associative GWR based on (Marsland et al. 2002)'s Grow-When-Required

@last-modified: 3 July 2018

@author: German I. Parisi (german.parisi@gmail.com)

Please cite this paper: Parisi, G.I., Weber, C., Wermter, S. (2015) Self-Organizing Neural Integration of Pose-Motion Features for Human Action Recognition. Frontiers in Neurorobotics, 9(3).
"""

import scipy.spatial
import numpy as np
import math


class AssociativeGWR:
    def __init__(self, dataSet, labelSet, initMethod, mE, iT, eeB, eeN):
        self.samples, self.dimension = dataSet.shape
        self.numNodes = 2
        self.dimension = dataSet.shape[1]
        self.weights = np.zeros((self.numNodes, self.dimension))
        self.edges = np.ones((self.numNodes, self.numNodes))
        self.ages = np.zeros((self.numNodes, self.numNodes))
        self.habn = np.ones(self.numNodes)
        self.node_labels = np.zeros((0, 4), dtype=np.float64)

        self.maxEpochs = mE
        self.insertionThreshold = iT
        self.epsilon_b = eeB
        self.epsilon_n = eeN

        self.distanceMetric = 1
        self.habThreshold = 0.1
        self.tau_b = 0.3
        self.tau_n = 0.1
        self.maxNodes = self.samples  # OK for batch, bad for incremental
        self.maxNeighbours = 6
        self.maxAge = 200
        self.newNodeValue = 0.5
        self.aIncreaseFactor = 1
        self.aDecreaseFactor = 0.1

        if (initMethod):
            self.weights[0] = dataSet[0]
            self.weights[1] = dataSet[1]
            self.node_labels = np.append(self.node_labels, np.asarray([labelSet[0]]), axis=0)
            self.node_labels = np.append(self.node_labels, np.asarray([labelSet[1]]), axis=0)

        else:
            randomIndex = np.random.randint(0, dataSet.shape[0], 2)
            self.weights[0] = dataSet[randomIndex[0]]
            self.weights[1] = dataSet[randomIndex[1]]
            self.node_labels = np.append(self.node_labels, labelSet[randomIndex[0]], axis=0)
            self.node_labels = np.append(self.node_labels, labelSet[randomIndex[1]], axis=0)

    def compute_new_label_pos(self, bb1, bb2, index, new_node, epsilon):
        # returns centroid and w, h of the bb
        bb1_data = self.calc_bb_centroid(bb1)
        bb2_data = self.calc_bb_centroid(bb2)

        if new_node:
            # calc_average
            new_centroid = np.array([np.dot(np.asarray(bb1_data[0]) + np.asarray(bb2_data[0]), 0.5)]).flatten()

        else:
            # incorporate learning rate
            delta_centroid = np.dot((np.asarray(bb1_data[0]) - np.asarray(bb2_data[0])), epsilon) * self.habn[index]
            new_centroid = np.asarray(bb2_data[0]) + delta_centroid

        # mean of width and height of two input bbs
        w = (bb1_data[1] + bb2_data[1]) / 2
        h = (bb1_data[2] + bb2_data[2]) / 2

        new_bb = [new_centroid[0] - (w / 2), new_centroid[1] - (h / 2),
                  new_centroid[0] + (w / 2), new_centroid[1] + (h / 2)]

        if new_node:
            return np.asarray([new_bb], dtype=np.float64)
        else:
            return np.asarray(new_bb, dtype=np.float64)

    def calc_bb_centroid(self, bb):
        t1 = (bb[0], bb[1])
        t2 = (bb[2], bb[3])
        width, heigth = tuple(np.subtract(t2, t1))
        centroid = t1[0] + (width / 2), t1[1] + (heigth / 2)

        return centroid, width, heigth

    def computeDistance(self, x, y, m):
        if m:
            return np.linalg.norm(x - y)  # Euclidean distance - np.sqrt(np.sum((x-y)**2))
        else:
            return scipy.spatial.distance.cosine(x, y)

    def habituateNeuron(self, index, tau):
        self.habn[index] += (tau * 1.05 * (1. - self.habn[index]) - tau)

    def updateNeuralWeight(self, input, index, epsilon):
        delta = np.array([np.dot((input - self.weights[index]), epsilon)]) * self.habn[index]
        self.weights[index] = self.weights[index] + delta

    def updateEdges(self, fi, si):
        neighboursFirst = np.nonzero(self.edges[fi])
        if (len(neighboursFirst[0]) >= self.maxNeighbours):
            remIndex = -1
            maxAgeNeighbour = 0
            for u in range(0, len(neighboursFirst[0])):
                if (self.ages[fi, neighboursFirst[0][u]] > maxAgeNeighbour):
                    maxAgeNeighbour = self.ages[fi, neighboursFirst[0][u]]
                    remIndex = neighboursFirst[0][u]
            self.edges[fi, remIndex] = 0
            self.edges[remIndex, fi] = 0
        self.edges[fi, si] = 1

    def removeOldEdges(self):
        for i in range(0, self.numNodes):
            neighbours = np.nonzero(self.edges[i])
            for j in range(0, len(neighbours[0])):
                if (self.ages[i, j] >= self.maxAge):
                    self.edges[i, j] = 0
                    self.edges[j, i] = 0

    def age_neurons(self, bmu_index):
        neighbor_indices = np.nonzero(self.edges[bmu_index])
        for neighbor in neighbor_indices[0]:
            self.ages[bmu_index, neighbor] += 1

    def removeIsolatedNeurons(self):
        indCount = 0
        while (indCount < self.numNodes):
            neighbours = np.nonzero(self.edges[indCount])
            if (len(neighbours[0]) < 1):
                self.weights = np.delete(self.weights, indCount, axis=0)
                self.node_labels = np.delete(self.node_labels, indCount, axis=0)
                self.edges = np.delete(self.edges, indCount, axis=0)
                self.edges = np.delete(self.edges, indCount, axis=1)
                self.ages = np.delete(self.ages, indCount, axis=0)
                self.ages = np.delete(self.ages, indCount, axis=1)
                self.habn = np.delete(self.habn, indCount)
                self.numNodes = self.weights.shape[0]
                print "(-- " + str(indCount) + ")"
            else:
                indCount += 1

    def trainAGWR(self, dataSet, labelSet):

        # Start training
        epochs = 0
        errorCounter = np.zeros(self.maxEpochs)
        while (epochs < self.maxEpochs):
            epochs += 1
            print ("(Epoch: " + str(epochs) + " )"),
            for iteration in range(0, self.samples):
                # Generate input sample
                input = dataSet[iteration]
                label = labelSet[iteration]

                # Find the best and second-best matching neurons
                distances = np.zeros(self.numNodes)
                for i in range(0, self.numNodes):
                    distances[i] = self.computeDistance(self.weights[i], input, self.distanceMetric)
                sort_index = np.argsort(distances)
                firstIndex = sort_index[0]
                firstDistance = distances[firstIndex]
                secondIndex = sort_index[1]
                errorCounter[epochs - 1] += firstDistance

                # Compute network activity
                a = math.exp(-firstDistance)

                if ((a < self.insertionThreshold) and (self.habn[firstIndex] < self.habThreshold) and (
                        self.numNodes < self.maxNodes)):
                    # Add new neuron
                    newWeight = np.array([np.dot(self.weights[firstIndex] + input, self.newNodeValue)])
                    self.weights = np.concatenate((self.weights, newWeight), axis=0)

                    new_label = self.compute_new_label_pos(label, self.node_labels[firstIndex], firstIndex, True, None)

                    self.node_labels = np.append(self.node_labels, new_label, axis=0)

                    newIndex = self.numNodes
                    self.numNodes += 1
                    self.habn.resize(self.numNodes)
                    self.habn[newIndex] = 1

                    # Update edges
                    self.edges.resize((self.numNodes, self.numNodes))
                    self.edges[firstIndex, secondIndex] = 0
                    self.edges[secondIndex, firstIndex] = 0
                    self.edges[firstIndex, newIndex] = 1
                    self.edges[newIndex, firstIndex] = 1
                    self.edges[newIndex, secondIndex] = 1
                    self.edges[secondIndex, newIndex] = 1

                    # Update ages
                    self.ages.resize((self.numNodes, self.numNodes))
                    self.age_neurons(firstIndex)
                    self.ages[firstIndex, newIndex] = 0
                    self.ages[newIndex, firstIndex] = 0
                    self.ages[newIndex, secondIndex] = 0
                    self.ages[secondIndex, newIndex] = 0

                    print ("(++ " + str(self.numNodes) + ')'),
                else:
                    # Adapt weights
                    self.updateNeuralWeight(input, firstIndex, self.epsilon_b)

                    new_label = self.compute_new_label_pos(label, self.node_labels[firstIndex], firstIndex, False,
                                                           self.epsilon_b)
                    self.node_labels[firstIndex] = new_label

                    # Habituate BMU
                    self.habituateNeuron(firstIndex, self.tau_b)

                    # Update ages
                    self.age_neurons(firstIndex)
                    self.ages[firstIndex, secondIndex] = 0
                    self.ages[secondIndex, firstIndex] = 0

                    # Update edges // Remove oldest ones
                    self.updateEdges(firstIndex, secondIndex)
                    self.updateEdges(secondIndex, firstIndex)

                    # Update topological neighbours
                    neighboursFirst = np.nonzero(self.edges[firstIndex])
                    for z in range(0, len(neighboursFirst[0])):
                        neIndex = neighboursFirst[0][z]
                        self.updateNeuralWeight(input, neIndex, self.epsilon_n)
                        self.habituateNeuron(neIndex, self.tau_n)
                        new_label = self.compute_new_label_pos(label, self.node_labels[neIndex], firstIndex, False,
                                                               self.epsilon_n)
                        self.node_labels[neIndex] = new_label

            # Remove old edges
            self.removeOldEdges()

            # Compute some metrics
            errorCounter[epochs - 1] /= self.samples
            print "AQE: " + str(errorCounter[epochs - 1])

            # Remove isolated neurons
            self.removeIsolatedNeurons()

        print ("Network size: " + str(self.numNodes))

        return self.weights, self.edges, self.node_labels, errorCounter

    # Test GWR ################################################################

    def predictAGWR(self, dataSet, weights, alabels):
        print ("Testing...")
        samples = dataSet.shape[0]
        bmus = -np.ones(samples)
        blabels = -np.ones(samples)
        nNodes = len(weights)
        distance = np.zeros(nNodes)
        activations = np.zeros(samples)

        # Iterate over the neurons to find BMUs
        for iterat in range(0, samples):
            input = dataSet[iterat]
            for i in range(0, nNodes):
                distance[i] = self.computeDistance(weights[i], input, 1)
            firstIndex = distance.argmin()
            firstDistance = distance.min()
            activations[iterat] = math.exp(-firstDistance)
            bmus[iterat] = firstIndex
            blabels[iterat] = np.argmax(alabels[firstIndex])

        return bmus, blabels, activations

    def computeAccuracy(self, labelSet, blabels):
        goodCounter = 0
        for iterat in range(0, len(labelSet)):
            if (labelSet[iterat] == blabels[iterat]):
                goodCounter += 1
        accuracyRate = 100 * goodCounter / len(labelSet)
        return accuracyRate
