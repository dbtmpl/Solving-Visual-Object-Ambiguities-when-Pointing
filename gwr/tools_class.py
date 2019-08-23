# -*- coding: utf-8 -*-
"""
Auxiliary tools for the Gamma-GWR

@last-modified: 23 May 2018

@author: German I. Parisi (german.parisi@gmail.com)

Please cite this paper: Parisi, G.I., Tani, J., Weber, C., Wermter, S. (2017) Lifelong Learning of Human Actions with Deep Neural Network Self-Organization. Neural Networks 96:137-149.
"""

import csv
import numpy as np
import numpy
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt_activations
import matplotlib.pyplot as plt_qee
#from numba import jit

class MyTools:
    
    # Returns list, number of samples, and dimensionality #####################
    def loadDataSet( self, fileName, fMode, delim ):
        reader = csv.reader(open(fileName,fMode),delimiter=delim)
        x = list(reader)
        result = numpy.array(x).astype('float')
        size = result.shape
        return result, size[0], size[1]
        
    # Write to CSV file #######################################################
    def writeToCsv(self, itera, fileName):
        with open(fileName, 'wb') as f:
            writer = csv.writer(f)
            writer.writerows(itera)
        
    # Normalize ###############################################################
    def normalizeData ( self, dataSet ): 
        oDataSet = np.copy(dataSet)
        samples = dataSet.shape[0]
        dimension = dataSet.shape[1]
        for i in range(0, dimension):
            maxColumn = max(dataSet[:,i])
            minColumn = min(dataSet[:,i])
            for j in range(0, samples):
                oDataSet[j,i] = ( dataSet[j,i] - minColumn ) / ( maxColumn - minColumn )
                
        return oDataSet
            
    # PCA #####################################################################
    def PCA(self,data, dims_rescaled_data=1):
        """
        returns: data transformed in 2 dims/columns + regenerated original data
        pass in: data as 2D NumPy array
        """
        from scipy import linalg as LA
        m, n = data.shape
        # mean center the data
        data -= data.mean(axis=0)
        # calculate the covariance matrix
        R = np.cov(data, rowvar=False)
        # calculate eigenvectors & eigenvalues of the covariance matrix
        # use 'eigh' rather than 'eig' since R is symmetric, 
        # the performance gain is substantial
        evals, evecs = LA.eigh(R)
        # sort eigenvalue in decreasing order
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:,idx]
        # sort eigenvectors according to same index
        evals = evals[idx]
        # select the first n eigenvectors (n is desired dimension
        # of rescaled data array, or dims_rescaled_data)
        evecs = evecs[:, :dims_rescaled_data]
        # carry out the transformation on the data using eigenvectors
        # and return the re-scaled data, eigenvalues, and eigenvectors
        return np.dot(evecs.T, data.T).T, evals, evecs
        
    def ComputePCA(self,data):    
        data_resc, data_orig, eigenvectors = self.PCA(data, dims_rescaled_data=2)
        m,n = data_resc.shape
        data_recovered = np.dot(eigenvectors, m).T
        data_recovered += data_recovered.mean(axis=0)
        return data_resc
        #plot(data_resc[:, 0], '.')
        
    def ComputePCAN(self,data,dimen):    
        data_resc, data_orig, eigenvectors = self.PCA(data, dims_rescaled_data=dimen)
        m,n = data_resc.shape
        data_recovered = np.dot(eigenvectors, m).T
        data_recovered += data_recovered.mean(axis=0)
        return data_resc
        #plot(data_resc[:, 0], '.')
            
    # 2D Plot #################################################################     
    def networkActiPlot(self, activations):
        plt_activations.figure()
        plt_activations.plot(activations,label="Line 2", linewidth=1)
        plt.axis([0, len(activations), 0, 1])
        plt_activations.show()
    
    # 1D Plot #################################################################
    def networkQEPlot(self, qee):
        plt_qee.figure()
        plt_qee.plot(qee,label="Line 2", linewidth=1)
        plt_qee.show()