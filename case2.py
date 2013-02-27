#!/usr/bin/python2
# -*- coding: UTF-8 -*-
# Statistical Methods for Machine Learning
# Case 2 source code
# Authors: Asbjørn Thegler, Andreas Bock, Joachim Vig
#
from __future__ import division
import math
from pylab import *
import numpy as np
import mpl_toolkits.mplot3d.axes3d as plot3d
from PIL import Image
import scipy.io

## Processing the data

# Matrix from bodyfat.mat
rawData = scipy.io.loadmat('Data/bodyfat.mat')['data']

# Take 80% of rawData as training set and 20% as test set
division = math.ceil(len(np.matrix(rawData))*0.8)
trainingSet = np.random.permutation(rawData)[:division]
testSet 	= np.random.permutation(rawData)[division:]

t = np.matrix([row[1] for row in trainingSet]).T

## II.1.1 Maximum likelihood solution

# Linear model y
def y(x,w):
	# Check dimensions
	if len(x) != len(w) - 1:
		raise Exception("Wrong dimensionality of parameters (requires w-1 = x)")
	result = w[0]
	for i in range(0,len(x)):
		result += np.dot(w[i+1], x[i])
	return result

# Construct design matrices
# Selection 1 (columns 4, 7, 8, 9) arranged as rows
design1 = np.matrix([[row[3], row[6], row[7], row[8]] for row in trainingSet])
# Selection 2 (column 8 transposed)
design2 = np.matrix(trainingSet.T[7]).T
print design2

# Compute ML estimate (training)
w_ml_sel1 = np.linalg.pinv(design1)*t # Contains 4 ML estimates
w_ml_sel2 = np.linalg.pinv(design2)*t # Contains a single ML estimate

# Extract the right test set data for each ML estimate
testSet_asCols = testSet.T
testSet_sel1 = np.matrix([testSet_asCols[3] ,testSet_asCols[6], testSet_asCols[7], testSet_asCols[8]])
testSet_sel2 = np.matrix(testSet_asCols[7])

# Apply each model to the test set 
#linmod1 = y(testSet_sel1, w_ml_sel1)  # Wrong?
#linmod2 = y(testSet_sel2, w_ml_sel2)

def rms (t,x,w):
	N = len(x)
	result = 0
	for i in range(N):
		result += (t[i] - y(x[i],w))**2
	return math.sqrt(result/N)

#rms_sel1 = rms(t, testSet_sel1, w_ml_sel1)
#rms_sel2 = rms(t, testSet_sel2, w_ml_sel2)

## II.1.2 Maximum a posteriori solution

## II.2.1 Linear discriminant analysis





