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

t = np.array([row[1] for row in trainingSet])

# II.1.1 Maximum likelihood solution

# Construct design matrices

# Linear model y
def y(x,w):
	# Check dimensions
	if len(x) != len(w) - 1:
		raise Exception("Wrong dimensionality of parameters (requires w-1 = x)")
	result = w[0]
	for i in range(0,len(x)):
		result += np.dot(w[i+1], x[i])
	return result

# Selection 1 (columns 4, 7, 8, 9) arranged as rows
design1 = np.matrix([[row[3], row[6], row[7], row[8]] for row in trainingSet])
# Selection 2 (column 8 transposed)
design2 = np.matrix([row[7] for row in trainingSet])

# Compute ML estimate (training)
w_ml_sel1 = t * np.linalg.pinv(design1).T # Why am I doing this? Alignment error if not transposed.
w_ml_sel2 = t * np.linalg.pinv(design2)

# Apply each model to the test set
print len(testSet)
print (w_ml_sel1)
#linmod1 = y(testSet, w_ml_sel1)
