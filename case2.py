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

t_train = np.matrix([row[1] for row in trainingSet]).T
t_test  = np.matrix([row[1] for row in testSet]).T

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
#print design2

# Compute ML estimate (training)
w_ml_sel1 = np.linalg.pinv(design1)*t_train
w_ml_sel2 = np.linalg.pinv(design2)*t_train

# Extract the right test set data for each ML estimate (x)
testSet_asCols = testSet.T
x_testSet_sel1 = np.matrix([testSet_asCols[3] ,testSet_asCols[6], testSet_asCols[7], testSet_asCols[8]])
x_testSet_sel2 = np.matrix(testSet_asCols[7])

# Root Mean Square

def rms (t,x,w):
	N = len(t)
	result = 0
	for i in range(N):
		tn = t[i][0,0]
		y  = x.T[i] * w
		result += (tn - y)**2
	return math.sqrt(result/N)

# RMS sel2
#rms_sel2 = 0
#for i in range(len(t_test)):
#	tn = t_test[i][0,0]
#	y  = x_testSet_sel2.T[i] * w_ml_sel2[0,0]
#	rms_sel2 += (tn - y)**2
#rms_sel2 = math.sqrt(rms_sel2/len(t_test))

rms_sel2 = rms(t_test, x_testSet_sel2, w_ml_sel2)
print rms_sel2

# RMS sel1
#rms_sel1 = 0
#for i in range(len(t_test)):
#	tn = t_test[i][0,0]
#	y  = x_testSet_sel1.T[i] * w_ml_sel1
#	rms_sel1 += (tn - y)**2
#rms_sel1 = math.sqrt(rms_sel1/len(t_test))

rms_sel1 = rms(t_test, x_testSet_sel1, w_ml_sel1)
print rms_sel1

## II.1.2 Maximum a posteriori solution

## II.2.1 Linear discriminant analysis

