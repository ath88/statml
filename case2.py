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

## Helper functions 

def multi_norm (x, sigma, mu):
	const = 1.0/(((2*np.pi)**(len(mu.T)/2))*np.sqrt(np.linalg.det(sigma)))
	part1 = 1/((2*np.pi)**(len(mu)/2))
	part2 = 1/(np.linalg.det(sigma)**0.5)
	x_mu = np.matrix((x-mu)).T
	precision = np.matrix(sigma).I
	return const*np.exp(-0.5*dot(x_mu.T, dot(precision, x_mu)))

## Processing the data

# Matrix from bodyfat.mat
rawData = scipy.io.loadmat('Data/bodyfat.mat')['data']
# MEAN OF THE SECOND COLUMN
#print sum(rawData.T[1])/len(rawData.T[1])

# Take 80% of rawData as training set and 20% as test set
division = math.ceil(len(np.matrix(rawData))*0.8)
# Permute data
randomRawData = np.random.permutation(rawData)
trainingSet = randomRawData[:division]
testSet	    = randomRawData[:division]
# Target variable t as columns
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
design1 = np.matrix([[row[3], row[6], row[7], row[8]] for row in trainingSet]).T
# Selection 2 (column 8 transposed)
design2 = np.matrix(trainingSet.T[7])

# Compute ML estimate (training)

w_ml_sel1 = np.linalg.pinv(design1).T*t_train # Why are we transposin'?
w_ml_sel2 = np.linalg.pinv(design2).T*t_train
#print "w_ml_sel1:", w_ml_sel1, "\n"
#print "w_ml_sel2:", w_ml_sel2, "\n"

# Extract the right test set data for each ML estimate (x'es)
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
		result += (tn - y[0,0])**2
	return math.sqrt(result/N)

#for i in range(len(t_test)):
#	tn = t_test[i][0,0]
#	y  = x_testSet_sel1.T[i] * w_ml_sel1
#	rms_sel1 += (tn - y)**2
#rms_sel1 = math.sqrt(rms_sel1/len(t_test))

rms_sel1 = rms(t_test, x_testSet_sel1, w_ml_sel1)
#print "RMS for ML 1:  ", rms_sel1

# RMS for selection 2
#rms_sel2 = 0
#for i in range(len(t_test)):
#	tn = t_test[i][0,0]
#	y  = x_testSet_sel2.T[i] * w_ml_sel2[0,0]
#	rms_sel2 += (tn - y)**2
#rms_sel2 = math.sqrt(rms_sel2/len(t_test))

rms_sel2 = rms(t_test, x_testSet_sel2, w_ml_sel2)
#print "RMS for ML 2:  ", rms_sel2

## II.1.2 Maximum a posteriori solution
## Estimate m_N and S_N (training)

beta  = 1

## alpha = 0.001
alpha = 0.001
# For selection 1
S_N_1 = alpha*np.identity(4)+beta*design1*design1.T
m_N_1 = beta*np.linalg.inv(S_N_1)*design1*t_train # MAP estimate

# For selection 2
S_N_2 = alpha+beta*design2*design2.T
m_N_2 = beta*np.linalg.inv(S_N_2)*design2*t_train # MAP estimate

rms_MAP_sel1 = rms(t_test, x_testSet_sel1, m_N_1)
rms_MAP_sel2 = rms(t_test, x_testSet_sel2, m_N_2)
#print "RMS for MAP 1 (a="+str(alpha)+"): ", rms_MAP_sel1
#print "RMS for MAP 2 (a="+str(alpha)+"): ", rms_MAP_sel2

## alpha = 1
alpha = 1
# For selection 1
S_N_1 = alpha*np.identity(4)+beta*design1*design1.T
m_N_1 = beta*np.linalg.inv(S_N_1)*design1*t_train # MAP estimate

# For selection 2
S_N_2 = alpha+beta*design2*design2.T
m_N_2 = beta*np.linalg.inv(S_N_2)*design2*t_train # MAP estimate

rms_MAP_sel1 = rms(t_test, x_testSet_sel1, m_N_1)
rms_MAP_sel2 = rms(t_test, x_testSet_sel2, m_N_2)
#print "RMS for MAP 1 (a="+str(alpha)+"): ", rms_MAP_sel1
#print "RMS for MAP 2 (a="+str(alpha)+"): ", rms_MAP_sel2

## alpha = 10
alpha = 10
# For selection 1
S_N_1 = alpha*np.identity(4)+beta*design1*design1.T
m_N_1 = beta*np.linalg.inv(S_N_1)*design1*t_train # MAP estimate
# For selection 2
S_N_2 = alpha+beta*design2*design2.T
m_N_2 = beta*np.linalg.inv(S_N_2)*design2*t_train # MAP estimate

rms_MAP_sel1 = rms(t_test, x_testSet_sel1, m_N_1)
rms_MAP_sel2 = rms(t_test, x_testSet_sel2, m_N_2)
#print "RMS for MAP 1 (a="+str(alpha)+"): ", rms_MAP_sel1
#print "RMS for MAP 2 (a="+str(alpha)+"): ", rms_MAP_sel2

## alpha = 1000
alpha = 1000
# For selection 1
S_N_1 = alpha*np.identity(4)+beta*design1*design1.T
m_N_1 = beta*np.linalg.inv(S_N_1)*design1*t_train # MAP estimate
# For selection 2
S_N_2 = alpha+beta*design2*design2.T
m_N_2 = beta*np.linalg.inv(S_N_2)*design2*t_train # MAP estimate

rms_MAP_sel1 = rms(t_test, x_testSet_sel1, m_N_1)
rms_MAP_sel2 = rms(t_test, x_testSet_sel2, m_N_2)
#print "RMS for MAP 1 (a="+str(alpha)+"): ", rms_MAP_sel1
#print "RMS for MAP 2 (a="+str(alpha)+"): ", rms_MAP_sel2

## II.2.1 Linear discriminant analysis

rawTrainingData = np.loadtxt('Iris/irisTrain.dt').T
length = rawTrainingData[0]
width = rawTrainingData[1]
classes = rawTrainingData[2]
inData = zip(length,width,classes)

#print width
#print inData

# Scatter plot
figure()
title('Scatter plot of training data.')
xlabel('Length')
ylabel('Width')
scatter(length,width,c=classes)
show()

# LDA

## II.2.2 Nearest neighbour with Euclidean metric
def dist(x,y):   
	return numpy.sqrt(numpy.sum((x[:2]-y[:2])**2))

def knn (neighbours, S, pnt):
	S_star = []
	while len(S_star) < neighbours:
		# Find neighbour
		closest = (float("inf"), float("inf"), float("inf"))
		for i in S:
			if dist(pnt, i) < closest:
				closest = i
		S_star.append(closest)
		S.remove(closest)
	## DECIDE WHICH ONE TO CHOOSE HERE
	return closest


