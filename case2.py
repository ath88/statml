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

def dims(x):
	print "Dimensions", len(x), "x", len(x.T)

# Construct design matrices
design1 = np.matrix([[row[3], row[6], row[7], row[8]] for row in trainingSet]).T
design2 = np.matrix(trainingSet.T[7])

# Add padding
padding = np.ones((division,1))
design1 = np.reshape(np.append(padding, design1), (5,202)).T
design2 = np.reshape(np.append(padding, design2), (2,202)).T

# Compute ML estimate (training)
w_ml_sel1 = np.linalg.pinv(design1)*t_train
w_ml_sel2 = np.linalg.pinv(design2)*t_train
#print "w_ml_sel1:", w_ml_sel1, "\n"
#print "w_ml_sel2:", w_ml_sel2, "\n"

# Convert from matrix to array type
w_ml_sel1 = np.squeeze(np.asarray(w_ml_sel1))
w_ml_sel2 = np.squeeze(np.asarray(w_ml_sel2))

# Extract the right test set data for each ML estimate (x'es)
testSet_asCols = testSet.T
x_testSet_sel1 = np.matrix([testSet_asCols[3] ,testSet_asCols[6], testSet_asCols[7], testSet_asCols[8]])
x_testSet_sel2 = np.matrix(testSet_asCols[7])

# Add a [1..1] column to the test set
x_testSet_sel1 = np.reshape(np.append(padding, x_testSet_sel1), (5,202)).T
x_testSet_sel2 = np.reshape(np.append(padding, x_testSet_sel2), (2,202)).T

# Root Mean Square
def rms (t,x,w):
	N = len(t)
	result = 0
	#print dims(t)
	for i in range(N):
		tn = t[i][0,0]
		y = np.dot(w, x[i])
		result += (tn - y)**2
	return math.sqrt(result/N)

rms_sel1 = rms(t_test, x_testSet_sel1, w_ml_sel1)
rms_sel2 = rms(t_test, x_testSet_sel2, w_ml_sel2)

print "RMS for ML 1:  ", rms_sel1
print "RMS for ML 2:  ", rms_sel2

## II.1.2 Maximum a posteriori solution

## Estimate m_N and S_N (training)
beta  = 1 # fixed throughout assignment

# BEGIN WORKING

## MAP for alpha = 1
alpha = 1
# For selection 1
S_N_1 = alpha*np.identity(5)+beta*np.dot(design1.T,design1)
m_N_1 = beta*np.dot(np.linalg.inv(S_N_1), design1.T)*t_train # MAP estimate

# For selection 2
S_N_2 = alpha*np.identity(2)+beta*np.dot(design2.T,design2)
m_N_2 = beta*np.dot(np.linalg.inv(S_N_2), design2.T)*t_train # MAP estimate

# Convert from matrix to array type
m_N_1 = np.squeeze(np.asarray(m_N_1))
m_N_2 = np.squeeze(np.asarray(m_N_2))

rms_MAP_sel1 = rms(t_test, x_testSet_sel1, m_N_1)
rms_MAP_sel2 = rms(t_test, x_testSet_sel2, m_N_2)

print "RMS for MAP 1 (a="+str(alpha)+"): ", rms_MAP_sel1
print "RMS for MAP 2 (a="+str(alpha)+"): ", rms_MAP_sel2

# END WORKING


## II.2.1 Linear discriminant analysis

rawTrainingData = np.loadtxt('Iris/irisTrain.dt').T
length = rawTrainingData[0]
width = rawTrainingData[1]
classes = rawTrainingData[2]
inData = list(map(list,zip(length,width,classes)))

#print width
#print inData

# Scatter plot
#figure()
#title('Scatter plot of training data.')
#xlabel('Length')
#ylabel('Width')
#scatter(length,width,c=classes)
#show()

## Linear Discriminant Analysis
no_classes = 3

## Compute mean estimates for each class
# Get x values (in this case length of the sepals)
c0s = [i[0] for i in inData if i[2] == 0.0]
c1s = [i[0] for i in inData if i[2] == 1.0]
c2s = [i[0] for i in inData if i[2] == 2.0]

mean_est_c0 = sum(c0s)/len(c0s)
mean_est_c1 = sum(c1s)/len(c1s)
mean_est_c2 = sum(c2s)/len(c2s)

## Compute common covariance matrix

## II.2.2 Nearest neighbour with Euclidean metric

def dist(x,y,M):
 	# Euclidean metric if M = identity matrix
	xm = x*M
	ym = y*M
	return np.sqrt(np.sum((np.array(xm-ym))**2))

def closestClass (S_star, k):
	classes = [c for [x,y,c] in S_star]
	maxCount = float("-inf")
	for i in np.random.permutation(classes):
		if maxCount < classes.count(i):
			maxCount = i
	return maxCount

def knn (k, S, pnt, M):
	pnt = np.array(pnt)
	S_star = []
	while len(S_star) < k:
		closest = S[0]
		for i in S[1:]:
			if dist(pnt, i[:2], M) < dist(pnt, closest[:2], M):
				closest = i
		S_star.append(closest)
		S.remove(closest)
	# Decide which class is argmax
	return closestClass(S_star,k)
M = np.matrix([[1,0],[0,10]])

#kek = knn(4, testData0, (0,0))
#kek = knn(3, testData, (3,4))

kek = knn (1, inData, (6,0.3), M)
#print kek
kek = knn (3, inData, (6,0.3), M)
#print kek
kek = knn (5, inData, (6,0.3), M)
#print kek
kek = knn (7, inData, (6,0.3), M)
#print kek





