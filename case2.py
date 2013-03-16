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
import copy

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

# Take 80% of rawData as training set and 20% as test set
division = math.ceil(len(np.matrix(rawData))*0.8)
# Permute data
randomRawData = np.random.permutation(rawData)
trainingSet = randomRawData[:division]
testSet	    = randomRawData[division:]
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
x_testSet_sel1 = np.reshape(np.append(np.ones((50,1)), x_testSet_sel1), (5,50)).T
x_testSet_sel2 = np.reshape(np.append(np.ones((50,1)), x_testSet_sel2), (2,50)).T

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

print "Maximum likelihood solution"
print "		- RMS for ML 1:  ", rms_sel1
print "		- RMS for ML 2:  ", rms_sel2

## II.1.2 Maximum a posteriori solution

## Estimate m_N and S_N (training)
#fixed throughout computations
beta  = 1
S_N_1_snd = beta*np.dot(design1.T,design1)
S_N_2_snd = beta*np.dot(design2.T,design2) 
all_rms1 = []
all_rms2 = []

## MAP for alpha = 0.001
alpha = 0.001
# For selection 1
S_N_1 = alpha*np.identity(5)+S_N_1_snd
m_N_1 = beta*np.dot(np.linalg.inv(S_N_1), design1.T)*t_train # MAP estimate
# For selection 2
S_N_2 = alpha*np.identity(2)+S_N_2_snd
m_N_2 = beta*np.dot(np.linalg.inv(S_N_2), design2.T)*t_train # MAP estimate
# Convert from matrix to array type
m_N_1 = np.squeeze(np.asarray(m_N_1))
m_N_2 = np.squeeze(np.asarray(m_N_2))
# Compute RMS
rms_MAP_sel1 = rms(t_test, x_testSet_sel1, m_N_1)
rms_MAP_sel2 = rms(t_test, x_testSet_sel2, m_N_2)
all_rms1.append(rms_MAP_sel1)
all_rms2.append(rms_MAP_sel2)

## MAP for alpha = 1
alpha = 1
# For selection 1
S_N_1 = alpha*np.identity(5)+S_N_1_snd
m_N_1 = beta*np.dot(np.linalg.inv(S_N_1), design1.T)*t_train # MAP estimate
# For selection 2
S_N_2 = alpha*np.identity(2)+S_N_2_snd
m_N_2 = beta*np.dot(np.linalg.inv(S_N_2), design2.T)*t_train # MAP estimate
# Convert from matrix to array type
m_N_1 = np.squeeze(np.asarray(m_N_1))
m_N_2 = np.squeeze(np.asarray(m_N_2))
# Compute RMS
rms_MAP_sel1 = rms(t_test, x_testSet_sel1, m_N_1)
rms_MAP_sel2 = rms(t_test, x_testSet_sel2, m_N_2)
all_rms1.append(rms_MAP_sel1)
all_rms2.append(rms_MAP_sel2)

## MAP for alpha = 10
alpha = 10
# For selection 1
S_N_1 = alpha*np.identity(5)+S_N_1_snd
m_N_1 = beta*np.dot(np.linalg.inv(S_N_1), design1.T)*t_train # MAP estimate
# For selection 2
S_N_2 = alpha*np.identity(2)+S_N_2_snd
m_N_2 = beta*np.dot(np.linalg.inv(S_N_2), design2.T)*t_train # MAP estimate
# Convert from matrix to array type
m_N_1 = np.squeeze(np.asarray(m_N_1))
m_N_2 = np.squeeze(np.asarray(m_N_2))
# Compute RMS
rms_MAP_sel1 = rms(t_test, x_testSet_sel1, m_N_1)
rms_MAP_sel2 = rms(t_test, x_testSet_sel2, m_N_2)
all_rms1.append(rms_MAP_sel1)
all_rms2.append(rms_MAP_sel2)

## MAP for alpha = 1000
alpha = 100
# For selection 1
S_N_1 = alpha*np.identity(5)+S_N_1_snd
m_N_1 = beta*np.dot(np.linalg.inv(S_N_1), design1.T)*t_train # MAP estimate
# For selection 2
S_N_2 = alpha*np.identity(2)+S_N_2_snd
m_N_2 = beta*np.dot(np.linalg.inv(S_N_2), design2.T)*t_train # MAP estimate
# Convert from matrix to array type
m_N_1 = np.squeeze(np.asarray(m_N_1))
m_N_2 = np.squeeze(np.asarray(m_N_2))
# Compute RMS
rms_MAP_sel1 = rms(t_test, x_testSet_sel1, m_N_1)
rms_MAP_sel2 = rms(t_test, x_testSet_sel2, m_N_2)
all_rms1.append(rms_MAP_sel1)
all_rms2.append(rms_MAP_sel2)

## Plot the RMS for different values of alpha
alphas = [0.001,  1, 10, 100]
figure()
subplot(223)
plot(alphas, all_rms1, c="green", label="Selection 1")
plot(alphas, all_rms2, c="red", label = "Selection 2")
legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
show()

print "Maximum a posteriori solution"
for i in range(len(alphas)):
	print "For alpha =",alphas[i],":"
	print "RMS MAP (Selection 1): ", all_rms1[i]
	print "RMS MAP (Selection 2): ", all_rms2[i]

## II.2.1 Linear discriminant analysis

rawTrainingData = np.loadtxt('Iris/irisTrain.dt').T
length = rawTrainingData[0]
width = rawTrainingData[1]
classes = rawTrainingData[2]
inData = list(map(list,zip(length,width,classes)))

rawTestData = np.loadtxt('Iris/irisTest.dt').T
test_length = rawTestData[0]
test_width = rawTestData[1]
test_classes = rawTestData[2]
test_inData = list(map(list,zip(test_length,test_width,test_classes)))

# Scatter plot
figure()
#title('Scatter plot of training data.')
xlabel('Length')
ylabel('Width')
scatter(length,width,c=classes)
show()

## Linear Discriminant Analysis
def discriminant (x,cls,cov,mu,prior):
	""" x: point to be classified 
	  cls: the class
	  cov: covariance of class
	   mu: mean of class
    prior: prior of class (estimate) """
	precision = np.linalg.inv(cov)
	return np.dot(np.dot(x.T,precision),mu)-0.5*np.dot(mu.T,np.dot(precision,mu))+math.log(prior)
def lda(x,c_cov,means):
	""" x: point to be classified 
	c_cov: covariance of classes
	means: list of (means, obs) for all classes
		n: total no. of observations
	"""
	no_classes = len(means)
	argmax = float("-inf")
	classification = None
	n = sum([i[1] for i in means]) # Needed to compute prior
	# Find argmax for discriminant
	for c in range(no_classes):
		mu 	  = means[c][0]
		prior = means[c][1]/n
		disc_k= discriminant(x,c,c_cov,mu,prior)
		if disc_k > argmax:
			argmax = disc_k
			classification = c
	return to_vect(classification, no_classes)
def to_vect(i,cls):
	v = np.zeros(cls)
	v[i] = 1.0
	return v

## Training of the model:
## Compute mean/covariance estimates for each class
c0s = [np.array(i[:2]) for i in inData if i[2] == 0.0]
c1s = [np.array(i[:2]) for i in inData if i[2] == 1.0]
c2s = [np.array(i[:2]) for i in inData if i[2] == 2.0]
len_c0s, len_c1s, len_c2s = len(c0s), len(c1s), len(c2s)
mean_est_c0 = np.sum(c0s, axis=0)/len_c0s
mean_est_c1 = np.sum(c1s, axis=0)/len_c1s
mean_est_c2 = np.sum(c2s, axis=0)/len_c2s
cov_c0 = np.cov(zip(*c0s))
cov_c1 = np.cov(zip(*c1s))
cov_c2 = np.cov(zip(*c2s))

## Compute common covariance matrix
l = len_c0s + len_c1s + len_c2s
no_classes = 3
class_cov = (cov_c0+cov_c1+cov_c2)/(l-no_classes)

means = [mean_est_c0, mean_est_c1, mean_est_c2]
meanobs = zip(means, [len_c0s, len_c1s, len_c2s])

print "Linear Discriminant Analysis"
print "  Training data:"
print "   - Class covariance:\n", class_cov
print "   - Means for the classes:\n"
for i in range(len(means)):
	print "         - Class number ", i, ": ", means[i]

# Training data
lda_training_errors = 0
for i in inData:
	new_class = lda(np.array(i[:2]), class_cov, meanobs)
	if not (new_class == to_vect(i[2],no_classes)).all():
		lda_training_errors += 1
print "  Training data errors: ", lda_training_errors

# Test data
lda_test_errors = 0
for i in test_inData:
	new_class = lda(np.array(i[:2]), class_cov, meanobs)
	if not (new_class == to_vect(i[2],no_classes)).all():
		lda_test_errors += 1
print "  Test data errors: ", lda_test_errors

## II.2.2 Nearest neighbour with Euclidean metric

def dist(x,y):
	return np.linalg.norm(x-y)

def closestClass (S_star, k):
	classes = [c for [x,y,c] in S_star]
	maxCount = float("-inf")
	for i in np.random.permutation(classes):
		if maxCount < classes.count(i):
			maxCount = i
	return maxCount

def knn (k, S, pnt):
	pnt = np.array(pnt)
	S_star = []
	while len(S_star) < k:
		closest = S[0]
		for i in S[1:]:
			if dist(pnt, i[:2]) < dist(pnt, closest[:2]):
				closest = i
		S_star.append(closest)
		S.remove(closest)
	# Decide which class is argmax
	return closestClass(S_star,k)

def knn_test(k, existingPoints, testData):
	errors = 0
	for i in testData:
		S_test = copy.deepcopy(existingPoints)
		new_class = knn(k, S_test, i[:2])
		if new_class != i[2]:
			errors += 1
	y.append(errors)
	print "    Errors for k=",k,": ", errors

y=[] # For plotting

## Accuracy on training set:
print "Nearest neighbour (Euclidean metric)"
print "Training set:"
knn_test(1,inData,inData)
knn_test(3,inData,inData)
knn_test(5,inData,inData)
knn_test(6,inData,inData)
knn_test(7,inData,inData)
knn_test(9,inData,inData)

y_train = copy.deepcopy(y)
y=[] # For plotting

## Accuracy on test set:
print "Nearest neighbour (Euclidean metric)"
print "Test set:"
knn_test(1,inData,test_inData)
knn_test(3,inData,test_inData)
knn_test(5,inData,test_inData)
knn_test(6,inData,test_inData)
knn_test(7,inData,test_inData)
knn_test(9,inData,test_inData)

# Plot
figure()
subplot(223)
plot([1,3,5,6,7,9], y_train, c="green", label="Training set")
plot([1,3,5,6,7,9], y, c="red", label="Test set")
legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
show()

## II.2.4 Nearest neighbour with non-standard metric

# Redefine metric
def dist(x,y):
	M = np.matrix([[1,0],[0,10]])
	xm = np.dot(M,x)
	ym = np.dot(M,y)
	return np.linalg.norm(xm-ym)

y=[] # For plotting

## Accuracy on training set:
print "Nearest neighbour (non-standard metric)"
print "Training set:"
knn_test(1,inData,inData)
knn_test(3,inData,inData)
knn_test(5,inData,inData)
knn_test(6,inData,inData)
knn_test(7,inData,inData)
knn_test(9,inData,inData)

y_train = copy.deepcopy(y)
y=[] # For plotting

## Accuracy on training set:
print "Nearest neighbour (non-standard metric)"
print "Test set:"
knn_test(1,inData,test_inData)
knn_test(3,inData,test_inData)
knn_test(5,inData,test_inData)
knn_test(6,inData,test_inData)
knn_test(7,inData,test_inData)
knn_test(9,inData,test_inData)

# Plot
figure()
subplot(223)
plot([1,3,5,6,7,9], y_train, c="green", label="Training set")
plot([1,3,5,6,7,9], y, c="red", label="Test set")
legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
show()

## II.2.5 LDA with rescaled data

# Transform data (same as multiplying width by 10)
inData_trans = list(map(list, zip(length,10*width,classes)))
test_inData_trans = list(map(list, zip(test_length,10*test_width,test_classes)))

## Training of the model:
## Compute mean/covariance estimates for each class
c0s_trans = [np.array(i[:2]) for i in inData_trans if i[2] == 0.0]
c1s_trans = [np.array(i[:2]) for i in inData_trans if i[2] == 1.0]
c2s_trans = [np.array(i[:2]) for i in inData_trans if i[2] == 2.0]
mean_est_c0_trans = np.sum(c0s_trans, axis=0)/len_c0s
mean_est_c1_trans = np.sum(c1s_trans, axis=0)/len_c1s
mean_est_c2_trans = np.sum(c2s_trans, axis=0)/len_c2s
cov_c0_trans = np.cov(zip(*c0s_trans))
cov_c1_trans = np.cov(zip(*c1s_trans))
cov_c2_trans = np.cov(zip(*c2s_trans))

## Compute common covariance matrix
class_cov_trans = (cov_c0_trans+cov_c1_trans+cov_c2_trans)/(l-no_classes)
means_trans = [mean_est_c0_trans, mean_est_c1_trans, mean_est_c2_trans]
meanobs_trans = zip(means_trans, [len_c0s, len_c1s, len_c2s])

print "Linear Discriminant Analysis [TRANSFORMED]"
print "  Training data:"
print "   - Class covariance:\n", class_cov_trans
print "   - Means for the classes:\n"
for i in range(len(means_trans)):
	print "         - Class number ", i, ": ", means_trans[i]

# Training data
lda_training_errors_trans = 0
for i in inData_trans:
	new_class = lda(np.array(i[:2]), class_cov_trans, meanobs_trans)
	if not (new_class == to_vect(i[2],no_classes)).all():
		lda_training_errors_trans += 1
print "  Training data errors: ", lda_training_errors_trans

# Test data
lda_test_errors_trans = 0
for i in test_inData_trans:
	new_class = lda(np.array(i[:2]), class_cov_trans, meanobs_trans)
	if not (new_class == to_vect(i[2],no_classes)).all():
		lda_test_errors_trans += 1
print "  Test data errors: ", lda_test_errors_trans
