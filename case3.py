#!/usr/bin/python2
# -*- coding: UTF-8 -*-
# Statistical Methods for Machine Learning
# Case 3 source code
# Authors: Asbjørn Thegler, Andreas Bock, Joachim Vig
from __future__ import division
import math
from pylab import *
import numpy as np
import scipy.io
from svm import * 

### III.1 Neural Networks
# III.1.1 Neural network implementation

def activation (a):
	return a/(1+abs(a))

def activationp (a):
	return 1/((1+abs(a))**2)

class NeuralNetwork ():
	"""
		A simple neural network class with one hidden layer.

		d: number of input neurons
		m: number of hidden neurons
		k: number of output neurons
		act: activation function
		actp: first derivative of activation function
	"""
	def __init__(self, d, m, k, act, actp):
		d += 1
		m += 1
		self.inn 	= d
		self.hidden = m
		self.out 	= k
		self.act    = act
		self.actp   = actp
		# Data structures for activation data and weights
		self.a_hd  = np.ones(m) 

		self.z_in  = np.ones(m)
		self.z_hd  = np.ones(m)    
		self.z_out = np.ones(k)    
		self.w_hd  = np.ones((m,d))
		self.w_out = np.ones((k,m))
		
		# Data structures for backpropagation
		self.delta_hd  = np.ones(m)
		self.delta_out = np.ones(k)
	
	def forwardPropagate (self, x):
		""" Forward propagate in neural network.
			x: training data
		"""
		if type(x) is not list:
			x = [x]
		x = np.concatenate((np.array([1]),np.array(x)),axis=0) # absorb w_0

		# No transformation, but needed later
		for i in range(self.inn):
			self.z_in[i] = x[i]
		
		# For every hidden neuron (1 hidden layer only!)
		for j in range(self.hidden):
			sumIn = 0
			for i in range(self.inn):
				sumIn += self.w_hd[j][i]*x[i]
			self.a_hd[j] = sumIn # Needed for backprop (5.56)
			self.z_hd[j] = self.act(sumIn)

		# For every output neuron
		for k in range(self.out):
			sumHdn = 0
			for j in range(self.hidden):
				sumHdn += self.w_out[k][j]*self.z_hd[j]
			self.z_out[k] = (sumHdn) # Linear output neurons = no activation function?
		return self.z_out

	def backPropagate (self, t):
		"""
			t: Target data
			errfn: Error function
		"""
		if type(t) is not list:
			t = [t]
		# Compute output deltas (using 5.54)
		for k in range(self.out):
			self.delta_out[k] += t[k] - self.z_out[k]

		# Compute hidden deltas (using 5.56)
		for j in range(self.hidden):
			deltaSum = 0
			for k in range(self.out):
				deltaSum += self.w_out[k][j]*self.delta_out[k]
			self.delta_hd[j] += self.actp(self.a_hd[j]) * deltaSum

	def updateWeights (self):
		n = 0.1 # learning rate
		# Update input/hidden layer
		for i in range(self.inn):
			for j in range(self.hidden):
				gradient = self.delta_hd[j]*self.z_in[i]
				self.w_hd[j][i] = self.w_hd[j][i] - n*gradient 

		# Update hidden/output layer
		for j in range(self.hidden):
			for k in range(self.out):
				gradient = self.delta_out[k]*self.z_hd[j]
				self.w_out[k][j] = self.w_out[k][j] - n*gradient 
		
	def training (self, xs, ts, it=100):
		"""
			xs: list of input vectors
			ts: list of target vectors
			it: number of iterations on the training data
		"""
		assert(len(xs) == len(ts)), 'Dimensions of training and target data do not correspond.'
		avg = lambda x: x/len(xs) 
		for j in range(5):
			delta_hds  = np.zeros(self.hidden)
			delta_outs = np.zeros(self.out)
			for i in range(len(xs)):
				self.forwardPropagate(xs[i])
				self.backPropagate(ts[i])
				# Save deltas
				delta_hds  += self.delta_hd
				delta_outs += self.delta_out
			# Take the average of each delta
			self.delta_hd  = map(avg, delta_hds)
			self.delta_out = map(avg, delta_outs)
			# Do update
			print "hd: ",  self.delta_hd
			print "out: ", self.delta_out
			self.updateWeights()

raw = np.loadtxt('data/sincTrain25.dt').T
ins, outs = raw[0], raw[1]

# Tests
nn = NeuralNetwork(1,2,1,activation, activationp)
nn.training(ins, outs)

# Get out predictions
for i in range(len(ins)):
	prediction = nn.forwardPropagate(ins[i])
	print outs[i], prediction

# III.1.1 Neural network training

## III.2 Support Vector Machines

# III.2.1 Data normalization
def normalise (x, avg=None, std=None):
	if avg is None:
		avg = np.average(x)
	if std is None:
		std = np.std(x)
	fn_trans = lambda x: (x-avg)/std
	return map(fn_trans, x)

# Load data 
raw_training_data    = np.loadtxt('data/parkinsonsTrainStatML.dt').T
training_data 	     = raw_training_data[:22]
training_target_data = raw_training_data[22]

raw_test_data    = np.loadtxt('data/parkinsonsTestStatML.dt').T
test_data 	     = raw_test_data[:22]
test_target_data = raw_test_data[22]

# Compute the means and variances
means_train = []
vars_train  = []
for i in training_data:
	means_train.append(np.average(i))
	vars_train.append(np.var(i))

means_test, vars_test = [], []
for i in test_data:
	means_test.append(np.average(i))
	vars_test.append(np.var(i))

# Now normalise and compute mean & variance
training_data_norm = []
means_train_norm = []
vars_train_norm  = []
for ft in training_data:
	training_data_norm.append(normalise(ft))
	means_train_norm.append(np.average(training_data_norm[-1]))
	vars_train_norm.append(np.var(training_data_norm[-1]))

test_data_norm = []
means_test_norm = []
vars_test_norm = []
for ft in range(len(test_data)):
	avg = means_train_norm[ft]
	std = math.sqrt(vars_train_norm[ft])
	# Apply f-mapping from training
	test_data_norm.append(normalise(test_data[ft],avg,std))
	means_test_norm.append(np.average(test_data_norm[-1]))
	vars_test_norm.append(np.var(test_data_norm[-1]))

# III.2.2 Model selection using grid-search

# III.2.3.1 Support vectors
