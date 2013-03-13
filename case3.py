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
	return 1/(1+abs(a))**2

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
		self.inn 	= d
		self.hidden = m
		self.out 	= k
		self.act    = act
		self.actp   = actp
		# Data structures for activation data and weights
		self.a_hd  = np.ones(m) 

		self.z_in  = np.ones(d)
		self.z_hd  = np.ones(m)    
		self.z_out = np.ones(k)    
		self.w_in  = np.ones((d,1))
		self.w_hd  = np.ones((m,d))
		self.w_out = np.ones((k,m))
		
		# Data structures for backpropagation
		self.delta_hd  = np.ones(m)
		self.delta_out = np.ones(k)
	
	def forwardPropagate (self, x):
		""" Forward propagate in neural network.
			x: training data
		"""
		assert(len(x) == self.inn), 'Wrong dimensions of input and weight vectors.'
		x = np.concatenate((np.array([1]),np.array(x)),axis=0) # absorb w_0

		# For every input neuron (in case we wanted to add non-linear activation)
		for i in range(self.inn):
			self.z_in[i] = self.w_in[i]*x[i]

		# For every hidden neuron (1 layer only!)
		for j in range(self.hidden):
			sumIn = 0
			for i in range(self.inn):
				sumIn += self.w_hd[j][i]*self.z_in[i]
			self.a_hd[j] = sumIn # Needed for backprop (5.56)
			self.z_hd[j] = self.act(sumIn)

		# For every output neuron
		for k in range(self.out):
			sumHdn = 0
			for j in range(self.hidden):
				sumHdn += self.w_out[k][j]*self.z_hd[j]
			self.z_out[k] = self.act(sumHdn) # Linear output neurons = no activation function?
			#self.z_out[k] = sumHdn
		return self.z_out

	def backPropagate (self, t, errfn=None):
		"""
			t: Target data
			errfn: Error function
		"""
		assert(len(t)==self.out), 'Target vector malformed.'
		if errfn is None:
			errfn = self.mse
		t = np.array(t)
		
		# Compute output deltas (using 5.54)
		for k in range(self.out):
			self.delta_out[k] = t[k] - self.z_out[k]
	
		# Compute hidden deltas (using 5.56)
		for j in range(self.hidden):
			deltaSum = 0
			for k in range(self.out):
				deltaSum += self.w_out[k][j]*self.delta_out[k]
			self.delta_hd[j] = self.actp(self.a_hd[j]) * deltaSum
			#self.delta_hd[j] = self.actp(self.z_hd[j]) * deltaSum
	
		# Update the weights
		n = 1 # learning rate

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
		
		# How did we do?
		return self.mse(t)

	def mse (self, t):
		error = 0
		for k in range(self.out):
			error += (t[k]-self.z_out[k])**2
		return error/self.out
	
	def training (self, xs, ts, it=100):
		"""
			xs: list of input vectors
			ts: list of target vectors
			it: number of iterations on the training data
		"""
		assert(len(xs) == len(ts)), 'Dimensions of training and target data do not correspond.'
		for i in range(it):
			for j in range(len(xs)):
				self.forwardPropagate(xs[j]) # output is stored in self.z_out
				self.backPropagate(ts[j])

# Test if it works on xor
xor_input  = [[0,0],[0,1],[1,0],[1,1]]
xor_input  = [[0,0],[0,1],[1,0],[1,1]]
xor_target = [[0],[1],[1],[0]]

#nn = NeuralNetwork(2,3,1,activation, activationp)
#nn.training(xor_input, xor_target)
#print "0,0: ", nn.forwardPropagate(xor_input[0])
#print "0,1: ", nn.forwardPropagate(xor_input[1])
#print "1,0: ", nn.forwardPropagate(xor_input[2])
#print "1,1: ", nn.forwardPropagate(xor_input[3])

# III.1.1 Neural network training

### III.2 Support Vector Machines

# III.2.1 Data normalization
def normalise (x):
	avg = np.average(x)
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
for ft in test_data:
	test_data_norm.append(normalise(ft))
	means_test_norm.append(np.average(test_data_norm[-1]))
	vars_test_norm.append(np.var(test_data_norm[-1]))

# III.2.2 Model selection using grid-search

# III.2.3.1 Support vectors
