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
	return 1/(1+abs(a))^2

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
		self.z_in  = np.ones(d)    #[0]*d
		self.z_hd  = np.ones(m)    #[0]*d
		self.z_out = np.ones(k)    #[0]*d
		self.w_in  = np.ones((d,1))  #mkMatrix(d,1)
		self.w_hd  = np.ones((m,d))  #mkMatrix(m,d)
		self.w_out = np.ones((k,m))  #mkMatrix(k,m)
		
		# Data structures for backpropagation
		self.delta_in  = np.ones(d)
		self.delta_hd  = np.ones(m)
		self.delta_out = np.ones(k)

	def forwardPropagate (self, x):
		""" Forward propagate in neural network.
			x: training data
			w: weights
		"""
		assert(len(x) == len(w)-1), 'Wrong dimensions of input and weight vectors.'
		x = np.concatenate((np.array([1]),np.array(x)),axis=0) # absorb w_0
		# For every input neuron (in case we wanted to add non-linear activation)
		for i in range(self.inn):
			self.z_in[i] = self.w_in[i]*x[i]

		# For every hidden neuron (1 layer only!)
		for j in range(self.hidden):
			sumIn = 0
			for i in range(self.inn):
				sumIn += self.w_hd[j][i]*self.z_in[i]
			self.z_hd[j] = self.act(sumIn)

		# For every output neuron
		for k in range(self.out):
			sumHdn = 0
			for j in range(self.hidden):
				sumHdn += self.w_out[k][j]*self.z_hd[i]
			self.z_hd[j] = self.act(sumHdn) # Linear output neurons = no activation function?
			#self.z_hd[j] = sumHdn
		return self.z_hd
	def backPropagate (self, t, errfn):
		"""
			t: Target data
			errfn: Error function
		"""
		assert(len(t)==self.out), 'Target vector malformed.'
		t = np.array(t)
		# Compute output deltas
		for k in range(self.out):
			self.delta_out[k] = t[k] - z_out[k]
	
		# Compute hidden deltas
		for j in range(self.hidden):
			deltaSum = 0
			for k in range(self.out):
				deltaSum += self.w_out[k][j]*self.delta_out[k]
			self.delta_hd[j] = self.actp(self.z_hd[j]) * deltaSum
	
		# Compute input deltas
		for i in range(self.inn):
			deltaSum = 0
			for j in range(self.out):
				deltaSum += self.w_hd[j][i]*self.delta_hd[j]
			self.delta_in[i] = self.actp(self.z_in[i]) * deltaSum
		
		# Update the weights using gradient descent
		n = 1 # learning rate
		# Update input layer
		for i in range(self.inn):
			pass #self.w_in[i] = self.w_in[i] - n*self.delta_in[i]*self.actp(

		# Update hidden layer

		# Update output layer
	def mse (self, x, y):
		pass
	def training (self, x, t):
		pass

# Test if it works on xor
xor = [
[[0,0], [0]],
[[0,1], [1]],
[[1,0], [1]],
[[1,1], [0]]
]
nn = NeuralNetwork(1,1,1,activation, activationp)
#print nn.training([2,3],[1,2,3])

# III.1.1 Neural network training

### III.2 Support Vector Machines

# III.2.1 Data normalization

# Load data

raw_training_data    = np.loadtxt('data/parkinsonsTrainStatML.dt').T
training_data 	     = raw_training_data[:22]
training_target_data = raw_training_data[22]

raw_test_data    = np.loadtxt('data/parkinsonsTestStatML.dt').T
test_data 	     = raw_test_data[:22]
test_target_data = raw_test_data[22]

means_train = []
for i in training_data:
	means_train.append(sum(i)/len(i))

means_test = []
for i in test_data:
	means_test.append(sum(i)/len(i))

# Compute the 

# III.2.2 Model selection using grid-search

# III.2.3.1 Support vectors
