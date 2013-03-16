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
from svmutil import * 

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
	
	def forwardPropagate (self, xs):
		""" Forward propagate in neural network.
			x: training data
		"""
		xs = np.concatenate((np.array([1]),np.array(xs)),axis=0) # absorb w_0
		for x in xs:
			x = [x]
			# For every input neuron (in case we wanted to add non-linear activation)
			for i in range(self.inn):
				self.z_in[i] = self.w_in[i]*x[i]

			# For every hidden neuron (1 hidden layer only!)
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
		# Take average
		data_len = len(xs)
		for k in range(self.out):
			self.z_out[k] /= data_len
		for j in range(self.hidden):
			self.z_hd[j] /= data_len
		return self.z_out

	def backPropagate (self, ts, errfn=None):
		"""
			t: Target data
			errfn: Error function
		"""
		if errfn is None:
			errfn = self.mse
		for t in ts:
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
		# Take average
		data_len = len(ts)
		for k in range(self.out):
			self.delta_out[k] /= data_len
		for j in range(self.hidden):
			self.delta_hd[j] /= data_len
		# Now do update	
		self.updateWeights()

	def updateWeights (self):
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
			self.forwardPropagate(xs) # output is stored in self.z_out
			self.backPropagate(ts)

raw = np.loadtxt('data/sincTrain25.dt').T
ins, outs = raw[0], raw[1]

# Tests
nn = NeuralNetwork(1,2,1,activation, activationp)
nn.training(ins, outs)

# Get out predictions
testouts = []
for i in ins:
	testouts.append(nn.forwardPropagate([i]))

#print "Errors: ", np.array(testouts)

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

raw_test_data        = np.loadtxt('data/parkinsonsTestStatML.dt').T
test_data 	     = raw_test_data[:22]
test_target_data     = raw_test_data[22]

# Compute the means and variances
means_train, vars_train = [], []
for i in training_data:
	means_train.append(np.average(i))
	vars_train.append(np.var(i))

means_test, vars_test = [], []
for i in test_data:
	means_test.append(np.average(i))
	vars_test.append(np.var(i))

# Now normalise and compute mean & variance
training_data_norm, means_train_norm, vars_train_norm  = [], [], []
for ft in training_data:
	training_data_norm.append(normalise(ft))
	means_train_norm.append(np.average(training_data_norm[-1]))
	vars_train_norm.append(np.var(training_data_norm[-1]))

test_data_norm, means_test_norm, vars_test_norm = [], [], []
for ft in range(len(test_data)):
	avg = means_train_norm[ft]
	std = math.sqrt(vars_train_norm[ft])
	# Apply f-mapping from training
	test_data_norm.append(normalise(test_data[ft],avg,std))
	means_test_norm.append(np.average(test_data_norm[-1]))
	vars_test_norm.append(np.var(test_data_norm[-1]))

training_data = transpose(training_data).tolist()
test_data = transpose(test_data).tolist()
training_data_norm = transpose(training_data_norm).tolist()
test_data_norm = transpose(test_data_norm).tolist()

# III.2.2 Model selection using grid-search

def grid_search(tr_samples,tr_targets,te_samples,te_targets):
	Cs     = [0.001,0.01,0.1,1,10,100,100,1000,10000]
	gammas = [0.001,0.01,0.1,1,10,100,100,1000,10000]
	problem = svm_problem(tr_targets,tr_samples)
	parameter = svm_parameter('-q')
	parameter.kernel_type = RBF
	parameter.cross_validation = 1
	parameter.nr_fold = 5

	best = 0
	values = (None, None)
	for C in Cs:
		for gamma in gammas:
			parameter.C = C
			parameter.gamma = gamma 

			sys.stdout = open(os.devnull,'w')
			accuracy = svm_train(problem, parameter)
			sys.stdout = sys.__stdout__
			
			if accuracy > best:
				best = accuracy
				values = (C, gamma)

	print "Best accuracy during model selection:", best
	print "C and gamma for best model: ", values

	parameter.C = values[0]
	parameter.gamma = values[1]
	parameter.cross_validation = 0
	model = svm_train(problem,parameter)

	sys.stderr = open(os.devnull,'w')
	sys.stdout = open(os.devnull,'w')
	result = svm_predict(te_targets,te_samples,model)
	sys.stderr = sys.__stderr__
	sys.stdout = sys.__stdout__

	print "Accuracy for test data using best model: ", result[1][0]

grid_search(training_data,training_target_data,test_data,test_target_data)
grid_search(training_data_norm,training_target_data,test_data_norm,test_target_data)


# III.2.3.1 Support vectors
