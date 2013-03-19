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
import copy

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

		self.z_in  = np.ones(d)
		self.z_hd  = np.ones(m)    
		self.z_out = np.ones(k)    
		self.w_hd  = np.ones((m,d))
		self.w_out = np.ones((k,m))
	
		self.g_hd  = np.zeros((m,d))
		self.g_out = np.zeros((k,m))
		
		# Data structures for backpropagation
		self.delta_hd  = np.zeros(m)
		self.delta_out = np.zeros(k)
	
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
				sumIn += self.w_hd[j][i]*self.z_in[i]
			self.a_hd[j] = sumIn # Needed for backprop (5.56)
			self.z_hd[j] = self.act(sumIn)

		# For every output neuron
		for k in range(self.out):
			sumHdn = 0
			for j in range(self.hidden):
				sumHdn += self.w_out[k][j]*self.z_hd[j]
			self.z_out[k] = sumHdn
		return self.z_out

	def backPropagate (self, t):
		""" Backpropagate through the network and
			set deltas and gradient matrix.
			t: Target data
			errfn: Error function
		"""
		if type(t) is not list:
			t = [t]
		# Compute output deltas (using 5.54)
		for k in range(self.out):
			self.delta_out[k] = self.z_out[k] - t[k]

		# Compute hidden deltas (using 5.56)
		for j in range(self.hidden):
			deltaSum = 0
			for k in range(self.out):
				deltaSum += self.w_out[k][j]*self.delta_out[k]
			self.delta_hd[j] = self.actp(self.a_hd[j]) * deltaSum
	
		# Accumulate gradients
		for i in range(self.inn):
			for j in range(self.hidden):
				self.g_hd[j][i] += self.delta_hd[j]*self.z_in[i]
		for j in range(self.hidden):
			for k in range(self.out):
				self.g_out[k][j] += self.delta_out[k]*self.z_hd[j]

	def updateWeights (self, n=0.05):
		""" Updates the network weights based on gradient matrix.
			n: learning rate
		""" 
		# Update input/hidden layer
		for i in range(self.inn):
			for j in range(self.hidden):
				self.w_hd[j][i] = self.w_hd[j][i] - n*self.g_hd[j][i] 
		# Update hidden/output layer
		for j in range(self.hidden):
			for k in range(self.out):
				self.w_out[k][j] = self.w_out[k][j] - n*self.g_out[k][j] 
		
	def training (self, xs, ts, desired_error=0.05, early_stopping=False):
		""" Trains the neural network on xs and ts.
			xs: list of input vectors
			ts: list of target vectors
			it: number of iterations on the training data
		"""
		assert(len(xs) == len(ts)), 'Dimensions of training and target data do not correspond.'
		self.forwardPropagate(xs[0])
		self.backPropagate(xs[0])
		early_stop = float("inf")
		this_error = float("inf")
		all_errors = 0
		j = 0
		while this_error >= desired_error:
			j +=1
			self.g_hd  = np.zeros((3,2))
			self.g_out = np.zeros((1,3))
			for i in range(len(xs)):
				self.forwardPropagate(xs[i])
				self.backPropagate(ts[i])
			# Do update
			self.updateWeights()
			# Calculate error
			this_error = self.calc_error(self.z_out, ts[i])
			print "Error: ",this_error
			all_errors += this_error
			if early_stopping:
				if this_error > early_stop:
					print "Early stop at iteration: ", (j-1)*len(xs)+i
					break
				early_stop = this_error
		print "Sum of squares error over all training data: ", all_errors
	def calc_error (self, output, target):
		""" Calculates error for output vector
			output: output vector (length == k)
			target: target vector (length == k)
		"""
		err = 0
		if target is not list:
			target = [target]
		#print "o",target
		for i in range(len(output)):
			err += 0.5*(output[i]-target[i])**2	
		print err
		return err

raw = np.loadtxt('data/sincTrain25.dt').T
ins, outs = raw[0], raw[1]

# Tests
nn = NeuralNetwork(1,2,1,activation, activationp)
nn.training(ins, outs)#,(early_stopping=True)

ps = []
for i in range(len(ins)):
	p = nn.forwardPropagate(ins[i])[0]
	ps.append(p)
	print p, outs[i]

figure()
ylabel('Predictions/Targets')
xlabel('x')
scatter(ins, outs)
scatter(ins, ps, c="red")
show()

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
test_data 	       	= raw_test_data[:22]
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

# Redefine for libSVM compatability
training_data = transpose(training_data).tolist()
test_data = transpose(test_data).tolist()
training_data_norm = transpose(training_data_norm).tolist()
test_data_norm = transpose(test_data_norm).tolist()

# III.2.2 Model selection using grid-search

def get_free_bounded (model, C):
	# Get coefficients
	coefs = [k[0] for k in model.get_sv_coef()]
	no_free, no_bounded = 0, 0
	for i in coefs:
		if abs(i) == C:
			no_bounded += 1
		else:
			no_free += 1
	return (no_free, no_bounded)

def grid_search(tr_samples,tr_targets,te_samples,te_targets, verbose=False, C=None):
	Cs     = [0.001,0.01,0.1,1,10,100,1000,10000]
	gammas = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
	problem = svm_problem(tr_targets,tr_samples)
	parameter = svm_parameter('-q')
	parameter.kernel_type = RBF
	parameter.cross_validation = 1
	parameter.nr_fold = 5

	best = 0
	values = (None, None)

	if C is None:
		for c in Cs:
			for gamma in gammas:
				parameter.C = c
				parameter.gamma = gamma

				sys.stdout = open(os.devnull,'w')
				accuracy = svm_train(problem, parameter)
				sys.stdout = sys.__stdout__
				
				if accuracy > best:
					best = accuracy
					values = (c, gamma)
	else:
		parameter.C = C
		for gamma in gammas:
			parameter.gamma = gamma

			sys.stdout = open(os.devnull,'w')
			accuracy = svm_train(problem, parameter)
			sys.stdout = sys.__stdout__
			if accuracy > best:
				best = accuracy
				values = (C, gamma)
	parameter.C = values[0]
	parameter.gamma = values[1]
	parameter.cross_validation = 0
	model = svm_train(problem,parameter)
	
	sys.stderr = open(os.devnull,'w') # Because libSVM pollutes stdout/err
	sys.stdout = open(os.devnull,'w')
	result = svm_predict(te_targets,te_samples,model)
	sys.stderr = sys.__stderr__
	sys.stdout = sys.__stdout__
	
	if verbose:
		print "Best accuracy during model selection:", best
		print "C and gamma for best model: ", values
		print "Accuracy for test data using best model: ", result[1][0]
		print get_free_bounded(model, C)
	return (model,values)

_, bestvals = grid_search(training_data,training_target_data,test_data,test_target_data,verbose=True)
model, bestvals_norm = grid_search(training_data_norm,training_target_data,test_data_norm,test_target_data,verbose=True)

# III.2.3.1 Support vectors

# Count number of free/bounded support vectors (using normalized data):
print "No. free & bounded support vectors for C=", bestvals[0], ": ",get_free_bounded(model,bestvals[0])

# Let's observe the impact of changing the value of regularization parameter C
model0, (c0,g0) = grid_search(training_data,training_target_data,test_data,test_target_data, C=0.0001)
model1, (c1,g1) = grid_search(training_data,training_target_data,test_data,test_target_data, C=0.14)
model2, (c2,g2) = grid_search(training_data,training_target_data,test_data,test_target_data, C=1.5)
model3, (c3,g3) = grid_search(training_data,training_target_data,test_data,test_target_data, C=300)

free0, bounded0 = get_free_bounded(model0, c0)
free1, bounded1 = get_free_bounded(model1, c1)
free2, bounded2 = get_free_bounded(model2, c2)
free3, bounded3 = get_free_bounded(model3, c3)

print "-----------------------"
print "C      | Free | Bounded"
print "-----------------------"
print "1*10^-4\t  ", free0, "\t ", bounded0
print "0.14\t ", free1, "\t ", bounded1
print "1.5\t ", free2, "\t  ", bounded2
print "300\t ", free3, "\t  ", bounded3
