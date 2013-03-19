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
import sys

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
		self.inn    = d
		self.hidden = m
		self.out    = k
		self.act    = act
		self.actp   = actp

		self.a_hd  = np.ones(m) # activation values for hidden neurons.

		self.z_in  = np.ones(d)
		self.z_hd  = np.ones(m)    
		self.z_out = np.ones(k)    
		self.w_hd  = np.ones((m,d)) # weights of connections hidden neurons. m x d matrix
		self.w_out = np.ones((k,m)) # weights of connections to output neurons. k*m matrix
	
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
	avg = means_train[ft]
	std = math.sqrt(vars_train[ft])
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
	free, bounded = 0, 0
	for i in coefs:
		if abs(i) == C:
			bounded += 1
		else:
			free += 1
	return (free, bounded)

def chunkIt(seq, num):
	avg = len(seq) / float(num)
	out = []
	last = 0.0
	while last < len(seq):
		out.append(seq[int(last):int(last + avg)])
		last += avg
	return out

def n_cross_valid(n, tr_targets,tr_samples,parameter):
	problem = svm_problem(tr_targets,tr_samples)

	#splitting both data and targets into n chunks
	split_samples = chunkIt(tr_samples,n)
	split_targets = chunkIt(tr_targets,n)
	
	accuracy = 0.0
	for i in range(0,n):
		#making copies of data and target
                samples = split_samples[:]
                targets = split_targets[:]

		#deleting validation set from the copies
		del samples[i]
		del targets[i]
                samples = [a[0] for a in samples]
                targets = [a[0] for a in targets]

		#making a model from the training set without the validation set
		problem = svm_problem(targets,samples)
        	model = svm_train(problem,parameter)

		#validating the model using the validation set
		sys.stderr = open(os.devnull,'w') # Because libSVM pollutes stdout/err
		sys.stdout = open(os.devnull,'w')
		result = svm_predict(split_targets[i],split_samples[i],model)
		sys.stderr = sys.__stderr__
		sys.stdout = sys.__stdout__

		# summing the accuracies of every validation
		accuracy += result[1][0]

	# return the average accuracy
	accuracy = accuracy / n
	return accuracy

def grid_search(tr_samples,tr_targets,te_samples,te_targets, verbose=False, C=None):
	Cs     = [0.001,0.01,0.1,1,10,100,1000]
	gammas = [0.001,0.01,0.1,1,10,100,1000]
	problem = svm_problem(tr_targets,tr_samples)

	best_accuracy = 0
	values = (None, None)

	if C is None:
		for c in Cs:
#			sys.stdout.write(str(c) + ' &')
			for gamma in gammas:
				parameter = svm_parameter('-q -c ' + str(c) + ' -g ' + str(gamma))
				accuracy = n_cross_valid(5,tr_targets,tr_samples, parameter)
				
				if accuracy > best_accuracy:
					best_accuracy = accuracy
					values = (c, gamma)
#				sys.stdout.write(" " + str(round(accuracy,2)) + "\\% &")
#			print("\\\\")
	else:
		for gamma in gammas:
			parameter = svm_parameter('-q -c ' + str(C) + ' -g ' + str(gamma))

			sys.stdout = open(os.devnull,'w')
			accuracy = n_cross_valid(5,tr_targets,tr_samples, parameter)
			sys.stdout = sys.__stdout__
			if accuracy > best_accuracy:
				best_accuracy = accuracy
				values = (C, gamma)

	model = svm_train(problem,"-q -c " + str(values[0]) + " -g " + str(values[1]))
	
	sys.stderr = open(os.devnull,'w') # Because libSVM pollutes stdout/err
	sys.stdout = open(os.devnull,'w')
	result = svm_predict(te_targets,te_samples,model)
	sys.stderr = sys.__stderr__
	sys.stdout = sys.__stdout__
	
	if verbose:
		print "Best accuracy during model selection:", best_accuracy
		print "C and gamma for best model: ", values
		print "Accuracy for test data using best model: ", result[1][0]
	return (model,values)

print "\nGrid-searching on non-normalised data"
_, bestvals = grid_search(training_data,training_target_data,test_data,test_target_data,verbose=True)
print "\nGrid-searching on normalised data"
model, bestvals_norm = grid_search(training_data_norm,training_target_data,test_data_norm,test_target_data,verbose=True)

# III.2.3.1 Support vectors

# Count number of free/bounded support vectors (using normalized data):
print "No. free & bounded support vectors ( C =", bestvals_norm[0], "): ",get_free_bounded(model,bestvals_norm[0])

# Let's observe the impact of changing the value of regularization parameter C
model0, (c0,g0) = grid_search(training_data,training_target_data,test_data,test_target_data, C=0.0001)
model1, (c1,g1) = grid_search(training_data,training_target_data,test_data,test_target_data, C=0.14)
model2, (c2,g2) = grid_search(training_data,training_target_data,test_data,test_target_data, C=1.4693)
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
print "1.4693\t ", free2, "\t ", bounded2
print "300\t ", free3, "\t  ", bounded3
