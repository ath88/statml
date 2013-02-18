#!/usr/bin/python3
# -*- coding: UTF-8 -*-
# Statistical Methods for Machine Learning
# Case 1 source code
# Authors: Asbjørn Thegler, Andreas Bock
#
import math
from scipy.stats import norm
from pylab import *
import numpy as np
import mpl_toolkits.mplot3d.axes3d as plot3d

# Helper functions
# Might be useful later?
#def multi_norm (x,y,sigma,mu):
#	const = 1.0/(np.power(2*np.pi,len(mu)/2)*np.sqrt(np.linalg.det(sigma)))
#	x_mu = np.array((x-mu))
#	precision = np.matrix(sigma).I
#	return const*np.exp(-0.5*x_mu*precision*x_mu.T)

# Question 1.1
ax = gca()
ax.yaxis.set_visible(False)

n = 50
a = np.random.normal(-1,1,n)
#scatter(a, [0]*n)
#draw()

a = np.random.normal(0,2,n)
#scatter(a, [1]*n)
#draw()

a = np.random.normal(2,3,n)
#scatter(a, [2]*n)
#show()

# Question 1.2

means = np.array([1,1])
covariance = np.matrix([[0.3,0.2],[0.2,0.2]])

# Generate 200 values and turn them into 100 1x2 vectors (z vector)
b = np.random.normal(0,1,200)
Zrandoms = np.reshape(b,(100,2))

# Cholesky transform on covariance matrix
L = linalg.cholesky(covariance)

ySamples = []
for i in Zrandoms:
	z = i.T # transpose
	ySamples.append((means + np.dot(L,z)))
# Question 1.3
# Estimation of sample μ and sample Σ:

observations = 100
x1s = [i[:,0][0,0] for i in ySamples]
x2s = [i[:,1][0,0] for i in ySamples]

# Sample mean
mu1 = sum(x1s)/observations
mu2 = sum(x2s)/observations
sampleMeans = [mu1,mu2]

# Sample variance
sampleVar = [[0,0],[0,0]]
for i in ySamples:
	diff = i-sampleMeans
	sampleVar+=diff*(diff.T)
sampleVar = sampleVar/observations

# Plot sample mean, mean and data points
title('Maximum likelihood sample mean')
scatter(x1s, x2s)
scatter(means[0], means[1], color="red")
scatter(sampleMeans[0], sampleMeans[1], color="green")

diff_in_mean = abs(sampleMeans - means)

# Question 1.5

bins = 8
x1s = []
x2s = []
for i in ySamples:
  x1s.append(i[:,0][0,0])
  x2s.append(i[:,1][0,0])

figure()
histo1 = histogram(x1s,bins)
xlocations1 = array(range(len(histo1[0])))+0.1
ax = gca()
ax.xaxis.set_visible(False)
title("x1 values")
bar(xlocations1,histo1[0])

figure()
histo2 = histogram(x2s,bins)
xlocations2 = array(range(len(histo2[0])))+0.1
ax = gca()
ax.xaxis.set_visible(False)
title("x2 values")
bar(xlocations2+0.2,histo2[0])

# Question 1.6

figure()
title('Histogram estimate of p(x1)')
pX1 = array([k for (k,v) in Zrandoms])
hist1 = histogram(pX1, density=True)
xlocs = array(range(len(hist1[0])))+0.1
#bar(xlocs, hist1[0])
#figure()

bar(xlocs, hist1[0])
prange = np.arange(0, 10, 0.001)
plt.plot(prange, normpdf(prange, 5, math.sqrt(0.3)))
show()

# Question 1.7
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
hist, xedges, yedges = histogram2d(x1s, x2s, bins=8)

elements = (len(xedges) - 1) * (len(yedges) - 1)
xpos, ypos = meshgrid(xedges[:-1]+0.1, yedges[:-1]+0.1)

xpos = xpos.flatten()
ypos = ypos.flatten()
zpos = np.zeros(elements)
dx = 0.34 * ones_like(zpos)
dy = dx.copy()
dz = hist.flatten()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='r', zsort='average')

plt.show()

# Question 1.8

#def ptransform
