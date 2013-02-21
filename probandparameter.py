#!/usr/bin/python2
# -*- coding: UTF-8 -*-
# Statistical Methods for Machine Learning
# Case 1 source code
# Authors: Asbjørn Thegler, Andreas Bock, Joachim Vig
#
from __future__ import division
import math
#from scipy.stats import norm
from pylab import *
import numpy as np
import mpl_toolkits.mplot3d.axes3d as plot3d
from PIL import Image

# Helper functions
# Might be useful later?
#def multi_norm (x,y,sigma,mu):
#	const = 1.0/(np.power(2*np.pi,len(mu)/2)*np.sqrt(np.linalg.det(sigma)))
#	x_mu = np.array((x-mu))
#	precision = np.matrix(sigma).I
#	return const*np.exp(-0.5*x_mu*precision*x_mu.T)

# Question 1.1

title("3 Gaussian distribution functions with different mean and standard deviation")
ylabel('y')
xlabel('x')
axis([-7,9,0,0.5])

ax = gca()
x = linspace(-6,8,200)

mean = -1
variance = 1
sigma = sqrt(variance)
#plot(x,normpdf(x,mean,sigma))

mean = 0
variance = 2
sigma = sqrt(variance)
#plot(x,normpdf(x,mean,sigma))

mean = 2
variance = 3
sigma = sqrt(variance)
#plot(x,normpdf(x,mean,sigma))


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
#scatter(x1s, x2s)
#scatter(means[0], means[1], color="red")
#scatter(sampleMeans[0], sampleMeans[1], color="green")

# The difference between the sampe and true mean
diff_in_mean = abs(sampleMeans - means)
print(diff_in_mean)

# Question 1.5
# Complete, 8 bins seems to be the best
bins = 8

#figure()
histo1 = histogram(x1s,bins)
xlocations1 = array(range(len(histo1[0])))+0.1
ax = gca()
ax.xaxis.set_visible(False)
title("x1 values")
#bar(xlocations1,histo1[0])

#figure()
histo2 = histogram(x2s,bins)
xlocations2 = array(range(len(histo2[0])))+0.1
ax = gca()
ax.xaxis.set_visible(False)
title("x2 values")
#bar(xlocations2+0.2,histo2[0])

# Question 1.6
#figure()
#title('Histogram estimate of p(x1)')
# Plot the histogram estimate
hist1 = histogram(x1s, density=False)
xlocs = array(range(len(hist1[0])))+0.1
# Plot the analytical solution
norm_xlocs = linspace(-2,12,200)
plot(norm_xlocs, normpdf(norm_xlocs, 5, math.sqrt(6)))
bar(xlocs, hist1[0]/100)
xlim(xlocs[0]-2, xlocs[-1]+2)
ylim(0,1)
show()
figure()

#bar(xlocs, hist1[0])
#prange = np.arange(0, 10, 0.001)
#plot(prange, normpdf(prange, 5, math.sqrt(0.3)))
#show()

# Question 1.7
N = 100
bins = 20

bBig = np.random.normal(0,1,N*2)
ZrandomsBig = np.reshape(bBig,(N,2))

ySamplesBig = []
for i in ZrandomsBig:
	z = i.T # transpose
	ySamplesBig.append((means + np.dot(L,z)))

x1b = [i[:,0][0,0] for i in ySamplesBig]
x2b = [i[:,1][0,0] for i in ySamplesBig]

hist, xedges, yedges = histogram2d(x1b, x2b, bins=bins)

elements = (len(xedges) - 1) * (len(yedges) - 1)
xpos, ypos = meshgrid(xedges[:-1]+0.1, yedges[:-1]+0.1)

xpos = xpos.flatten()
ypos = ypos.flatten()
zpos = np.zeros(elements)
dx = 0.12 * ones_like(zpos)
dy = dx.copy()
dz = hist.flatten()

title("Histogram, 20 bins, 100 samples")

#draw graph
#fig = figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='r', zsort='average')

#show()

# Question 1.8
def ptransform(y,lda):
	if y < 0:
		raise Exception("y must be non-negative.")
	return lda*exp(-lda*y)

def generateValues (lda, L, count):
	mu_y = 1/lda
	mu_est = 0
	for i in range(1,count):
		y = np.random.uniform(0,1,L)
		tmpySum = 0
		for i in range(1,L):
			tmpySum += y[i-1]**i
		tmpySum /= L
		mu_est += abs(mu_y - tmpySum)
	mu_est /= count
	return mu_est

# We fix lambda = 10
l = 10

# We now plot the expected absolute deviation
# x-values
lvalues = range(1,500,5)
abs_deviations = [generateValues(l, i, 10) for i in lvalues]


# Plotting the absolute deviation
title('Expected absolute deviation')
ax.xaxis.set_visible(True)
ylabel('y')
xlabel('x')
xlim(6,500)
ylim(0,0.25)
grid(True)
plot(lvalues, abs_deviations)
#show()


# Plotting a transformed value
fig = figure()
title('Expected absolute deviation [transformed]')
grid(True)
ax = fig.add_subplot(1,1,1)
ax.set_yscale('log')
ax.set_xscale('log')
ylabel('y')
xlabel('x')

#plot(lvalues, abs_deviations)
#show()

# Question 1.9
im = Image.open("kande1.pnm").crop((150,264,330,328))

r = []
g = []
b = []
for a in im.getcolors(10000000): #number is max amount of different colors. output i (a (r,g,b)) where a is occurrences of the color
  for i in range(0,a[0]):        #expanding, so every pixel is represented by an rgb-tuple exactly once
    r.append(a[1][0])
    g.append(a[1][1]) 
    b.append(a[1][2]) 

rMean = sum(r)/len(r)
gMean = sum(g)/len(g)
bMean = sum(b)/len(b)

#raw = matrix([r,g,b]).transpose()
#print raw
#
#ones = matrix(ones((len(r),len(r)),dtype=int))
#print ones
#
#dot = ones.dot(raw) * 1/len(r)
#print dot
#
#a = raw - dot
#print a
#
#aa = a.transpose().dot(a)
#print aa
#
#cov = aa* 1/len(r)
#print cov
#
#figure()
#scatter(r, g)
#figure()
#scatter(r, b)
#figure()
#scatter(b, g)
#show()

