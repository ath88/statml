#!/usr/bin/python2
# -*- coding: UTF-8 -*-
# Statistical Methods for Machine Learning
# Case 1 source code
# Authors: Asbjørn Thegler, Andreas Bock, Joachim Vig
#
from __future__ import division
import math
from pylab import *
import numpy as np
import mpl_toolkits.mplot3d.axes3d as plot3d
from PIL import Image

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
#print(diff_in_mean)

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
plot(norm_xlocs, normpdf(norm_xlocs, 5+0.3, math.sqrt(6)))
bar(xlocs, hist1[0]/100)
xlim(xlocs[0]-2, xlocs[-1]+2)
ylim(0,1)
#show()
#figure()

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
lvalues = range(1,500,5)
abs_deviations = [generateValues(l, i, 10) for i in lvalues]

# Plotting the absolute deviation
fig1 = figure()
ax = gca()
title('Expected absolute deviation')
ax.xaxis.set_visible(True)
ylabel('y')
xlabel('x')
xlim(6,500)
ylim(0,0.25)
grid(True)
plot(lvalues, abs_deviations)
show()
#fig1.savefig('q18_1.jpg')

# Plotting a transformed value
fig2 = figure()
ax = gca()
title('Expected absolute deviation [transformed]')
grid(True)
ax = fig2.add_subplot(1,1,1)
ax.set_yscale('log')
ax.set_xscale('log')
ylabel('y')
xlabel('x')
plot(lvalues, abs_deviations)
show()
#fig2.savefig('q18_2.jpg')

# Question 1.9

# Helper function
def multi_norm (x, sigma, mu):
	const = 1.0/(((2*np.pi)**(len(mu.T)/2))*np.sqrt(np.linalg.det(sigma)))
	part1 = 1/((2*np.pi)**(len(mu)/2))
	part2 = 1/(np.linalg.det(sigma)**0.5)
	x_mu = np.matrix((x-mu)).T
	precision = np.matrix(sigma).I
	return const*np.exp(-0.5*dot(x_mu.T, dot(precision, x_mu)))

def reDraw(pixel, sigma, mu):
	mnorm = multi_norm(pixel, sigma, mu)
	meanColour = multi_norm(mu, sigma, mu)
	if mnorm > meanColour/1.07:
		return (255,255,255) # white
	elif mnorm > meanColour/1.1:
		return (192,192,192)
	elif mnorm > meanColour/1.2:
		return (112,112,112)
	return (0,0,0) # black

# Process training set
im = Image.open("kande1.jpg").crop((150,264,330,328))

r = []
g = []
b = []
length = 0
for a in im.getcolors(10000000): 
	occ = a[0]
	r.append(a[1][0]*occ)
	g.append(a[1][1]*occ)
	b.append(a[1][2]*occ)

# Maximum likelihood estimate for sample mean
mean = []
mean.append(sum(r)/len(r))
mean.append(sum(g)/len(g))
mean.append(sum(b)/len(b))
mean = matrix(mean)

# Maximum likelihood estimate for sample cov matrix (2.122)
cov = matrix(zeros((3,3),dtype=float64))
for i in range(0,len(r)):
	raw = matrix([r[i],g[i],b[i]])
	sub = raw - mean
	cov += dot(sub.transpose(),sub)
cov /= len(r)

# Process all pixels
im = Image.open("kande1.jpg")
pixs = im.load()

# UNCOMMENT THE FOLLOWING TO SEE THE NEW IMAGE

# Generate new image
for i in range(0,im.size[0]): # width of image
	for j in range(0,im.size[1]): #height of image
		pixs[i,j] = reDraw(pixs[i,j], cov, mean)
im.save('new_kande1.jpg')

# Question 1.10

# Weighted average position
qhat = [0,0]
Z = 0
for i in range(0,im.size[0]): 	  # width of image
	for j in range(0,im.size[1]): # height of image
		pixs[i,j] = reDraw(pixs[i,j], cov, mean)
		norm_const = multi_norm(pixs[i,j], cov, mean)[0,0]
		Z += norm_const
		qhat += list(map(lambda x: x*norm_const, np.array([i,j])))
qhat /= Z

# Spatial covariance
C = 0
for i in range(0,im.size[0]): # width of image
	for j in range(0,im.size[1]): #height of image
		pixs[i,j] = reDraw(pixs[i,j], cov, mean)
		norm_const = multi_norm(pixs[i,j], cov, mean)[0,0]
		qdiff = np.array([i,j]) - qhat
		C += np.dot(qdiff, qdiff.T)*norm_const
C /= Z

# Plot q hat and contours of C on top of our image

# Question 1.11

im2 = Image.open("kande2.jpg")
pixs = im2.load()

# Generate new image for the second pitcher
for i in range(0,im2.size[0]): # width of image
	for j in range(0,im2.size[1]): #height of image
		pixs[i,j] = reDraw(pixs[i,j], cov, mean)

im2.save('new_kande2.jpg')

