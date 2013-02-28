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
from scipy import misc

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
figure()
title('Histogram estimate of p(x1)')

# Plot the histogram estimate
hist1 = histogram(x1s)
xlocs = array(range(len(hist1[0])))

# Plot the analytical solution, u=1, var=0.3
norm_xlocs = linspace(-0.5,2.5,200)

# Correcting analytical solution because of bin width
bar(hist1[1][:-1], hist1[0]/100,np.diff(hist1[1]))
plot(norm_xlocs, normpdf(norm_xlocs, 1, 0.3)/10.0, color="red")

#xlim(xlocs[0]-2, xlocs[-1]+2)
#ylim(0,1)
#show()
figure()

#bar(xlocs, hist1[0])
#prange = np.arange(0, 10, 0.001)
#plot(prange, normpdf(prange, 5, (0.3))/20)
#show()
#
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
def average(y):
	return sum(y)/len(y)

# Question 1.8
def generateValues (lda, L, count):
	mu_y = 1/lda
	mu_est = 0
	for i in range(count):
		randomYs = np.random.uniform(0,1,L)
		y = map(lambda x: -mu_y*math.log(1-x),randomYs)
		mu_est += abs(mu_y - average(y))
	mu_est /= count
	return mu_est

# We fix lambda = 10
l = 10

# We now plot the expected absolute deviation
lvalues = range(1,10,1)
lvalues = [10**i for i in lvalues]
abs_deviations = [generateValues(l, i, 10) for i in lvalues]

# Plotting the absolute deviation
figure()
ax = gca()
title('Expected absolute deviation')
ax.xaxis.set_visible(True)
ylabel('y')
xlabel('x')
ylim(0,0.04)
grid(True)
plot(lvalues, abs_deviations)
#show()
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
#show()
#fig2.savefig('q18_2.jpg')

# Question 1.9

# Helper function
global cache
cache = {} 

def multi_norm (x, sigma, mu):
        key = tuple(x.tolist()[0])
        try:
            return cache[key]
        except:
            pass
	const = pow((2*np.pi), (len(mu.T)/2)) * pow(np.linalg.det(sigma), 0.5)
	x_mu = np.matrix((x-mu)).T
	precision = np.matrix(sigma).I
        exp = -0.5*dot(x_mu.T, dot(precision, x_mu))
	cache[key] = (np.exp(exp)/const).tolist()[0][0]
        return cache[key]

def reDraw(pixel, sigma, mu):
	mnorm = multi_norm(pixel, sigma, mu)
	meanColor = multi_norm(mu, sigma, mu)
        return thermcolor(mnorm,meanColor)

def thermcolor(value,maxvalue):
        ratio = value/maxvalue
        scale = ratio * 5 * 255
        if scale <= 255:
            return [0,0,scale]
        elif scale > 255 and scale <= 255*2:
            return [scale - 255, 0, 255]
        elif scale > 255*2 and scale <= 255*3:
            return [255, 0, 255 - (scale - 255*2)]
        elif scale > 255*3 and scale <= 255*4:
            return [255, scale - 255*3, 0]
        elif scale > 255*4:
            return [255, 255, scale - 255*4]
        return [0,0,0]


# Process training set
im = Image.open("kande1.jpg").crop((150,264,330,328))

r = []
g = []
b = []
for a in im.getcolors(10000000): 
    for i in range(0,a[0]):
	r.append(a[1][0])
	g.append(a[1][1])
	b.append(a[1][2])

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

figure()
img = misc.imread("kande1.jpg")

# Question 1.10
qhat = np.array([0,0])
C = matrix(zeros((2,2)))
Z = 0
for i in range(0,img.shape[0]): 	  # width of image
	for j in range(0,img.shape[1]): # height of image
		norm_const = multi_norm(matrix(img[i,j]), cov, mean)

		Z += norm_const
                 
                #Weighted Average
		qhat = qhat + dot(np.array([i,j]),norm_const)


qhat /= Z

for i in range(0,img.shape[0]): 	  # width of image
	for j in range(0,img.shape[1]): # height of image
		norm_const = multi_norm(matrix(img[i,j]), cov, mean)
		#Spatial Covarianec
		qdiff = np.matrix([i,j]) - np.matrix(qhat)
		C = C + np.dot(qdiff.T, qdiff) * norm_const

                #redrawing according to covariance
		img[i,j] = reDraw(matrix(img[i,j]), cov, mean)
C /= Z

imshow(img)
scatter(qhat[1],qhat[0],color='green',s=100)

contours = zeros((img.shape[0],img.shape[1]))
for i in range(0,img.shape[0]): 	  # width of image
	for j in range(0,img.shape[1]): # height of image
                contours[i][j] = multi_norm(matrix([i,j]),C,qhat)

x = range(0,640)
y = range(0,480)
contour(y,x,contours,colors='#00ff00')
title("Attempt at detecting the pitcher in kande1.jpg")

# clear cache
cache = {}

# Question 1.11

figure()
img = misc.imread("kande2.jpg")

qhat = np.array([0,0])
C = matrix(zeros((2,2)))
Z = 0
for i in range(0,img.shape[0]): 	  # width of image
	for j in range(0,img.shape[1]): # height of image
		norm_const = multi_norm(matrix(img[i,j]), cov, mean)

		Z += norm_const
                 
                #Weighted Average
		qhat = qhat + dot(np.array([i,j]),norm_const)

qhat /= Z

for i in range(0,img.shape[0]): 	  # width of image
	for j in range(0,img.shape[1]): # height of image
		norm_const = multi_norm(matrix(img[i,j]), cov, mean)
		#Spatial Covarianec
		qdiff = np.matrix([i,j]) - np.matrix(qhat)
		C = C + np.dot(qdiff.T, qdiff) * norm_const

                #redrawing according to covariance
		img[i,j] = reDraw(matrix(img[i,j]), cov, mean)
C /= Z

imshow(img)
scatter(qhat[1],qhat[0],color='green',s=100)

contours = zeros((img.shape[0],img.shape[1]))
for i in range(0,img.shape[0]): 	  # width of image
	for j in range(0,img.shape[1]): # height of image
                contours[i][j] = multi_norm(matrix([i,j]),C,qhat,)

x = range(0,640)
y = range(0,480)
title("Attempt at detecting the pitcher in kande2.jpg")
contour(y,x,contours,colors='#00ff00')

show()
