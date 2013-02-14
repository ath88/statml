# Statistical Methods for Machine Learning
# Case 1 source code
# Authors: Asbjørn Thegler, Andreas Bock
#

import numpy as np
from pylab import *
from mpl_toolkits.mplot3d import Axes3D

# Question 1.1
ax = gca()
ax.yaxis.set_visible(False)

n = 50
a = np.random.normal(-1,1,n)
scatter(a, [0]*n)
#draw()

a = np.random.normal(0,2,n)
scatter(a, [1]*n)
#draw()

a = np.random.normal(2,3,n)
scatter(a, [2]*n)
#show()

# Question 1.2

means = np.array([1,1])

# Generate 200 values and turn them into 100 1x2 vectors (z vector)
b = np.random.normal(0,1,200)
Zrandoms = np.reshape(b,(100,2))

# Cholesky transform on covariance matrix
L = linalg.cholesky(array([[0.3,0.2],[0.2,0.2]]))

ySamples = []
for i in Zrandoms:
	z = i.T # transpose
	ySamples.append((means + np.dot(L,z)))
	#ySamples.append(np.transpose(np.matrix([1,1])) + np.array(L*z)) # <--- how we did it before, lamesauce

#print(ySamples[0:2])

# Question 1.3
# Estimation of sample μ and sample Σ:

# sample mean
mu1 = sum([k for (k,v) in ySamples])/100
mu2 = sum([v for (k,v) in ySamples])/100
sampleMeans = [mu1,mu2]

# sample variance
sampleVar = [[0,0],[0,0]]
for i in ySamples:
	diff = i-sampleMeans
	sampleVar+=diff*(diff.T)
sampleVar = sampleVar/100 

print(sampleMeans) # looks good
print(sampleVar) # check output compared to text's variance. Might be buggy, or maybe it's just supposed to deviate

# Now draw the stuff
fig = figure()
ax = Axes3D(fig)
X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(X, Y)
R = sam
Z = np.sin(R)

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot')
show()






