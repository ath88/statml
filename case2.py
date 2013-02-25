#!/usr/bin/python2
# -*- coding: UTF-8 -*-
# Statistical Methods for Machine Learning
# Case 2 source code
# Authors: Asbjørn Thegler, Andreas Bock, Joachim Vig
#
from __future__ import division
import math
from pylab import *
import numpy as np
import mpl_toolkits.mplot3d.axes3d as plot3d
from PIL import Image
import scipy.io

## Processing the data
# Matrix from bodyfat.mat
rawData = scipy.io.loadmat('Data/bodyfat.mat')['data']

# Selection 1 - columns 4, 7, 8, 9
sel1 = [[row[3], row[6], row[7], row[8]] for row in rawData]

# Selection 2 - columns
sel2 = [row[2] for row in sel1]

# II.1.1 Maximum likelihood solution

def y(x,w):

