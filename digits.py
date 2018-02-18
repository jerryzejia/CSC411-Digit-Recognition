from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random

import cPickle

import os
from scipy.io import loadmat

#Part 1

def part1():
    M = loadmat("mnist_all.mat")
    f, axarr = plt.subplots(10,10)
    for i in range(10):
        for j in range(10):
            axarr[i,j].imshow(M["train"+str(i)][j].reshape((28,28)),cmap=cm.gray)
    plt.show()

#Part 2

def softmax(y):
    '''Return the output of the softmax function for the matrix of output y. y
    is an NxM matrix where N is the number of outputs for a single case, and M
    is the number of cases'''
    return exp(y)/tile(sum(exp(y),0), (len(y),1))

def f(x,w):
    print x.shape
    print w.shape
    return softmax(dot(w.T,x))

#Part 3

#Cost function
def NLL(y, y_):
    return -sum(y_*log(y))

def gradient(x,y,w):
	p = f(x,w)
	return dot(x,(y-p).T)

#Part 4
def input():
    M = loadmat("mnist_all.mat")
    x = M["train0"][0:5000]
    for i in range(9):
    	x = vstack((x,M["train"+str(i+1)][0:5000]/255.0))
    x = x.T
    print x.shape
    x = vstack((np.ones(50000),x))
    y = np.zeros((50000,10))
    j = 0
    for i in range(50000):
    	if i%5000 == 0:
            j += 1
            print i
            print j
    	y[i][j-1] = 1
    y = y.T
    return x, y

def gd_vanilla(x, y, init_w, alpha, max_iter):
    EPS = 1e-5   #EPS = 10**(-5)
    prev_w = init_w-10*EPS
    w = init_w.copy()
    iter  = 0
    succ = list()
    i_list = list()
    M = loadmat("mnist_all.mat")

    while norm(w - prev_w) >  EPS and iter < max_iter:

        prev_w = w.copy()
        w -= alpha*gradient(x,y,w)

        if iter % 5 == 0:
            count = 0
            for i in range(10):
        		for j in range(100):
        		    test = hstack((1,M["test"+str(i)][j].T))   #originally M["train5"][148:149].T
        		    if argmax(f(test,w)) == i:
        		    	count += 1
            succ.append(count/10.0)
            i_list.append(iter)

        if iter % 500 == 0:
            print "Iter", iter
            print "Gradient: ", gradient(x, y, w), "\n"

        iter += 1

    plt.plot(i_list,succ)
    plt.show()

    return w

def part4(init_w,alpha,max_iter):
	x, y = input()
	gd_vanilla(x, y, init_w, alpha, max_iter)
	return
#Part 5

def gd_momentum(x, y, init_w, alpha, max_iter):
    EPS = 1e-5   #EPS = 10**(-5)
    prev_w = init_w-10*EPS
    w = init_w.copy()
    iter  = 0
    succ = list()
    i_list = list()
    M = loadmat("mnist_all.mat")

    while norm(w - prev_w) >  EPS and iter < max_iter:

        prev_w = w.copy()
        v = gamma*v + gradient(x,y,w)
        w -= v

        if iter % 5 == 0:
            count = 0
    	    for i in range(10):
    			for j in range(100):
				    test = hstack((1,M["test"+str(i)][j].T))   #originally M["train5"][148:149].T
				    if argmax(f(test, w)) == i:
				    	count += 1
            succ.append(count/10.0)
            i_list.append(iter)

        if iter % 500 == 0:
            print "Iter", iter
            print "Gradient: ", gradient(x, y, w), "\n"

        iter += 1

    plt.plot(i_list,succ)
    plt.show()

    return w

def part5(init_w,alpha,max_iter):
	x, y = input()
	gd_momentum(x, y, init_w, alpha, max_iter)
	return

#get the index at which the output is the largest
def deriv_multilayer(W0, b0, W1, b1, x, L0, L1, y, y_):
    '''Incomplete function for computing the gradient of the cross-entropy
    cost function w.r.t the parameters of a neural network'''
    dCdL1 =  y - y_
    dCdW1 =  dot(L0, dCdL1.T )

    dCdL0 = dot(W1, dCdL1.T)
    dL0dW0 = dot(x, (1-tanh_layer(x,w0,b0)**2).T )
    dCdW0 = dCdL0 * dL0dW0

    return dCdW1, dCdW0

def tanh_layer(y, W, b):
    '''Return the output of a tanh layer for the input matrix y. y
    is an NxM matrix where N is the number of inputs for a single case, and M
    is the number of cases'''
    return tanh(dot(W.T, y)+b)

def forward(x, W0, b0, W1, b1):
    L0 = tanh_layer(x, W0, b0)
    L1 = dot(W1.T, L0) + b1
    output = softmax(L1)
    return L0, L1, output

def multilayer(digit, ind): #which digit's picture, ind is tuple of indices
    #Load sample weights for the multilayer neural network
    snapshot = cPickle.load(open("snapshot50.pkl"))
    W0 = snapshot["W0"]
    b0 = snapshot["b0"].reshape((300,1))
    W1 = snapshot["W1"]
    b1 = snapshot["b1"].reshape((10,1))

    #Load one example from the training set, and run it through the
    #neural network
    M = loadmat("mnist_all.mat")
    x = M["train"+str(digit)][ind].T   #originally M["train5"][148:149].T
    x = x/255.0
    L0, L1, output = forward(x, W0, b0, W1, b1)
    return argmax(output)


################################################################################
#Code for displaying a feature from the weight matrix mW
#fig = figure(1)
#ax = fig.gca()
#heatmap = ax.imshow(mW[:,50].reshape((28,28)), cmap = cm.coolwarm)
#fig.colorbar(heatmap, shrink = 0.5, aspect=5)
#show()
################################################################################

def main():
	#part1()
    iw = np.zeros((785,10))
    # x(785*50000), w.T(10*785), y(10*50000)
    part4(iw,0.001,3000)

main()
