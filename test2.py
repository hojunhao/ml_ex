# -*- coding: utf-8 -*-
"""
Created on Sat Oct 04 20:17:37 2014

@author: junhao
"""

import numpy as np
import matplotlib.pyplot as plt

def compute_cost(x,y,theta):
    m=len(y)
    cost= np.sum(np.power((x*theta-y),2))/(2*m)
    return cost
    
def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros((num_iters, 1))
    for i in xrange(num_iters):
        theta = theta - (alpha/m)*(X.T*(X*theta-y))
        J_history[i]=compute_cost(X, y, theta)
    return theta, J_history
    
data =np.matrix(np.loadtxt('ex1data1.txt', delimiter=','))
print data[1:10,:]
X = data[:,0]
y = data[:,1]
m=len(data)


X = np.c_[np.ones((m,1)), data[:,0]]
#print 'X'
#print X[1:20,:]
#
#print 'y'
#print y[1:10,]


theta = np.mat(np.zeros((2,1)))
print compute_cost(X,y,theta)

#Gradient Descent
alpha = 0.01
num_iters = 1500

theta, J_history =  gradient_descent(X, y, theta, alpha, num_iters)
print theta
print J_history[1:100:10]

predict1 = np.mat([1, 3.5])*theta
predict2 = np.mat([1, 7])*theta

print 'For population = 35,000, prediction {0:.4e}'.format(sum(predict1)*1000 )
print 'For population = 70,000, prediction {0:.4e}'.format(sum(predict2)*1000 )
