# -*- coding: utf-8 -*-
# Machine Learning Online Class - Exercise 1: Multiple Linear Regression
import numpy as np
import matplotlib.pyplot as plt

# Support functions
def feature_normalise(x):
    X_norm = x
    mu = np.mat(np.zeros((1,x.shape[1])))
    sigma = np.mat(np.zeros((1, x.shape[1])))
    for i in range(x.shape[1]):
        mu[0,i] = np.mean(x[:,i], axis=0)
        sigma[0,i] = np.std(x[:,i], axis=0, ddof=1)
    X_norm = np.divide((x - mu),sigma)
    return X_norm, mu, sigma

def plot_data(x,y):
    plt.figure(figsize=(10,6), dpi=200)
    plt.hold(True)
    plt.plot(x,y, 'rx', markersize=10,  label='Training Data')
    plt.ylabel('Profit in $10,000s')
    plt.xlabel('Population of City in 10,000s')
    
def compute_cost_multi(x,y,theta):
    m=len(y)
    cost= np.sum(np.power((x*theta-y),2))/(2*m)
    return cost
    
def gradient_descent_multi(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros((num_iters, 1))
    for i in xrange(num_iters):
        theta = theta - (alpha/m)*(X.T*(X*theta-y))
        J_history[i]=compute_cost_multi(X, y, theta)
    return theta, J_history
    
# Part 1: Feature Normalization
print 'Loading data ...'

# Load data
data =np.matrix(np.loadtxt('ex1data2.txt', delimiter=','))
X = data[:,0:2]
y = data[:,2]
m=len(y)


# Print out some data point
print 'First 10 examples from the dataset: '
print np.array_str(np.array(
np.c_[X[0:10,:], y[0:10,]]),precision=3)

# Scale features and set them to zero mean
print 'Normalizing Features...'
X, mu, sigma = feature_normalise(X)



# Add intercept term to x
X = np.c_[np.ones((m,1)), X]
print (X[0:10,:])
print mu
print sigma

# Part 2: Gradient Descent
print 'Running gradient descent...'
alpha = 0.01
num_iters=400
# initialise Theta and Run Gradient Descent
theta = np.mat(np.zeros((3,1)))
theta[0,0]=1
theta[1,0]=5
theta[2,0]=0.2
theta, J_history = gradient_descent_multi(X, y, theta, alpha, num_iters)


# Plot the convergence graph
plt.figure(figsize=(10,6), dpi=200)
plt.plot(np.arange(1,len(J_history)+1), J_history,'green', lw=2)
plt.ylabel('Cost J')
plt.xlabel('Number of iterations')
plt.show()

# Display the gradient descent's result
print ' Theta computed from gradient descent:'
print np.array_str(np.array(theta.T), precision=3)

# Estimate the price of a 1650 sq ft, 3br house
x_input=np.divide( np.subtract(np.mat([1, 1650, 3]),np.mat(np.c_[0,mu])),
                  np.c_[1,sigma])
price = x_input*np.mat(theta)
print 'Predicted price of a 1650 sq-ft, 3 br house:' + \
' {:.5e}'.format(price[0,0])

