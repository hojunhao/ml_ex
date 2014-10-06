# Machine Learning Online Class - Exercise 1: Linear Regression

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Support functions
def warmup_exercise():
    return np.eye(5)

def plot_data(x,y):
    plt.figure(figsize=(10,6), dpi=200)
    plt.hold(True)
    plt.plot(x,y, 'rx', markersize=10,  label='Training Data')
    plt.ylabel('Profit in $10,000s')
    plt.xlabel('Population of City in 10,000s')
    
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
    
# Part 1: Basic Function
print 'Running warmUpExercise ...'
print '5x5 Identity Matrix:\n'
print warmup_exercise()

# Part 2: Plotting
print 'Plotting Data ...'
data =np.matrix(np.loadtxt('ex1data1.txt', delimiter=','))
X = data[:,0]
y = data[:,1]
m=len(data)
plot_data(X,y)
plt.show()
plot_data(X,y)

# Part 3: Gradient descent
print 'Running Gradient Descent ...'
X = np.c_[np.ones((m,1)), data[:,0]]
theta = np.mat(np.zeros((2,1)))

# Gradient descent settings
iterations = 1500
alpha = 0.01

# Compute and display initial cost
print compute_cost(X, y, theta)

# Run Gradient Descent
theta, J_history = gradient_descent(X, y, theta, alpha, iterations)

# Print results
print theta
print 'Theta found by gradient descent: {0:.3e} {1:.3e}'.format(theta[0,0], theta[1,0])

# Plot the linear fit
plt.plot(X[:,1], X*theta, '-', label='Linear Regression')
plt.legend()
plt.show()

# Predict values for population of sizes
# 35,000 and 70,000
predict1 = np.mat([1, 3.5])*theta
predict2 = np.mat([1, 7])*theta

print 'For population = 35,000, prediction {0:.3e}'.format(predict1[0,0]*1000 )
print 'For population = 70,000, prediction {0:.3e}'.format(predict2[0,0]*1000 )

# Part 4: Visualizing J(theta_0, theta_1)
print 'Visualizing J(theta_0, theta_1) ...'

# Grid over which we will calculate J
theta0_vals = np.linspace(-10,10,100)
theta1_vals = np.linspace(-1,4,100)
T0, T1 =np.meshgrid(theta0_vals,theta1_vals)

# Initialise J_vals to a matrix of 0's
J_vals = np.zeros((len(theta0_vals),len(theta1_vals)))

# Fill Up J_vals
for i in xrange(len(theta0_vals)):
    for j in xrange(len(theta1_vals)):
        t = np.mat([theta0_vals[i], theta1_vals[j]]).T
        J_vals[i,j] = compute_cost(X, y, t)

# Plot surface 
fig = plt.figure(figsize=(10,6), dpi=100)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(T0, T1, J_vals.T, rstride=1, cstride=1, cmap=cm.Accent,
        linewidth=0.3, antialiased=True)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()   

# Plot contours
fig = plt.figure(figsize=(10,6), dpi=100)
CS = plt.contour(T0,T1,J_vals.T, levels=np.logspace(-2,3,20))
plt.plot(theta[0,0], theta[1,0], 'rx')
plt.xlabel(r'$\theta_0$')
plt.ylabel(r'$\theta_1$')
plt.show()



 