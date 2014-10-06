# Machine Learning using Gradient Descent 
# Linear Regression

from numpy import *

A = array([[1,1,1], 
           [0,1,3]])

B = array([[2,0,4], 
           [3,4,4]])

C = array([[0],[0]])
print A
print '\n'
print B
print '\n'

print C
print '\n'
d = reshape(A[:,2], (-1,1))
print  power(mat(d),2)
print C-d



# a = ones((4,1), dtype=int)
# 
# b = ones((4,1), dtype=int)*2
# print A**2

# c= hstack((a,b))
# print append(a,b,axis=0).flatten()
# # print c_[a,c]

# print ones((1,10))