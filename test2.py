# -*- coding: utf-8 -*-
"""
Created on Sat Oct 04 20:17:37 2014

@author: junhao
"""

import numpy as np
import matplotlib.pyplot as plt

a = np.mat([[1.1,2.2,3.3], [3.1,4.1,5.1]])
print a 
print '\n'
print [i.tolist() for i in a]
print '\n'
print [i[0].ravel() for i in a]
print '\n'

print ["1:{0} 2:{1} 3:{2}".format(*i.tolist()[0]) 
for i in a] 


print '\n\n======= Part B: ======='
b = np.mat([1,2,3])
print b
print b.tolist()[0]
print "{0} {1} {2}".format(*b.tolist()[0])