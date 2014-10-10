# -*- coding: utf-8 -*-
import numpy as np
mu = np.zeros((1,4))
print mu

print mu[0,1]

a = mat([[1,2,3],[4,5,6],[1,1,10]])
print a
m = np.mean(a, axis=0)
print type(m)

norm = a-m
print norm