# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 14:30:42 2014

@author: junhao
"""
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

df = pd.read_csv('iris.data',delimiter=',', header=None)
df = df.dropna()
print df.head()
c= df.iloc[:,4]

f = pd.Categorical.from_array(c)
print f.labels

model=TSNE(n_components=2, init='random', early_exaggeration=4, learning_rate=200, random_state=1)
xy= model.fit_transform(df.iloc[:,0:3])


plt.scatter(xy[:,0],xy[:,1], c=f.labels)
ax.legend()
plt.show()

print model.get_params()