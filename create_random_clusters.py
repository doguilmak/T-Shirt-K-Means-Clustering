# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 14:57:44 2021

@author: doguilmak

"""
#%%
# Libraries

import numpy as np
import pandas as pd

#%%
# Generating random values for the clusters.

size=1

np.random.seed(2)
x1 = np.random.normal(0.125173913, 0.015, 1500)
y1 = np.random.normal(0.320040134, 0.015, 1500)
df1 = pd.DataFrame({"Weight" : x1, "measurements_m2" : y1})
df1.to_csv("submission1.csv", index=False)

x2 = np.random.normal(0.136763333, 0.015, 1500)
y2 = np.random.normal(0.379373333, 0.015, 1500)
df2 = pd.DataFrame({"Weight" : x2, "measurements_m2" : y2})
df2.to_csv("submission2.csv", index=False)

x3 = np.random.normal(0.146681063, 0.015, 1500)
y3 = np.random.normal(0.4300299, 0.015, 1500)
df3 = pd.DataFrame({"Weight" : x3, "measurements_m2" : y3})
df3.to_csv("submission3.csv", index=False)

#%%
# Plotting
"""
import matplotlib.pyplot as plt

X=[]
Y=[]

X.append(x1)
X.append(x2)
X.append(x3)

Y.append(y1)
Y.append(y2)
Y.append(y3)

plt.scatter(X, Y)
plt.show()

plt.figure(figsize=(9, 9))
plt.xlabel('Weight')
plt.ylabel('Width x Height as m\u00b2')
plt.scatter(x1, y1, s=size)
plt.scatter(x2, y2, s=size)
plt.scatter(x3, y3, s=size)
plt.show()
"""