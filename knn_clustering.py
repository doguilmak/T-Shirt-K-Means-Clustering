# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 16:16:03 2020

@author: doguilmak

https://scikit-learn.org/stable/modules/clustering.html
https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1

dataset: Dataset created as average measures(S-M-L) from myself. 

"""
#%%
# Libraries 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

#%%
# Data Preprocessing

start = time.time()
data = pd.read_csv('dataset.csv')
print(data.info()) # Looking for the missing values
print(data.describe().T)
print(data.isnull().sum())

X = data.values
plt.figure(figsize=(12, 12))
plt.xlabel('Weight')
plt.ylabel('Width x Height as m\u00b2')
x = data.iloc[:, 0].values
y = data.iloc[:, 1].values
plt.scatter(x, y, s=2)
plt.show()

#%%
# K-Means

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, init='k-means++') # The number of clusters.(default=8) / k-means++, random, ndarray
kmeans.fit(X)

print("K-Means Cluster Centers:\n",kmeans.cluster_centers_) # Center Coordinates of the Clusters on the X and Y Axes

# Additional Visualization(Showing Cluster Centers on Graph)
kmeans = KMeans (n_clusters = 3, init='k-means++', random_state= 123)
Y_pred= kmeans.fit_predict(X)
print("\nThe Cluster Numbers From Which The Data Were Found:\n", Y_pred)  

#%%
# Specifying and Visualizing Cluster Centers

plt.figure(figsize=(12, 12))
sns.set_style('whitegrid')
plt.plot(X[:,0], X[:,1], 'bo', color='blue', marker='o', markerfacecolor='black', markersize=3)
plt.plot(kmeans.cluster_centers_[0][0], kmeans.cluster_centers_[0][1], 'bo', color='red', marker='o', markerfacecolor='black', markersize=10)
plt.plot(kmeans.cluster_centers_[1][0], kmeans.cluster_centers_[1][1], 'bo', color='yellow', marker='o', markerfacecolor='black', markersize=10)
plt.plot(kmeans.cluster_centers_[2][0], kmeans.cluster_centers_[2][1], 'bo', color='blue', marker='o', markerfacecolor='black', markersize=10)

# Specifying Set Elements
plt.scatter(X[Y_pred==0, 0], X[Y_pred==0, 1], s=30, c='red')
plt.scatter(X[Y_pred==1, 0], X[Y_pred==1, 1], s=30, c='yellow')
plt.scatter(X[Y_pred==2, 0], X[Y_pred==2, 1], s=30, c='blue')
plt.title('KMeans')
plt.xlabel('Weight')
plt.ylabel('Width x Height as m\u00b2')
plt.show()

# Predicting Cluster
sample_test=np.array([0.149632272, 0.435760593])  # Weight and m²
second_test=sample_test.reshape(1, -1)
if kmeans.predict(second_test) == 0:
    print("Model predicted as medium size. Class: ", kmeans.predict(second_test))
elif kmeans.predict(second_test) == 1:
    print("Model predicted as large size. Class: ", kmeans.predict(second_test))
else:
    print("Model predicted as small size. Class: ", kmeans.predict(second_test))

#%%
# 10 Clusters are created and K-Means Success Values ​​(WCSS) are compared

results = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=123)
    kmeans.fit(X)
    results.append(kmeans.inertia_) # Success value is calculated and added to results (WCSS)

# Visualizing WCSS
plt.figure(figsize=(10, 6))
plt.title("Comparing Success of K-Means(WCSS)")
plt.xlabel("Number of Clustures")
plt.ylabel("WCSS Values")
plt.plot(range(1, 11), results)
plt.show()

end = time.time()
cal_time = end - start
print("\nTook {} seconds to cluster objects.".format(cal_time))
