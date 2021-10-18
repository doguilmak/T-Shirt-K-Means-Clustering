# Clustering T-Shirt Size using K-Means Clustering Algorithm

## Problem Statement

The goal of this algorithm is to find ideal number of clusters and group them with K-Means algorithm. The aim of this study is to cluster the t-shirts produced with the same type of fabric according to their weight and area (m²) values.

## Dataset

Data set is created by myself. Values ​​are generated to be completely random. You can find the code prepared to create the data set in the file named **create_random_clusters.py**.

> You can increase or decrease variables in create_random_clusters.py file to create your own data set.

## Methodology

For understanding the methodology you are free to visit the [K-Means clustering scikit-learn](https://scikit-learn.org/stable/modules/clustering.html)  website.

## Analysis

| # | Column | Non-Null Count | Dtype |
|--|--|--|--|
| 0 | Weight | 4500 non-null | float64
| 1 | measurements_m2 | 4500 non-null | float64

### Description

 #| Weight | measurements_m2 |
|--|--|--|
| count | 4500.000000 | 4500.000000 
| mean |  0.135717 | 0.376382
| std | 0.017545 | 0.047372 
| min | 0.075271 | 0.272992 
| 25% | 0.123695 | 0.330740 
| 50% | 0.135462 | 0.379124 
| 75% | 0.147488 | 0.419964 
| max | 0.198764 | 0.475819

> Took 2.8759424686431885 seconds to cluster objects.

## How to Run Code

Before running the code make sure that you have these libraries:

 - numpy 
 - pandas 
 - matplotlib
 - seaborn
 - time
 - sklearn
    
## Contact Me

If you have something to say to me please contact me: 

 - Twitter: [Doguilmak](https://twitter.com/Doguilmak)
 - Mail address: doguilmak@gmail.com
