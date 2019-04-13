# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score

np.set_printoptions(threshold=np.inf)


names = ["KMeans",
         "DBSCAN",  # default
         "AHC",
         "AP"]
clusters = [KMeans(),
            DBSCAN(),
            AgglomerativeClustering(),
            AffinityPropagation()]


# read data-----------------------------------
iris = datasets.load_iris()
X = iris.data[:, :]
y = iris.target

print('X:')
for x in X:
    print(x)
print('y:\n', y)


# iterate over classifiers-------------------------------------------
score = []
for name, algorithm in zip(names, clusters):
    algorithm.fit(X)
    print(name, ':\n', algorithm.get_params())
    if hasattr(algorithm, 'labels_'):
        y_pred = algorithm.labels_.astype(np.int)
    else:
        y_pred = algorithm.predict(X)
    print('y_pred:\n', y_pred.tolist())
    s = adjusted_rand_score(y, y_pred)
    score.append(s)


print('score', score)


# draw score pic---------------------------------------
x = list(range(len(names)))
total_width, n = 0.8, 1
width = total_width / n
plt.figure(figsize=(6, 4), dpi=120)
ymin = min(score)
ymax = max(score)
plt.ylim((max(0.0, ymin-(ymax-ymin)/2), ymax+(ymax-ymin)/2))
plt.bar(x, score, width=width, label='iris score', tick_label=names, fc='b')
plt.legend()
plt.title("iris cluster score")

plt.savefig("img/iris.png")
plt.show()
