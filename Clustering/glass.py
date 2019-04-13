# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
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
file = open('./data/glass.data')
next(file)
X = []
y = []
for line in file.readlines():
    curLine = line.strip().split(", ")
    X.append([float(i) for i in curLine[0:-1]])
    y.append(curLine[-1].strip('.'))

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
plt.bar(x, score, width=width, label='glass score', tick_label=names, fc='r')
plt.legend()
plt.title("glass cluster score")
plt.savefig("img/glass.png")

plt.show()
