# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score

np.set_printoptions(threshold=np.inf)

# read glass data-----------------------------------
file = open('./data/glass.data')
next(file)
X = []
y = []
for line in file.readlines():
    curLine = line.strip().split(", ")
    X.append([float(i) for i in curLine[0:-1]])
    y.append(curLine[-1].strip('.'))


# iterate over classifiers-------------------------------------------
glass_score = []
params = range(1, 19, 1)
for param in params:
    algorithm = KMeans(n_clusters=param)
    algorithm.fit(X)
    y_pred = algorithm.predict(X)
    s = adjusted_rand_score(y, y_pred)
    glass_score.append(s)
print('glass_score', glass_score)

# draw score pic---------------------------------------
plt.figure(figsize=(6, 4), dpi=120)
plt.grid()
plt.xlabel('n_clusters for KMeans')
plt.xticks(params)
plt.plot(params, glass_score, label='glass_score', color='g')
plt.legend()
plt.title("glass KMeans score")

plt.savefig("img/KMeans.png")

plt.show()
