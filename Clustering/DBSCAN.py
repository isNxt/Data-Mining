# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
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
params = np.arange(0.1, 2, 0.1)
print(params)
for param in params:
    algorithm = DBSCAN(eps=param)
    algorithm.fit(X)
    y_pred = algorithm.labels_.astype(np.int)
    s = adjusted_rand_score(y, y_pred)
    glass_score.append(s)

print('glass_score', glass_score)


# draw score pic---------------------------------------
plt.figure(figsize=(6, 4), dpi=120)
plt.grid()
plt.xlabel('eps for DBSCAN')
plt.xticks(params)
plt.plot(params, glass_score, label='glass_score', color='g')
plt.legend()
plt.title("glass DBSCAN score")

plt.savefig("img/DBSCAN.png")

plt.show()
