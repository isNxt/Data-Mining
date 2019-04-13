# -*- coding: utf-8 -*-


import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

start = time.time()

le = LabelEncoder()

names = ["Nearest Neighbors",
         "Decision Tree",
         "Naive Bayes"]

classifiers = [
    KNeighborsClassifier(3),
    DecisionTreeClassifier(max_depth=5),
    GaussianNB()]


# read data-------------------------------------------------------------------------
file = open('./data/house-votes-84.data')
X = []
X_label = ['y', 'n', '?']
X_number = le.fit_transform(X_label)
y = []
y_label = ['democrat', 'republican']
y_number = le.fit_transform(y_label)
for line in file.readlines():
    if line != '\n':
        line = line.strip('\n')
        if line[0] == '%' or line[0] == '@':
            continue
        for i in range(len(y_label)):
            line = line.replace(y_label[i - 1], str(y_number[i - 1]))
        for i in range(len(X_label)):
            line = line.replace(X_label[i - 1], str(X_number[i - 1]))
        curLine = line.strip().split(",")
        X.append([int(i) for i in curLine[1:]])
        y.append(int(curLine[0]))

print('X:')
for x in X:
    print(x)
print('y:\n', y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)

# iterate over classifiers-------------------------------------------
score = []
elapsed = []
efficiency = []
data_time = time.time() - start
for name, clf in zip(names, classifiers):
    middle = time.time()
    clf.fit(X_train, y_train)
    s = clf.score(X_test, y_test)
    score.append(s)
    t = data_time + time.time() - middle
    elapsed.append(t)
    efficiency.append(s/t)

print('score', score)
print('elapsed', elapsed)


# draw score and time pic---------------------------------------
x = list(range(len(names)))
total_width, n = 1, 2
width = total_width / n
plt.figure(figsize=(15, 6), dpi=150)  # 画图之前首先设置figure对象，此函数相当于设置一块自定义大小的画布，使得后面的图形输出在这块规定了大小的画布上，其中参数figsize设置画布大小

plt.subplot(131)
ymin = min(score)
ymax = max(score)
plt.ylim((ymin-(ymax-ymin)/2, ymax+(ymax-ymin)/2))
plt.bar(x, score, width=width, label='score', tick_label=names, fc='b')
plt.legend()

plt.subplot(132)
ymin = min(elapsed)
ymax = max(elapsed)
plt.ylim((ymin-(ymax-ymin)/2, ymax+(ymax-ymin)/2))
plt.bar(x, elapsed, width=width, label='time', tick_label=names, fc='r')
plt.legend()

plt.subplot(133)
ymin = min(efficiency)
ymax = max(efficiency)
plt.ylim((ymin-(ymax-ymin)/2, ymax+(ymax-ymin)/2))
plt.bar(x, efficiency, width=width, label='efficiency', tick_label=names, fc='g')
plt.legend()

plt.savefig("./img/house.png")
plt.show()
