# -*- coding: utf-8 -*-
import time

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split

le = LabelEncoder()

# read train data-----------------------------------
file = open('./data/segment-train.txt')
X = []
y = []

for line in file.readlines():
    if line != '\n':
        line = line.strip('\n')
        if line[0] == '%' or line[0] == '@':
            continue
        curLine = line.strip().split(",")
        X.append([float(i) for i in curLine[0:-1]])
        y.append(curLine[-1])

file = open('./data/segment-test.txt')

for line in file.readlines():
    if line != '\n':
        line = line.strip('\n')
        if line[0] == '%' or line[0] == '@':
            continue
        curLine = line.strip().split(",")
        X.append([float(i) for i in curLine[0:-1]])
        y.append(curLine[-1])

# X = StandardScaler().fit_transform(X)
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=20)


# DecisionTreeClassifier------------------------------

params = range(8, 24, 2)
train_score = []
test_score = []
elapsed = []

for param in params:
    start = time.time()
    model = DecisionTreeClassifier(max_depth=param)
    model.fit(X_train, y_train)
    train_s = metrics.accuracy_score(y_train, model.predict(X_train))
    test_s = metrics.accuracy_score(y_test, model.predict(X_test))
    train_score.append(train_s)
    test_score.append(test_s)
    t = time.time() - start
    elapsed.append(t)

print('train_score', train_score)
print('test_score', test_score)
print('elapsed', elapsed)
plt.figure(figsize=(6, 4), dpi=120)
plt.grid()
plt.xlabel('max depth of decison tree of segment')
plt.plot(params, train_score, label='train_score')
plt.plot(params, test_score, label='test_score')

plt.legend()
plt.savefig("./img/segment-tree.png")


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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)

# DecisionTreeClassifier------------------------------
params = range(4, 20, 2)
train_score = []
test_score = []
elapsed = []
for param in params:
    start = time.time()
    model = DecisionTreeClassifier(max_depth=param)
    model.fit(X_train, y_train)
    train_s = metrics.accuracy_score(y_train, model.predict(X_train))
    test_s = metrics.accuracy_score(y_test, model.predict(X_test))
    train_score.append(train_s)
    test_score.append(test_s)
    t = time.time() - start
    elapsed.append(t)

print('train_score', train_score)
print('test_score', test_score)
print('elapsed', elapsed)

plt.figure(figsize=(6, 4), dpi=120)
plt.grid()
plt.xlabel('max depth of decison tree of house')
plt.plot(params, train_score, label='train_score')
plt.plot(params, test_score, label='test_score')
plt.legend()
plt.savefig("./img/house-tree.png")

plt.show()