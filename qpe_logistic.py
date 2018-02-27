#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
Logistic Regression 3-class Classifier
=========================================================

"""

import csv
import numpy as np
from sklearn import linear_model


reader = csv.reader(open("train.csv", "rb"), delimiter=",")
x = list(reader)
train_data = np.array(x).astype("float")
reader = csv.reader(open("test.csv", "rb"), delimiter=",")
x = list(reader)
test_data = np.array(x).astype("float")

X_train = train_data[:, :-1]
Y_train = train_data[:, train_data.shape[1] - 1]
h = .02  # step size in the mesh
print X_train.shape
logreg = linear_model.LogisticRegression(C=1e5)

# we create an instance of Neighbours Classifier and fit the data.
logreg.fit(X_train, Y_train)

X_test = test_data[:, :-1]
Y_test = test_data[:, test_data.shape[1] - 1]
Z = logreg.predict(X_test)
correct = 0
for i in range(0, len(Z)):
    if Z[i] == Y_test[i]:
        correct += 1
print "accuracy = " + str(float(correct) / float(len(Z)))
