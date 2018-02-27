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


# Load the train and test data
reader = csv.reader(open("train.csv", "rb"), delimiter=",")
x = list(reader)
train_data = np.array(x).astype("float")
reader = csv.reader(open("test.csv", "rb"), delimiter=",")
x = list(reader)
test_data = np.array(x).astype("float")

# Divide data into features and target
X_train = train_data[:, :-1]
Y_train = train_data[:, train_data.shape[1] - 1]
X_test = test_data[:, :-1]
Y_test = test_data[:, test_data.shape[1] - 1]

# Remove columns with all zeros
X = np.concatenate((X_train, X_test), axis=0)
X = X[:, ~np.all(X == 0, axis=0)]
X_train = X[0:len(X_train), :]
X_test = X[len(X_train):len(X), :]

# Create an instance of Neighbours Classifier and fit the data.
h = .02
logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(X_train, Y_train)

# Predict with test data
Z = logreg.predict(X_test)
correct = 0
for i in range(0, len(Z)):
    if Z[i] == Y_test[i]:
        correct += 1
print "accuracy = " + str(float(correct) / float(len(Z)))
