#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
Logistic Regression 2-class Classifier
=========================================================

"""

import csv
import numpy as np
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt


# Load the train and test data

reader = csv.reader(open("micro_dataset2.csv", "rb"), delimiter=",")
micro_benchmark_dataset = np.array(list(reader)).astype("float")

X = micro_benchmark_dataset[:, :-1]
X = np.log10(X)
y = micro_benchmark_dataset[:, micro_benchmark_dataset.shape[1] - 1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print X_test, y_test

# Create an instance of Neighbours Classifier and fit the data.
h = .02
clf = linear_model.LogisticRegression(C=1e5)
scores = cross_val_score(clf, X, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(clf, X, y, cv=5)
print metrics.accuracy_score(y, predicted)

clf.fit(X_train, y_train)

# and plot the result
plt.figure(1, figsize=(4, 3))
plt.clf()
plt.scatter(X.ravel(), y, color='black', zorder=20)
X_test = np.linspace(0, np.amax(X), 300)


def model(x):
    return 1 / (1 + np.exp(-x))


loss = model(X_test * clf.coef_ + clf.intercept_).ravel()
plt.plot(X_test, loss + 1, color='red', linewidth=3)
plt.show()

