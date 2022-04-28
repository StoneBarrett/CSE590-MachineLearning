# Stone Barrett
# CSE 590: Machine Learning
# Homework 4

# Importing libraries
from enum import auto
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mglearn
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Reading datasets
xTrainSet = pd.read_csv("./Homework 4/dataset/dataset/X_train.csv")
xTestSet = pd.read_csv("./Homework 4/dataset/dataset/X_test.csv")
yTrainSet = pd.read_csv("./Homework 4/dataset/dataset/y_train.csv")
yTestSet = pd.read_csv("./Homework 4/dataset/dataset/y_test.csv")

# Converting to NumPy array
xTrain = np.array(xTrainSet)
xTest = np.array(xTestSet)
yTrain = np.array(yTrainSet)
yTest = np.array(yTestSet)

# Looking at shape of data
print("xtrain shape:", xTrain.shape)
print("ytrain shape:", yTrain.shape)
print("xtest shape:", xTest.shape)
print("ytest shape:", yTest.shape)

# Different preprocessing options
# StandardScaler
sscaler = StandardScaler()
sscaler.fit(xTrain)
xTrainSScaled = sscaler.transform(xTrain)
xTestSScaled = sscaler.transform(xTest)

# RobustScaler
rscaler = RobustScaler()
rscaler.fit(xTrain)
xTrainRScaled = rscaler.transform(xTrain)
xTestRScaled = rscaler.transform(xTest)

# MinMaxScaler
mscaler = MinMaxScaler()
mscaler.fit(xTrain)
xTrainMScaled = mscaler.transform(xTrain)
xTestMScaled = mscaler.transform(xTest)

# Different classification methods
# Kernel Support Vector Machine with no preprocessing
def KSVM():
    # svm = SVC(kernel='rbf', C=10, gamma=.1).fit(xTrain, yTrain)
    # mglearn.plots.plot_2d_separator(svm, xTrain, eps=.5)
    # mglearn.discrete_scatter(xTrain[:, 0], xTrain[:, 1], yTrain)
    # # plot support vectors
    # sv = svm.support_vectors_
    # # class labels of SV are given by the sign of the dual coefficients
    # sv_labels = svm.dual_coef_.ravel() > 0
    # mglearn.discrete_scatter(sv[:, 0], sv[:, 1], sv_labels, s=15, markeredgewidth=3)
    # plt.xlabel("Feature 0")
    # plt.ylabel("Feature 1")

    svc = SVC(C=10)

    # Cross validation 
    scores = cross_val_score(svc, xTrain, yTrain, cv=4)
    print("Cross-validation scores: {}".format(scores))
    print("Average cross-validation score: {}".format(scores.mean()))

    svc.fit(xTrain, yTrain)
    print("Accuracy on training set: {:.2f}".format(svc.score(xTrain, yTrain)))
    print("Accuracy on test set: {:.2f}".format(svc.score(xTest, yTest)))
    # SVC.predict_proba(xTrain)

# Kernel Support Vector Machine with standard scaling
def standardKSVM():
    svc = SVC(C=10)

    # scores = cross_val_score(svc, xTrainSScaled, yTrain, cv=4)
    # print("Cross-validation scores: {}".format(scores))
    # print("Average cross-validation score: {}".format(scores.mean()))

    svc.fit(xTrainSScaled, yTrain)
    print("Accuracy on training set: {:.2f}".format(svc.score(xTrainSScaled, yTrain)))
    print("Accuracy on test set: {:.2f}".format(svc.score(xTestSScaled, yTest)))

# Kernel Support Vector Machine with robust scaling
def robustKSVM():
    svc = SVC(C=10, probability=True)

    # scores = cross_val_score(svc, xTrainRScaled, yTrain, cv=4)
    # print("Cross-validation scores: {}".format(scores))
    # print("Average cross-validation score: {}".format(scores.mean()))

    svc.fit(xTrainRScaled, yTrain)
    print("Accuracy on training set: {:.2f}".format(svc.score(xTrainRScaled, yTrain)))
    print("Accuracy on test set: {:.2f}".format(svc.score(xTestRScaled, yTest)))
    
    SVC.predict_proba(xTestRScaled[:50])

# Kernel Support Vector Machine with min max scaling
def minmaxKSVM():
    svc = SVC(C=10)

    # scores = cross_val_score(svc, xTrainMScaled, yTrain, cv=4)
    # print("Cross-validation scores: {}".format(scores))
    # print("Average cross-validation score: {}".format(scores.mean()))

    svc.fit(xTrainMScaled, yTrain)
    print("Accuracy on training set: {:.2f}".format(svc.score(xTrainMScaled, yTrain)))
    print("Accuracy on test set: {:.2f}".format(svc.score(xTestMScaled, yTest)))

# Multilayered Perceptron with no preprocessing
def MLPs():
    mlp= MLPClassifier(alpha=.1, random_state=1)

    scores = cross_val_score(mlp, xTrain, yTrain, cv=4)
    print("Cross-validation scores: {}".format(scores))
    print("Average cross-validation score: {}".format(scores.mean()))

    mlp.fit(xTrain, yTrain)
    print("Accuracy on training set: {:.2f}".format(mlp.score(xTrain, yTrain)))
    print("Accuracy on test set: {:.2f}".format(mlp.score(xTest, yTest)))

# Multilayered Perceptron with standard scaling
def standardMLP():
    mlp = MLPClassifier(alpha=.1, random_state=1)

    # scores = cross_val_score(mlp, xTrainSScaled, yTrain, cv=4)
    # print("Cross-validation scores: {}".format(scores))
    # print("Average cross-validation score: {}".format(scores.mean()))
    
    mlp.fit(xTrainSScaled, yTrain)
    print("Accuracy on training set: {:.2f}".format(mlp.score(xTrainSScaled, yTrain)))
    print("Accuracy on test set: {:.2f}".format(mlp.score(xTestSScaled, yTest)))

# Multilayered Perceptron with robust scaling
def robustMLP():
    mlp = MLPClassifier(alpha=.1, random_state=1)

    # scores = cross_val_score(mlp, xTrainRScaled, yTrain, cv=4)
    # print("Cross-validation scores: {}".format(scores))
    # print("Average cross-validation score: {}".format(scores.mean()))

    mlp.fit(xTrainRScaled, yTrain)
    print("Accuracy on training set: {:.2f}".format(mlp.score(xTrainRScaled, yTrain)))
    print("Accuracy on test set: {:.2f}".format(mlp.score(xTestRScaled, yTest)))

# Multilayered Perceptron with min max scaling
def minmaxMLP():
    mlp = MLPClassifier(alpha=.1, random_state=1)

    # scores = cross_val_score(mlp, xTrainMScaled, yTrain, cv=4)
    # print("Cross-validation scores: {}".format(scores))
    # print("Average cross-validation score: {}".format(scores.mean()))

    mlp.fit(xTrainMScaled, yTrain)
    print("Accuracy on training set: {:.2f}".format(mlp.score(xTrainMScaled, yTrain)))
    print("Accuracy on test set: {:.2f}".format(mlp.score(xTestMScaled, yTest)))

# Testing
# Problem 1
# KSVM()
# standardKSVM()
# robustKSVM()
# minmaxKSVM()

# Problem 2
# MLPs()
# standardMLP()
# robustMLP()
# minmaxMLP()