# Stone Barrett
# CSE 590: Machine Learning
# Exam 1

from cgi import test
from random import random
import mglearn
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import time
from mglearn import tools
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import load_digits
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier

#######################################################################################################
##############       PROBLEM 3A         ###############################################################
#######################################################################################################

def problem3a():

    cal_housing = fetch_california_housing()
    X, y = cal_housing['data'], cal_housing['target']
    labels = cal_housing['feature_names']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=38)

    # Looking at shape of data
    # print("x_train shape:", X_train.shape)
    # print("y_train shape:", y_train.shape)
    # print("X_test shape:", X_test.shape)
    # print("y_test shape:", y_test.shape)

    lasso0001 = Lasso(alpha=.001, max_iter=100000).fit(X_train, y_train)
    lasso = Lasso().fit(X_train, y_train)
    lasso001 = Lasso(alpha=.01, max_iter=100000).fit(X_train, y_train)
    lasso00001 = Lasso(alpha=.0001, max_iter=100000).fit(X_train, y_train)

    training_accuracy = []
    testing_accuracy = []
    values = [.0001, .001, .01, 1]

    # Printing scores
    print("\nTraining set score: {:.2f}".format(lasso0001.score(X_train, y_train)))
    print("Testing set score {:.2f}".format(lasso0001.score(X_test, y_test)))
    print("Number of features used: {}".format(np.sum(lasso0001.coef_ != 0)))
    training_accuracy.append(lasso0001.score(X_train, y_train))
    testing_accuracy.append(lasso0001.score(X_test, y_test))
    print("\nTraining set score: {:.2f}".format(lasso.score(X_train, y_train)))
    print("Testing set score {:.2f}".format(lasso.score(X_test, y_test)))
    print("Number of features used: {}".format(np.sum(lasso.coef_ != 0)))
    training_accuracy.append(lasso.score(X_train, y_train))
    testing_accuracy.append(lasso.score(X_test, y_test))
    print("\nTraining set score: {:.2f}".format(lasso001.score(X_train, y_train)))
    print("Testing set score {:.2f}".format(lasso001.score(X_test, y_test)))
    print("Number of features used: {}".format(np.sum(lasso001.coef_ != 0)))
    training_accuracy.append(lasso001.score(X_train, y_train))
    testing_accuracy.append(lasso001.score(X_test, y_test))
    print("\nTraining set score: {:.2f}".format(lasso00001.score(X_train, y_train)))
    print("Testing set score {:.2f}".format(lasso00001.score(X_test, y_test)))
    print("Number of features used: {}".format(np.sum(lasso00001.coef_ != 0)))
    training_accuracy.append(lasso00001.score(X_train, y_train))
    testing_accuracy.append(lasso00001.score(X_test, y_test))

    training_accuracy.sort()
    testing_accuracy.sort()

    # Graphing
    plt.plot(lasso.coef_, 's', label="Lasso alpha=1")
    plt.plot(lasso001.coef_, '^', label="Lasso alpha=0.01")
    plt.plot(lasso00001.coef_, 'v', label="Lasso alpha=0.0001")
    plt.plot(lasso0001.coef_, 'o', label="Lasso alpha=.001")
    plt.xlabel("Coefficient index")
    plt.ylabel("Coefficient magnitude")
    plt.ylim(-25, 25)
    plt.legend()
    plt.show()

    x = 1
    if x == 1:
        plt.plot(values, training_accuracy, label="training accuracy")
        plt.plot(values, testing_accuracy, label="testing accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Alpha Value")
        plt.legend()
        plt.show()
    

#######################################################################################################
##############       PROBLEM 3B         ###############################################################
#######################################################################################################

def problem3b():
    cal_housing = fetch_california_housing()
    X, y = cal_housing['data'], cal_housing['target']
    labels = cal_housing['feature_names']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=38)

    training_accuracy = []
    testing_accuracy = []
    values = [1, 2, 3, 4, 5, 10, 20]

    tree20 = DecisionTreeRegressor(max_depth=20).fit(X_train, y_train)
    tree10 = DecisionTreeRegressor(max_depth=10).fit(X_train, y_train)
    tree5 = DecisionTreeRegressor(max_depth=5).fit(X_train, y_train)
    tree4 = DecisionTreeRegressor(max_depth=4).fit(X_train, y_train)
    tree3 = DecisionTreeRegressor(max_depth=3).fit(X_train, y_train)
    tree2 = DecisionTreeRegressor(max_depth=2).fit(X_train, y_train)
    tree1 = DecisionTreeRegressor(max_depth=1).fit(X_train, y_train)

    print("\nTraining set score: {:.2f}".format(tree20.score(X_train, y_train)))
    print("Testing set score {:.2f}".format(tree20.score(X_test, y_test)))
    training_accuracy.append(tree20.score(X_train, y_train))
    testing_accuracy.append(tree20.score(X_test, y_test))
    print("\nTraining set score: {:.2f}".format(tree10.score(X_train, y_train)))
    print("Testing set score {:.2f}".format(tree10.score(X_test, y_test)))
    training_accuracy.append(tree10.score(X_train, y_train))
    testing_accuracy.append(tree10.score(X_test, y_test))
    print("\nTraining set score: {:.2f}".format(tree5.score(X_train, y_train)))
    print("Testing set score {:.2f}".format(tree5.score(X_test, y_test)))
    training_accuracy.append(tree5.score(X_train, y_train))
    testing_accuracy.append(tree5.score(X_test, y_test))
    print("\nTraining set score: {:.2f}".format(tree4.score(X_train, y_train)))
    print("Testing set score {:.2f}".format(tree4.score(X_test, y_test)))
    training_accuracy.append(tree4.score(X_train, y_train))
    testing_accuracy.append(tree4.score(X_test, y_test))
    print("\nTraining set score: {:.2f}".format(tree3.score(X_train, y_train)))
    print("Testing set score {:.2f}".format(tree3.score(X_test, y_test)))
    training_accuracy.append(tree3.score(X_train, y_train))
    testing_accuracy.append(tree3.score(X_test, y_test))
    print("\nTraining set score: {:.2f}".format(tree2.score(X_train, y_train)))
    print("Testing set score {:.2f}".format(tree2.score(X_test, y_test)))
    training_accuracy.append(tree2.score(X_train, y_train))
    testing_accuracy.append(tree2.score(X_test, y_test))
    print("\nTraining set score: {:.2f}".format(tree1.score(X_train, y_train)))
    print("Testing set score {:.2f}".format(tree1.score(X_test, y_test)))
    training_accuracy.append(tree1.score(X_train, y_train))
    testing_accuracy.append(tree1.score(X_test, y_test))

    training_accuracy.sort()
    testing_accuracy.sort()

    plt.plot(values, training_accuracy, label="training accuracy")
    plt.plot(values, testing_accuracy, label="testing accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Max Depth (Pruning Parameter)")
    plt.legend()
    plt.show()


#######################################################################################################
##############       PROBLEM 4A         ###############################################################
#######################################################################################################

def problem4a():
    digits = load_digits(n_class=10, return_X_y=False, as_frame=False)
    X, y = digits['data'], digits['target']
    labels = digits['feature_names']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    forest = RandomForestClassifier(n_estimators=10000, random_state=0)
    #feature_names = [f"feature {i}" for i in range(X_train.shape[1])]
    feature_names = labels
    scores = cross_val_score(forest, X_train, y_train, cv=4)
    print("Cross-validation scores: {}".format(scores))
    print("Average cross-validation score: {}".format(scores.mean()))
    forest.fit(X_train, y_train)

    # Looking at shape of data
    # print("x_train shape:", X_train.shape)
    # print("y_train shape:", y_train.shape)
    # print("X_test shape:", X_test.shape)
    # print("y_test shape:", y_test.shape)

    print("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))
    print("Accuracy on testing set: {:.3f}".format(forest.score(X_test, y_test)))

    start_time = time.time()
    result = permutation_importance(forest, X_test, y_test, n_repeats=10, random_state=0, n_jobs=-1)
    elapsed_time = time.time() - start_time
    print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

    forest_importances = pd.Series(result.importances_mean, index=feature_names)
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
    ax.set_title("Feature importances using permutation on full model")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    plt.show()

    resultTranslated = [0, 0, -0.0002, 0, -0.0006, 0.0019, 0.0002, 0, 0, -0.002, -0.001, 0.0006, -0.001, 0.0039, 0, 0, 0, -0.0015, -0.001, 0.0026, -0.0015, 0.025, 0.001, 0, 0, -0.0015, 0.0099, 0.0048, -0.003, 0.00088, 0.0019, 0, 0, 0.004, 0.0019, -0.0026, 0.001, -0.0028, 0.0008, 0, 0, -0.0002, 0.01, 0.01, -0.0017, -0.00066, 0.00066, 0, 0, 0, 0.0015, 0.001, -0.0002, -1.1102, -0.001, 0, 0, 0, 0.00088, -0.0002, 0.0004, -0.001, -0.0008, 0]

    
    relevance = np.array(resultTranslated).reshape(8, 8)

    #print(relevance)
    #mglearn.tools.heatmap(relevance, xlabel='features', ylabel='features', xticklabels=range(1,9), yticklabels=range(1,9))

    # fig = px.imshow(relevance, color_continuous_scale=px.colors.sequential.Cividis_r)
    # fig.show()
    


#######################################################################################################
##############       TESTING            ###############################################################
#######################################################################################################

#problem3a()
problem3b()
#problem4a()
