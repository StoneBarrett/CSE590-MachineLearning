# Stone Barrett
# CSE 590: Machine Learning
# Homework 3
# Lectures 13-17

# Importing libraries
from random import Random
import time
import matplotlib.pyplot as plt
import mglearn
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Reading datasets
train_setX = pd.read_csv("./Homework 3/IMDB dataset/IMDB dataset/X_train.csv")
train_setY = pd.read_csv("./Homework 3/IMDB dataset/IMDB dataset/y_train.csv")
test_setX = pd.read_csv("./Homework 3/IMDB dataset/IMDB dataset/X_test.csv")
test_setY = pd.read_csv("./Homework 3/IMDB dataset/IMDB dataset/y_test.csv")

# Splitting into independent (x) and dependent (y) variables
# Insertting into NumPy arrays
x_train = np.array(train_setX[list(train_setX.columns)])
y_train = np.array(train_setY)
x_test = np.array(test_setX[list(test_setX.columns)])
y_test = np.array(test_setY)

# Fixing shape of data
# y_train.reshape(-1)
# y_test.reshape(-1)

# Looking at shape of data
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)

#######################################################################################################
##############       PROBLEM 1A - Multinomial Naive Bayes         #####################################
#######################################################################################################

def problem1a():
    clf = MultinomialNB()
    scores = cross_val_score(clf, x_train, y_train, cv=4)
    print("Cross-validation scores: {}".format(scores))
    print("Average cross-validation score: {}".format(scores.mean()))
    clf.fit(x_train, y_train)
    print("Accuracy on training set: {:.3f}".format(clf.score(x_train, y_train)))
    print("Accuracy on testing set: {:.3f}".format(clf.score(x_test, y_test)))


#######################################################################################################
##############       PROBLEM 1B - Random Forest         ###############################################
#######################################################################################################

def problem1b():
    forest = RandomForestClassifier(n_estimators=100, random_state=0)
    feature_names = [f"feature {i}" for i in range(x_train.shape[1])]
    scores = cross_val_score(forest, x_train, y_train, cv=4)
    print("Cross-validation scores: {}".format(scores))
    print("Average cross-validation score: {}".format(scores.mean()))
    forest.fit(x_train, y_train)

    # fig, axes = plt.subplots(2,3, figsize=(20,10))
    # for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
    #     ax.set_title("Tree {}".format(i))
    #     mglearn.plots.plot_tree_partition(x_train, y_train, tree, ax=ax)
    
    # mglearn.plots.plot_2d_separator(forest, x_train, fill=True, ax=axes[-1, -1], alpha=.4)
    # axes[-1, -1].set_title("Random Forest")
    # mglearn.discrete_scatter(x_train[:, 0], x_train[:, 1], y_train)

    print("Accuracy on training set: {:.3f}".format(forest.score(x_train, y_train)))
    print("Accuracy on testing set: {:.3f}".format(forest.score(x_test, y_test)))

    start_time = time.time()
    result = permutation_importance(forest, x_test, y_test, n_repeats=10, random_state=0, n_jobs=-1)
    elapsed_time = time.time() - start_time
    print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

    forest_importances = pd.Series(result.importances_mean, index=feature_names)
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
    ax.set_title("Feature importances using permutation on full model")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    plt.show()

    


#######################################################################################################
##############       PROBLEM 1C - Gradient Boosted Regression Trees         ###########################
#######################################################################################################

def problem1c():
    gbrt = GradientBoostingClassifier(random_state=0, max_depth=1, learning_rate=.01)
    scores = cross_val_score(gbrt, x_train, y_train, cv=4)
    print("Cross-validation scores: {}".format(scores))
    print("Average cross-validation score: {}".format(scores.mean()))
    gbrt.fit(x_train, y_train)
    print("Accuracy on training set: {:.3f}".format(gbrt.score(x_train, y_train)))
    print("Accuracy on testing set: {:.3f}".format(gbrt.score(x_test, y_test)))


#######################################################################################################
##############       Main         #####################################################################
#######################################################################################################

#problem1a()
problem1b()
#problem1c()