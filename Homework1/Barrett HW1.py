# Stone Barrett
# CSE 590: Machine Learning
# Homework 1

# Importing libraries
import matplotlib.pyplot as plt
from matplotlib import testing
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

# Reading datasets
train_set = pd.read_csv("./Homework 1/datasets_Hwk1/datasets_Hwk1/wine_train.csv")
test_set = pd.read_csv("./Homework 1/datasets_Hwk1/datasets_Hwk1/wine_test.csv")

# Splitting into independent (x) variables and dependent (y) variable
# Dropping ID column
# Inserting into NumPy arrays
x_train = np.array(train_set[list(train_set.columns[1:-1])])
y_train = np.array(train_set["quality"])
x_test = np.array(test_set[list(test_set.columns[1:-1])])
y_test = np.array(test_set["quality"])

# Looking at shape of data
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)

#######################################################################################################
##############       PROBLEM 1A         ###############################################################
#######################################################################################################

# # KNN

# # Needed for graphing
# training_accuracy = []
# testing_accuracy = []
# neighbors_tries = range(1, 51)

# # Main loop
# for k in neighbors_tries:
#     # Instantiating model and setting k value
#     neigh = KNeighborsRegressor(n_neighbors=k)

#     # Fitting model using training data
#     neigh.fit(x_train, y_train)

#     # To make predictions on the test set
#     print("\n---------------------------------\nTest set predictions for k =", k)
#     print("\n{}".format(neigh.predict(x_test)))

#     # Record scores
#     training_accuracy.append(neigh.score(x_train, y_train))
#     testing_accuracy.append(neigh.score(x_test, y_test))

#     # Printing Scores
#     print("\nScores for k =", k)
#     print("Training Data Score: ", neigh.score(x_train, y_train))
#     print("Testing Data Score: ", neigh.score(x_test, y_test))

# # Graphing
# plt.plot(neighbors_tries, training_accuracy, label="training accuracy")
# plt.plot(neighbors_tries, testing_accuracy, label="testing accuracy")
# plt.ylabel("Accuracy")
# plt.xlabel("k Value")
# plt.legend()
# plt.show()


#######################################################################################################
##############       PROBLEM 1B         ###############################################################
#######################################################################################################

# OLS

# print("\nScores for OLS")

# # Fitting to training data
lr = LinearRegression().fit(x_train, y_train)

# # Printing linear coefficients and intercept
# print("lr.coef_: {}".format(lr.coef_))
# print("lr.intercept_: {}".format(lr.intercept_))

# # Printing scores
# print("Training set score: {:.2f}".format(lr.score(x_train, y_train)))
# print("Testing set score: {:.2f}".format(lr.score(x_test, y_test)))

# Graphing


#######################################################################################################
##############       PROBLEM 1C         ###############################################################
#######################################################################################################

# Ridge

print("\nScores for Ridge")

# Fitting to training data
ridge = Ridge().fit(x_train, y_train)
ridge01 = Ridge().fit(x_train, y_train)
ridge10 = Ridge().fit(x_train, y_train)

# Printing scores
# print("Training set score for alpha = 1: {:.2f}".format(ridge.score(x_train, y_train)))
# print("Testing set score for alpha = 1: {:.2f}".format(ridge.score(x_test, y_test)))
# print("Training set score for alpha = .1: {:.2f}".format(ridge01.score(x_train, y_train)))
# print("Testing set score for alpha = .1: {:.2f}".format(ridge01.score(x_test, y_test)))
# print("Training set score for alpha = 10: {:.2f}".format(ridge10.score(x_train, y_train)))
# print("Testing set score for alpha = 10: {:.2f}".format(ridge10.score(x_test, y_test)))

# # Graphing
# plt.plot(ridge.coef_, 's', label="Ridge alpha=1")
# plt.plot(ridge01.coef_, '^', label="Ridge alpha=0.1")
# plt.plot(ridge10.coef_, 'v', label="Ridge alpha=10")
# plt.plot(lr.coef_, 'o', label="LinearRegression")
# plt.xlabel("Coefficient index")
# plt.ylabel("Coefficient magnitude")
# plt.hlines(0, 0, len(lr.coef_))
# plt.ylim(-25, 25)
# plt.legend()
# plt.show()

#######################################################################################################
##############       PROBLEM 1D         ###############################################################
#######################################################################################################

# LASSO

# Fitting to training dataset
lasso = Lasso().fit(x_train, y_train)
lasso001 = Lasso(alpha=.01, max_iter=100000).fit(x_train, y_train)
lasso00001 = Lasso(alpha=.0001, max_iter=100000).fit(x_train, y_train)

# Printing scores
print("\nTraining set score: {:.2f}".format(lasso.score(x_train, y_train)))
print("Testing set score {:.2f}".format(lasso.score(x_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso.coef_ != 0)))
print("\nTraining set score: {:.2f}".format(lasso001.score(x_train, y_train)))
print("Testing set score {:.2f}".format(lasso001.score(x_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso001.coef_ != 0)))
print("\nTraining set score: {:.2f}".format(lasso00001.score(x_train, y_train)))
print("Testing set score {:.2f}".format(lasso00001.score(x_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso00001.coef_ != 0)))

# Graphing
plt.plot(lasso.coef_, 's', label="Lasso alpha=1")
plt.plot(lasso001.coef_, '^', label="Lasso alpha=0.01")
plt.plot(lasso00001.coef_, 'v', label="Lasso alpha=0.0001")
plt.plot(ridge01.coef_, 'o', label="Ridge alpha=0.1")
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.ylim(-25, 25)
plt.legend()
plt.show()