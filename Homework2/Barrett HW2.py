# Stone Barrett
# CSE 590: Machine Learning
# Homework 2

# Importing libraries
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import testing
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# Reading datasets
train_set = pd.read_csv("./Homework 2/data/data/spam_train.csv")
test_set = pd.read_csv("./Homework 2/data/data/spam_test.csv")

# Splitting into independent (x) variables and dependent (y) variable
# Dropping ID column
# Inserting into NumPy arrays
x_train = np.array(train_set[list(train_set.columns[1:-1])])
y_train = np.array(train_set["class"])
x_test = np.array(test_set[list(test_set.columns[1:-1])])
y_test = np.array(test_set["class"])

# Looking at shape of data
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)

# 5-fold cross validation
# k = 5
# num_val_samples= len(x_train) // k
# num_epochs= 100
# all_scores= [ ]

# for iin range(k):
# Print ('processing fold #', i)
# val_data= train_data[i*num_val_samples: (i+1)*num_val_samples]
# val_targets= train_targets[i*num_val_samples: (i+1)*num_val_samples]
# partial_train_data= np.concatenate( [train_data[:i* num_val_samples],
# train_data[(i+ 1) * num_val_samples:]], axis=0)
# partial_train_targets= np.concatenate( [train_targets[:i* num_val_samples],
# train_targets[(i+ 1) * num_val_samples:]], axis=0)
# Train and Validate
# all_scores.append(accuracy)


#######################################################################################################
##############       PROBLEM 1A - KNN Binary Classifier         #######################################
#######################################################################################################

def problem1a():
    # Needed for graphing
    training_accuracy = []
    testing_accuracy = []
    neighbors_tries = range(1,11)

    # Loop
    for k in neighbors_tries:

        # Instantiating model and setting k value
        clf = KNeighborsClassifier(n_neighbors=k)
    
        # Fitting model using training data
        clf.fit(x_train, y_train)

        print("\n---------------------------------\nTest set predictions for k =", k)
        print("\n{}".format(clf.predict(x_test)))

        # Recording scores
        training_accuracy.append(clf.score(x_train, y_train))
        testing_accuracy.append(clf.score(x_test, y_test))
    
        # Printing Scores
        print("\nScores for k =", k)
        print("Training Data Score: ", clf.score(x_train, y_train))
        print("Testing Data Score: ", clf.score(x_test, y_test))

    # Graphing
    plt.plot(neighbors_tries, training_accuracy, label="training accuracy")
    plt.plot(neighbors_tries, testing_accuracy, label="testing accuracy")
    #plt.ylim(ymin=0)
    plt.ylabel("Accuracy")
    plt.xlabel("k Value")
    plt.legend()
    plt.show()


#######################################################################################################
##############       PROBLEM 1B - Logistic Regression Classifier         ##############################
#######################################################################################################

def problem1b():
    logreg = LogisticRegression(max_iter=1000000).fit(x_train, y_train)
    print("\nPrinting for C = 1")
    print("Training set score: {:.3f}".format(logreg.score(x_train, y_train)))
    print("Testing set score: {:.3f}".format(logreg.score(x_test, y_test)))

    logreg100 = LogisticRegression(max_iter=1000000,C=100).fit(x_train, y_train)
    print("\nPrinting for C = 100")
    print("Training set score: {:.3f}".format(logreg100.score(x_train, y_train)))
    print("Testing set score: {:.3f}".format(logreg100.score(x_test, y_test)))

    logreg001 = LogisticRegression(max_iter=1000000,C=.01).fit(x_train, y_train)
    print("\nPrinting for C = .01")
    print("Training set score: {:.3f}".format(logreg001.score(x_train, y_train)))
    print("Testing set score: {:.3f}".format(logreg001.score(x_test, y_test)))

    plt.plot(logreg.coef_.T, 'o', label="C=1")
    plt.plot(logreg100.coef_.T, '^', label="C=100")
    plt.plot(logreg001.coef_.T, 'v', label="C=0.001")
    plt.xlabel("Attributes")
    plt.ylabel("Coefficient magnitude")
    plt.legend()
    plt.show()

    input("Press Enter to continue...")

    for C, marker in zip([.001, 1, 100], ['o', '^', 'v']):
        lr_l1 = LogisticRegression(C=C, max_iter=1000000).fit(x_train, y_train)
        print("\nTraining accuracy of l1 logreg with C={:.3f}: {:.2f}".format(C, lr_l1.score(x_train, y_train)))
        print("Testing accuracy of l1 logreg with C={:.3f}: {:.2f}".format(C, lr_l1.score(x_test, y_test)))
        plt.plot(lr_l1.coef_.T, marker, label="C={:.3f}".format(C))
    
    plt.xlabel("Features")
    plt.ylabel("Coefficient magnitude")
    plt.legend()
    plt.show()


#######################################################################################################
##############       PROBLEM 1C - Linear Support Vector Machines Classifier         ###################
#######################################################################################################

def problem1c():
    linsvc = LinearSVC(C=1, max_iter=10000000).fit(x_train, y_train)
    print("\nPrinting for C = 1")
    print("Training set score: {:.3f}".format(linsvc.score(x_train, y_train)))
    print("Testing set score: {:.3f}".format(linsvc.score(x_test, y_test)))

    linsvc100 = LinearSVC(C=100, max_iter=10000000).fit(x_train, y_train)
    print("\nPrinting for C = 100")
    print("Training set score: {:.3f}".format(linsvc.score(x_train, y_train)))
    print("Testing set score: {:.3f}".format(linsvc.score(x_test, y_test)))

    linsvc001 = LinearSVC(C=.01, max_iter=10000000).fit(x_train, y_train)
    print("\nPrinting for C = .01")
    print("Training set score: {:.3f}".format(linsvc.score(x_train, y_train)))
    print("Testing set score: {:.3f}".format(linsvc.score(x_test, y_test)))

    plt.plot(linsvc.coef_.T, 'o', label="C=1")
    plt.plot(linsvc.coef_.T, '^', label="C=100")
    plt.plot(linsvc.coef_.T, 'v', label="C=0.001")
    plt.xlabel("Attributes")
    plt.ylabel("Coefficient magnitude")
    plt.legend()
    plt.show()


#######################################################################################################
##############       Main         #####################################################################
#######################################################################################################

# problem1a()
# problem1b()
# problem1c()