# Stone Barrett
# CSE 590: Machine Learning
# Final Project

# Testing
# import os
# os.getcwd()

# Importing libraries
from ast import Mult
from os import link
from turtle import stamp
import mglearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, randomized_svd
from sklearn.metrics import accuracy_score, silhouette_samples
from sklearn.metrics import silhouette_score
from sklearn.tree import DecisionTreeClassifier
from sympy import Mul
from sklearn.model_selection import cross_val_score
from mglearn.plots import plot_cross_val_selection
from sklearn.model_selection import train_test_split
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import ward, single, complete

# Reading databases
x1 = pd.read_csv("./Final Project/Part 1/final-project-dataset/final-project-dataset/extracted_features.csv")
y1 = pd.read_csv("./Final Project/Part 1/final-project-dataset/final-project-dataset/labels.csv")
images = pd.read_csv("./Final Project/Part 1/final-project-dataset/final-project-dataset/raw_images.csv")

# file = open("./Final Project/Part 1/final-project-dataset/final-project-dataset/raw_images.csv", mode="r")
# img = file.read()
# file.close()

# Converting to np array
x = np.array(x1)
y9 = np.array(y1)
y = np.squeeze(y9, axis=1)
# img = np.array(img1)

# Looking at shape of data
# print("x shape: ", x.shape)
# print("y shape: ", y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y)

# print("x train shape: ", x_train.shape)
# print("y train shape: ", y_train.shape)
# print("x test shape: ", x_test.shape)
# print("y test shape: ", y_test.shape)

# plt.imshow(img[1].reshape(150,150,3))

# Section 1
def Part1():
    # Creating all pipelines
    # pipe1 = Pipeline([("scaler1", StandardScaler()), ("classifier1", KNeighborsClassifier())])
    # pipe2 = Pipeline([("scaler2", StandardScaler()), ("classifier2", LogisticRegression())])
    # pipe3 = Pipeline([("scaler3", StandardScaler()), ("classifier3", LinearSVC())])
    # pipe4 = Pipeline([("scaler4", StandardScaler()), ("classifier4", RandomForestClassifier())])
    # pipe5 = Pipeline([("scaler5", MinMaxScaler()), ("classifier5", KNeighborsClassifier())])
    # pipe6 = Pipeline([("scaler6", MinMaxScaler()), ("classifier6", LogisticRegression())])
    # pipe7 = Pipeline([("scaler7", MinMaxScaler()), ("classifier7", LinearSVC())])
    # pipe8 = Pipeline([("scaler8", MinMaxScaler()), ("classifier8", RandomForestClassifier())])

    # # Fitting to data
    # pipe1.fit(x_train,y_train)
    # pipe2.fit(x_train,y_train)
    # pipe3.fit(x_train,y_train)
    # pipe4.fit(x_train,y_train)
    # pipe5.fit(x_train,y_train)
    # pipe6.fit(x_train,y_train)
    # pipe7.fit(x_train,y_train)
    # pipe8.fit(x_train,y_train)

    # # Printing scores
    # print("Train score for Standard + KNN:\t\t{:.2f}".format(pipe1.score(x_train, y_train)))
    # print("Test score for Standard + KNN:\t\t{:.2f}".format(pipe1.score(x_test, y_test)))
    # print("Train score for Standard + Logistic:\t{:.2f}".format(pipe2.score(x_train, y_train)))
    # print("Test score for Standard + Logistic:\t{:.2f}".format(pipe2.score(x_test, y_test)))
    # print("Train score for Standard + SVM:\t\t{:.2f}".format(pipe3.score(x_train, y_train)))
    # print("Test score for Standard + SVM:\t\t{:.2f}".format(pipe3.score(x_test, y_test)))
    # print("Train score for Standard + Forest:\t{:.2f}".format(pipe4.score(x_train, y_train)))
    # print("Test score for Standard + Forest:\t{:.2f}".format(pipe4.score(x_test, y_test)))
    # print("Train score for MinMax + KNN:\t\t{:.2f}".format(pipe5.score(x_train, y_train)))
    # print("Test score for MinMax + KNN:\t\t{:.2f}".format(pipe5.score(x_test, y_test)))
    # print("Train score for MinMax + Logistic:\t{:.2f}".format(pipe6.score(x_train, y_train)))
    # print("Test score for MinMax + Logistic:\t{:.2f}".format(pipe6.score(x_test, y_test)))
    # print("Train score for MinMax + SVM:\t\t{:.2f}".format(pipe7.score(x_train, y_train)))
    # print("Test score for MinMax + SVM:\t\t{:.2f}".format(pipe7.score(x_test, y_test)))
    # print("Train score for MinMax + Forest:\t{:.2f}".format(pipe8.score(x_train, y_train)))
    # print("Test score for MinMax + Forest:\t\t{:.2f}".format(pipe8.score(x_test, y_test)))

    # Grid search for optimal parameters
    # best_score = 0
    # for n_estimators in [10,20,30,40,50,60,70,80,90,100]:
    #     for max_depth in [2,10,30,50,None]:
    #         for random_state in [42,None]:
    #             for max_samples in [2,10,30,50,None]:
    #                 pipe = Pipeline([("scaler", StandardScaler()), ("classifier", RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, max_samples=max_samples))])
    #                 pipe.fit(x,y)
    #                 score = pipe.score(x,y)
    #                 if score > best_score:
    #                     best_score = score
    #                     best_parameters = {'n_estimators': n_estimators, 'max_depth': max_depth, 'random_state': random_state, 'max_samples': max_samples}
    
    # pipebest = Pipeline([("scaler", StandardScaler()), ("classifier", RandomForestClassifier(**best_parameters))])
    # print("Results for Standard Scaler on Random Forest")
    # print("Best score found: {:.20f}".format(best_score))
    # print("Best parameters: ", best_parameters)

    # best_scoreb = 0
    # for n_estimators in [10,20,30,40,50,60,70,80,90,100]:
    #     for max_depth in [2,10,30,50,None]:
    #         for random_state in [42,None]:
    #             for max_samples in [2,10,30,50,None]:
    #                 pipeb = Pipeline([("scaler", MinMaxScaler()), ("classifier", RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, max_samples=max_samples))])
    #                 pipeb.fit(x,y)
    #                 scoreb = pipeb.score(x,y)
    #                 if scoreb > best_scoreb:
    #                     best_scoreb = scoreb
    #                     best_parametersb = {'n_estimators': n_estimators, 'max_depth': max_depth, 'random_state': random_state, 'max_samples': max_samples}
    
    # pipebestb = Pipeline([("scaler", MinMaxScaler()), ("classifier", RandomForestClassifier(**best_parametersb))])
    # print("Results for MinMax Scaler on Random Forest")
    # print("Best score found: {:.20f}".format(best_scoreb))
    # print("Best parameters: ", best_parametersb)

    # Grid search + cross-validation
    # best_score = 0
    # for n_estimators in [10,20,30,40,50,60,70,80,90,100]:
    #     for max_depth in [2,10,30,50,None]:
    #         for random_state in [42,None]:
    #             for max_samples in [2,10,30,50,None]:
    #                 pipe = Pipeline([("scaler", MinMaxScaler()), ("classifier", RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, max_samples=max_samples))])
    #                 scores = cross_val_score(pipe, x, y, cv=4)
    #                 score = np.mean(scores)
    #                 if score > best_score:
    #                     best_score = score
    #                     best_parameters = {'n_estimators': n_estimators, 'max_depth': max_depth, 'random_state': random_state, 'max_samples': max_samples}
    
    # print("Random Forest + Standard Scaler Highest Cross-Validation Average:")
    # print(best_score)
    # print("Parameters:")
    # print(best_parameters)

    best_score = 0
    for C in [.001, .01, .1, 1, 10, 100]:
        for random_state in [None, 42]:
            for max_iter in [100, 1000, 10000, 100000, 1000000]:
                pipe = Pipeline([("scaler", MinMaxScaler()), ("classifier", LinearSVC(C=C, random_state=random_state, max_iter=max_iter))])
                scores = cross_val_score(pipe, x_train, y_train, cv=4)
                score = np.mean(scores)
                if score > best_score:
                    best_score = score
                    best_parameters = {'C': C, 'random_state': random_state, 'max_iter': max_iter}
    print("Linear SVM + MinMax Scaler Highest Cross-Validation Average:")
    print("{:.2f}".format(best_score))
    print("Parameters:")
    print(best_parameters)
    # mglearn.plots.plot_cross_val_selection()



def Part2():
    # scaler = StandardScaler
    # scaler.fit(x,y)
    # x_scaled = scaler.transform(x)

    # Testing 2
    #pca = PCA(n_components=8)
    lsvm = LinearSVC(C=1, random_state=None, max_iter=100)

    # x_pca = pca.fit_transform(x)
    # rfc.fit(x_pca, y)

    # print("Score on scaled data: {:.20f}".format(rfc.score(x_pca,y)))
    # print("PCA Component Shape: {}".format(pca.components_.shape))
    # print("PCA Components:\n{}".format(pca.components_))

    # pca.fit(x)

    # x_pca = pca.transform(x)
    # print("Original Shape: {}".format(str(x.shape)))
    # print("Reduced Shape: {}".format(str(x_pca.shape)))
    # plt.figure(figsize=(8,8))
    # mglearn.discrete_scatter(x_pca[:,0], x_pca[:,1])
    # plt.gca().set_aspect("equal")
    # plt.xlabel("First Principal Component")
    # plt.ylabel("Second Principal Component")
    # plt.show()

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    # x_train_pca = pca.fit_transform(x_train_scaled)
    # x_test_pca = pca.transform(x_test_scaled)
    # lsvm.fit(x_train_pca, y_train)
    # print("Original Shape: {}".format(str(x_train_scaled.shape)))
    # print("Reduced Shape: {}".format(str(x_train_pca.shape)))
    # print("PCA Component Shape: {}".format(pca.components_.shape))
    # print("PCA Components:\n{}".format(pca.components_))
    # plt.figure(figsize=(8,8))
    # mglearn.discrete_scatter(x_train_pca[:,0], x_train_pca[:,1])
    # plt.gca().set_aspect("equal")
    # plt.xlabel("First Principal Component")
    # plt.ylabel("Second Principal Component")
    # plt.show()

    scores = []
    ncomponents = []
    for i in range(1,65):
        pca = PCA(n_components=i)
        # scaler = StandardScaler()
        # x_train_scaled = scaler.fit_transform(x_train)
        # x_test_scaled = scaler.transform(x_test)
        x_train_pca = pca.fit_transform(x_train_scaled)
        x_test_pca = pca.transform(x_test_scaled)
        lsvm.fit(x_train_pca, y_train)
        print("Score on PCA Data: {:.2f}".format(lsvm.score(x_test_pca, y_test)))
        scores.append(lsvm.score(x_test_pca, y_test))
        ncomponents.append(i)
    plt.plot(ncomponents, scores)
    plt.title("Accuracy Score vs. PCA Components")
    plt.xlabel("n_components")
    plt.ylabel("Accuracy (%)")
    plt.show()

def Part3():
    # tree = DecisionTreeClassifier(criterion='entropy', max_depth=None, random_state=1)
    # # bag = BaggingClassifier(base_estimator=tree, n_estimators=2000, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=1, random_state=1)
    # tree = tree.fit(x_train, y_train)
    # y_train_pred = tree.predict(x_train)
    # y_test_pred = tree.predict(x_test)
    # tree_train = accuracy_score(y_train, y_train_pred)
    # tree_test = accuracy_score(y_test, y_test_pred)
    # print("Decision tree train, test accuracies %.3f, %.3f" % (tree_train, tree_test))

    # lsvm = LinearSVC(C=1, random_state=None, max_iter=100)
    # bag = BaggingClassifier(base_estimator=lsvm, n_estimators=2000, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=1, random_state=1)
    # lsvm = lsvm.fit(x_train, y_train)
    # y_train_pred = lsvm.predict(x_train)
    # y_test_pred = lsvm.predict(x_test)
    # lsvm_train = accuracy_score(y_train, y_train_pred)
    # lsvm_test = accuracy_score(y_test, y_test_pred)
    # print("LSVM train, test accuracies %.3f, %.3f" % (lsvm_train, lsvm_test))

    # bag = bag.fit(x_train, y_train)
    # y_train_pred = bag.predict(x_train)
    # y_test_pred = bag.predict(x_test)
    # bag_train = accuracy_score(y_train, y_train_pred) 
    # bag_test = accuracy_score(y_test, y_test_pred) 
    # print("Bagging train, test accuracies %.3f, %.3f" % (bag_train, bag_test))

    lsvm = LinearSVC(C=1, random_state=None, max_iter=100)
    # bag = BaggingClassifier(base_estimator=lsvm, n_estimators=2000, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=1, random_state=1)
    lsvm = lsvm.fit(x_train, y_train)
    y_train_pred = lsvm.predict(x_train)
    y_test_pred = lsvm.predict(x_test)
    lsvm_train = accuracy_score(y_train, y_train_pred)
    lsvm_test = accuracy_score(y_test, y_test_pred)
    print("LSVM train, test accuracies %.3f, %.3f" % (lsvm_train, lsvm_test))

    ada = AdaBoostClassifier(base_estimator=lsvm, n_estimators=100, learning_rate=.1, random_state=1, algorithm='SAMME')
    ada = ada.fit(x_train, y_train)
    y_train_pred = ada.predict(x_train)
    y_test_pred = ada.predict(x_test)
    ada_train = accuracy_score(y_train, y_train_pred) 
    ada_test = accuracy_score(y_test, y_test_pred) 
    print("AdaBoost train, test accuracies %.3f, %.3f" % (ada_train, ada_test))

def Part4():
    # K-Means Clustering
    # inertias = []
    # for i in range(2,21):
    #     kmeans = KMeans(n_clusters=i)
    #     kmeans.fit(x_train)
    #     inertias.append(kmeans.inertia_)

    # plt.plot(range(2,21), inertias, marker='o')
    # plt.xlabel("Number of Clusters")
    # plt.ylabel("SSE")
    # plt.tight_layout()
    # plt.show()

    # Choose n = 3
    n_clusters = 6

    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    # ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(x_train) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.

    # Creating sillhouette plots for different types of clustering
    #clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    #clusterer = AgglomerativeClustering(n_clusters=4, linkage='ward')
    #clusterer = AgglomerativeClustering(n_clusters=2, linkage='single')
    clusterer = AgglomerativeClustering(n_clusters=6, linkage='complete')
    cluster_labels = clusterer.fit_predict(x_train)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(x_train, cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "the average silhouette_score is:",
        silhouette_avg,
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(x_train, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(
        x_train[:, 3], x_train[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )

    # Labeling the clusters
    # centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    # ax2.scatter(
    #     centers[:, 0],
    #     centers[:, 1],
    #     marker="o",
    #     c="white",
    #     alpha=1,
    #     s=200,
    #     edgecolor="k",
    # )

    # for i, c in enumerate(centers):
    #     ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")


    # TITLE CHANGE
    plt.suptitle(
        "Silhouette analysis for Agglomerative clustering (complete) on sample data with n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )
    plt.show()


    # Agglomerative
    # Making Dendrograms
    #linkage_array = ward(x_train)
    #linkage_array = single(x_train)
    #linkage_array = complete(x_train)

    # dendrogram(linkage_array, truncate_mode='level', p=5)
    # # Mark the cuts in the tree that signify two or three clusters
    # ax = plt.gca()
    # bounds = ax.get_xbound()
    # ax.plot(bounds, [7.25, 7.25], '--', c='k')
    # ax.plot(bounds, [4, 4], '--', c='k')
    # ax.text(bounds[1], 7.25, 'two clusters',va='center',fontdict={'size': 15})
    # ax.text(bounds[1], 4, ' three clusters', va='center', fontdict={'size': 15})
    # plt.xlabel("Sample index")
    # plt.ylabel("Cluster distance")
    # plt.show()

    # # Ward Linkage
    # agg = AgglomerativeClustering(n_clusters=4, linkage='ward')
    # assignment = agg.fit_predict(x)
    # mglearn.discrete_scatter(x[:, 3], x[:, 1], assignment)
    # plt.legend(["Cluster 0", "Cluster 1", "Cluster 2", "Cluster 3"], loc="best")
    # plt.xlabel("Feature 3")
    # plt.ylabel("Feature 1")
    # plt.show()

    # # Single Link
    # agg = AgglomerativeClustering(n_clusters=4, linkage='single')
    # assignment = agg.fit_predict(x)
    # mglearn.discrete_scatter(x[:, 3], x[:, 1], assignment)
    # plt.legend(["Cluster 0", "Cluster 1", "Cluster 2", "Cluster 3"], loc="best")
    # plt.xlabel("Feature 3")
    # plt.ylabel("Feature 1")
    # plt.show()

    # # Complete Link
    # agg = AgglomerativeClustering(n_clusters=4, linkage='complete')
    # assignment = agg.fit_predict(x)
    # mglearn.discrete_scatter(x[:, 3], x[:, 1], assignment)
    # plt.legend(["Cluster 0", "Cluster 1", "Cluster 2", "Cluster 3"], loc="best")
    # plt.xlabel("Feature 3")
    # plt.ylabel("Feature 1")
    # plt.show()
    
# Testing
# Part1()
# Part2()
# Part3()
# Part4()