from ast import Mult
from os import link
from random import random
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
from sklearn.pipeline import Pipeline, make_pipeline
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


x = pd.read_csv("./Final Project/Part 1/final-project-dataset/final-project-dataset/extracted_features.csv")
y = pd.read_csv("./Final Project/Part 1/final-project-dataset/final-project-dataset/labels.csv")
images = pd.read_csv("./Final Project/Part 1/final-project-dataset/final-project-dataset/raw_images.csv")

index = range(0,2219)


X_train, X_test, y_train, y_test, i_train, i_test = train_test_split(x,y,index)

TrainingScore = []
TestingScore = []
sample = []
artist = []

normalizer = MinMaxScaler()
classifier = LinearSVC(C=1, random_state=None, max_iter=100)

piper = make_pipeline(normalizer, classifier)

piper.fit(X_train,y_train)
correctCounter = 0
incorrectcounter = 0

predicted = piper.predict(X_test)
for i in range(0,predicted.size):
    correct=True
    if predicted[i] == y_test._values[i]:
        artist.append("Art by ")
    else:
        artist.append("Incorrectly classified as ")
        correct = False

    if predicted[i] == 0:
        artist[i] = artist[i] + 'Pierre-Auguste_Renoir'
    if predicted[i] == 1:
        artist[i] = artist[i] + 'Raphael'
    if predicted[i] == 2:
        artist[i] = artist[i] + 'Leonardo_da_Vinci'
    if predicted[i] == 3:
        artist[i] = artist[i] + 'Sandro_Botticelli'
    if predicted[i] == 4:
        artist[i] = artist[i] + 'Francisco_Goya'
    if predicted[i] == 5:
        artist[i] = artist[i] + 'Vincent_van_Gogh'
    if predicted[i] == 6:
        artist[i] = artist[i] + 'Pablo_Picasso'
    if predicted[i] == 7:
        artist[i] = artist[i] + 'Albrecht_Durer'
    if predicted[i] == 8:
        artist[i] = artist[i] + 'Others'
    

    if correct == False:
        if y_test._values[i] == 0:
            artist[i] = artist[i] + ' should be Pierre-Auguste_Renoir'
        if y_test._values[i] == 1:
            artist[i] = artist[i] + ' should be Raphael'
        if y_test._values[i] == 2:
            artist[i] = artist[i] + ' should be Leonardo_da_Vinci'
        if y_test._values[i] == 3:
            artist[i] = artist[i] + ' should be Sandro_Botticelli'
        if y_test._values[i] == 4:
            artist[i] = artist[i] + ' should be Francisco_Goya'
        if y_test._values[i] == 5:
            artist[i] = artist[i] + ' should be Vincent_van_Gogh'
        if y_test._values[i] == 6:
            artist[i] = artist[i] + ' should be Pablo_Picasso'
        if y_test._values[i] == 7:
            artist[i] = artist[i] + ' should be Albrecht_Durer'
        if y_test._values[i] == 8:
            artist[i] = artist[i] + ' should be Others'

    print(artist[i])

    if correct == True:
        correctCounter += 1
        image = images.iloc[i_test[i]].values.reshape(150,150,3)
        plt.figure(artist[i])
        plt.imshow(image)
    if correct == False:
        incorrectcounter += 1
        image = images.iloc[i_test[i]].values.reshape(150,150,3)
        plt.figure(artist[i])
        plt.imshow(image)

# image = images.iloc[i_test[0:1]].values.reshape(150,150,3)
# plt.figure("Art by "+artist[i])
# plt.imshow(image)
print("Correct/incorrect : {}/{}".format(correctCounter,incorrectcounter))
print("Total: {}".format(correctCounter+incorrectcounter))
plt.show()

# print(Artist)

