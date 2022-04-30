from ast import Mult
from os import link
from random import random
from re import X
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


x = pd.read_csv("./Part 1/final-project-dataset/final-project-dataset/extracted_features.csv")
y = pd.read_csv("./Part 1/final-project-dataset/final-project-dataset/labels.csv")
X_b = pd.read_csv("./Part 1/final-project-dataset/final-project-dataset/blind_test_data.csv", header=None)

normalizer = MinMaxScaler()
classifier = LinearSVC(C=1, random_state=None, max_iter=100)

piper = make_pipeline(normalizer, classifier)
piper.fit(x,y)
y_pred = piper.predict(X_b)

pd.DataFrame(y_pred).to_csv("./Part 1/final-project-dataset/final-project-dataset/Barrett_Stone.csv", index=None, header=None)