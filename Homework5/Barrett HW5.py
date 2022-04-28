# Stone Barrett
# CSE 590: Machine Learning
# Homework 5

# Importing libraries
from matplotlib.cbook import silent_list
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import cluster
from sklearn.cluster import KMeans, dbscan
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.cluster.hierarchy import dendrogram, ward
from sklearn.cluster import AgglomerativeClustering
import mglearn
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# Reading datasets
x1 = pd.read_csv("./Homework 5/dataset_hwk5/dataset_hwk5/X.csv")
y1 = pd.read_csv("./Homework 5/dataset_hwk5/dataset_hwk5/y.csv")

# Converting to np array
x = np.array(x1)
y9 = np.array(y1)
y = np.squeeze(y9, axis=1)

# Looking at shape of data
print("x shape: ", x.shape)
print("y shape: ", y.shape)

# # K-Means Clustering
# inertias = []
# for i in range(2,21):
#     kmeans = KMeans(n_clusters=i)
#     kmeans.fit(x)
#     inertias.append(kmeans.inertia_)

# plt.plot(range(2,21), inertias, marker='o')
# plt.xlabel("Number of Clusters")
# plt.ylabel("SSE")
# plt.tight_layout()
# plt.show()

# Choose i = 5
n_clusters = 4

# # Create a subplot with 1 row and 2 columns
# fig, (ax1, ax2) = plt.subplots(1, 2)
# fig.set_size_inches(18, 7)

# # The 1st subplot is the silhouette plot
# # The silhouette coefficient can range from -1, 1 but in this example all
# # lie within [-0.1, 1]
# # ax1.set_xlim([-0.1, 1])
# # The (n_clusters+1)*10 is for inserting blank space between silhouette
# # plots of individual clusters, to demarcate them clearly.
# ax1.set_ylim([0, len(x) + (n_clusters + 1) * 10])

# # Initialize the clusterer with n_clusters value and a random generator
# # seed of 10 for reproducibility.
# #clusterer = KMeans(n_clusters=n_clusters, random_state=10)
# #clusterer = AgglomerativeClustering(n_clusters=4)
# #clusterer = AgglomerativeClustering(n_clusters=4, linkage='single')
# clusterer = AgglomerativeClustering(n_clusters=4, linkage='complete')
# cluster_labels = clusterer.fit_predict(x)

# # The silhouette_score gives the average value for all the samples.
# # This gives a perspective into the density and separation of the formed
# # clusters
# silhouette_avg = silhouette_score(x, cluster_labels)
# print(
#     "For n_clusters =",
#     n_clusters,
#     "The average silhouette_score is :",
#     silhouette_avg,
# )

# # Compute the silhouette scores for each sample
# sample_silhouette_values = silhouette_samples(x, cluster_labels)

# y_lower = 10
# for i in range(n_clusters):
#     # Aggregate the silhouette scores for samples belonging to
#     # cluster i, and sort them
#     ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

#     ith_cluster_silhouette_values.sort()

#     size_cluster_i = ith_cluster_silhouette_values.shape[0]
#     y_upper = y_lower + size_cluster_i

#     color = cm.nipy_spectral(float(i) / n_clusters)
#     ax1.fill_betweenx(
#         np.arange(y_lower, y_upper),
#         0,
#         ith_cluster_silhouette_values,
#         facecolor=color,
#         edgecolor=color,
#         alpha=0.7,
#     )

#     # Label the silhouette plots with their cluster numbers at the middle
#     ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

#     # Compute the new y_lower for next plot
#     y_lower = y_upper + 10  # 10 for the 0 samples

# ax1.set_title("The silhouette plot for the various clusters.")
# ax1.set_xlabel("The silhouette coefficient values")
# ax1.set_ylabel("Cluster label")

# # The vertical line for average silhouette score of all the values
# ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

# ax1.set_yticks([])  # Clear the yaxis labels / ticks
# ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

# # 2nd Plot showing the actual clusters formed
# colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
# ax2.scatter(
#     x[:, 0], x[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
# )

# # Labeling the clusters
# # centers = clusterer.cluster_centers_
# # Draw white circles at cluster centers
# # ax2.scatter(
# #     centers[:, 0],
# #     centers[:, 1],
# #     marker="o",
# #     c="white",
# #     alpha=1,
# #     s=200,
# #     edgecolor="k",
# # )

# # for i, c in enumerate(centers):
# #     ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

# ax2.set_title("The visualization of the clustered data.")
# ax2.set_xlabel("Feature space for the 1st feature")
# ax2.set_ylabel("Feature space for the 2nd feature")

# plt.suptitle(
#     "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
#     % n_clusters,
#     fontsize=14,
#     fontweight="bold",
# )

# plt.show()


# km = KMeans(n_clusters=20, init='k-means++', n_init=10, max_iter=300, tol=1e-04, random_state=0)
# y_km = km.fit_predict(x)
# cluster_labels = np.unique(y_km)
# n_clusters = 5
# silhouette_vals = silhouette_samples(x, y_km, metric='euclidean')
# y_ax_lower, y_ax_upper = 0, 0
# yticks = []
# for i, c in enumerate(cluster_labels):
#     c_silhouette_vals = silhouette_vals[y_km == c]
#     c_silhouette_vals.sort()
#     y_ax_upper += len(c_silhouette_vals)
#     color = cm.jet(float(i) / n_clusters)
#     plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, edgecolor='none', color=color)
# yticks.append((y_ax_lower+y_ax_upper) / 2)
# y_ax_lower += len(c_silhouette_vals)
# silhouette_avg = np.mean(silhouette_vals)
# plt.axvline(silhouette_avg, color="red")
# plt.yticks(yticks, cluster_labels + 1)
# plt.ylabel('Cluster')
# plt.xlabel('Silhouette Coefficient')
# plt.tight_layout()
# plt.show()


# Agglomerative Clustering
# linkage_array = ward(x)
# dendrogram(linkage_array, p=5, truncate_mode='level')
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

# Ward Linkage
# agg = AgglomerativeClustering(n_clusters=4)
# assignment = agg.fit_predict(x)
# mglearn.discrete_scatter(x[:, 0], x[:, 1], assignment)
# plt.legend(["Cluster 0", "Cluster 1", "Cluster 2", "Cluster 3"], loc="best")
# plt.xlabel("Feature 0")
# plt.ylabel("Feature 1")
# plt.show()

# # Single Link
# agg = AgglomerativeClustering(n_clusters=4)
# assignment = agg.fit_predict(x)
# mglearn.discrete_scatter(x[:, 0], x[:, 1], assignment)
# plt.legend(["Cluster 0", "Cluster 1", "Cluster 2", "Cluster 3"], loc="best")
# plt.xlabel("Feature 0")
# plt.ylabel("Feature 1")
# plt.show()

# # Complete Link
# agg = AgglomerativeClustering(n_clusters=4)
# assignment = agg.fit_predict(x)
# mglearn.discrete_scatter(x[:, 0], x[:, 1], assignment)
# plt.legend(["Cluster 0", "Cluster 1", "Cluster 2", "Cluster 3"], loc="best")
# plt.xlabel("Feature 0")
# plt.ylabel("Feature 1")
# plt.show()

# ARIs
# rescale the data to zero mean and unit variance
# scaler = StandardScaler()
# scaler.fit(x)
# X_scaled = scaler.transform(x)

# fig, axes = plt.subplots(1, 4, figsize=(15, 3), subplot_kw={'xticks': (), 'yticks': ()})
# # make a list of algorithms to use
# algorithms = [KMeans(n_clusters=5), AgglomerativeClustering(n_clusters=4)]
# # create a random cluster assignment for reference
# random_state = np.random.RandomState(seed=0)
# random_clusters = random_state.randint(low=0, high=2, size=len(x))
# # plot random assignment
# axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=random_clusters, cmap=mglearn.cm3, s=60)
# axes[0].set_title("Random assignment - ARI: {:.2f}".format(adjusted_rand_score(y, random_clusters)))
# for ax, algorithm in zip(axes[1:], algorithms):
#     # plot the cluster assignments and cluster centers
#     clusters = algorithm.fit_predict(X_scaled)
#     ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm3, s=60)
#     ax.set_title("{} - ARI: {:.2f}".format(algorithm.__class__.__name__, adjusted_rand_score(y, clusters)))

# plt.show()

# Finding Core and Boundary Points
scaler = StandardScaler()
scaler.fit(x)
x9 = scaler.transform(x)
dbscan = DBSCAN(eps=6, min_samples=5, metric='euclidean')
clusters = dbscan.fit_predict(x9)
print("Cluster memberships:\n{}".format(clusters))
q=0
for i in clusters:
    if i != -1:
        print("DBSCAN Value: ", i, "\tSample Number: ", q)
    q+=1
