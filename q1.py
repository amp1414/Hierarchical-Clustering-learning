# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 18:29:05 2024

@author: m_pan
"""

from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np

# Load the Olivetti faces dataset
faces = fetch_olivetti_faces()

X = faces.data  # Feature matrix (images)
y = faces.target  # Labels (person id)


from sklearn.model_selection import StratifiedShuffleSplit

# Create stratified splits for training, validation, and test sets
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_index, test_index = next(sss.split(X, y))

# Split into training + validation and test set
X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]

# Further split training into actual training and validation set (e.g., 80-20 split)
sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)  # 0.25 of the remaining 80% for validation
train_index, val_index = next(sss_val.split(X_train, y_train))

X_train, X_val = X_train[train_index], X_train[val_index]
y_train, y_val = y_train[train_index], y_train[val_index]

from sklearn.model_selection import cross_val_score

# Initialize the classifier
clf = RandomForestClassifier(random_state=42)

# Use 5-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation
cv_scores = cross_val_score(clf, X_train, y_train, cv=skf)

print("Cross-validation scores: ", cv_scores)
print("Mean cross-validation score: ", cv_scores.mean())

# Train on the entire training set and evaluate on the validation set
clf.fit(X_train, y_train)
y_val_pred = clf.predict(X_val)

# Evaluate the performance
print(classification_report(y_val, y_val_pred))

from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import minkowski, euclidean
from sklearn.preprocessing import StandardScaler

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to reduce dimensionality for visualization after clustering (optional)
pca = PCA(n_components=2)  # Reduce to 2 dimensions for plotting
X_pca = pca.fit_transform(X_scaled)

# Agglomerative Clustering with Euclidean Distance
agg_euclidean = AgglomerativeClustering(n_clusters=40, affinity='euclidean', linkage='ward')

# Fit the model
agg_euclidean_labels = agg_euclidean.fit_predict(X_scaled)

# Visualize clustering result (using PCA-reduced data)
import matplotlib.pyplot as plt

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=agg_euclidean_labels, cmap='rainbow', s=30)
plt.title('Agglomerative Clustering with Euclidean Distance')
plt.show()

# Define a custom metric for Minkowski distance
from scipy.spatial.distance import pdist, squareform

# Use p=3 for example (you can change this value to experiment)
p = 3
minkowski_dist = squareform(pdist(X_scaled, metric='minkowski', p=p))

# Agglomerative Clustering using Minkowski Distance
agg_minkowski = AgglomerativeClustering(n_clusters=40, affinity='precomputed', linkage='complete')

# Fit the model (precomputed distance matrix)
agg_minkowski_labels = agg_minkowski.fit_predict(minkowski_dist)

# Visualize the clustering result
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=agg_minkowski_labels, cmap='rainbow', s=30)
plt.title('Agglomerative Clustering with Minkowski Distance (p=3)')
plt.show()

# Compute cosine similarity matrix
cos_sim = cosine_similarity(X_scaled)

# Convert cosine similarity to a distance matrix (1 - cosine similarity) and clip negative values
cosine_dist = np.clip(1 - cos_sim, 0, None)

# Agglomerative Clustering using Cosine Distance
agg_cosine = AgglomerativeClustering(n_clusters=40, affinity='precomputed', linkage='complete')

# Fit the model (precomputed distance matrix)
agg_cosine_labels = agg_cosine.fit_predict(cosine_dist)

# Visualize the clustering result
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=agg_cosine_labels, cmap='rainbow', s=30)
plt.title('Agglomerative Clustering with Cosine Similarity')
plt.show()

from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering


# Define a function to compute silhouette scores for different cluster numbers
def compute_silhouette(X, metric='euclidean', linkage='ward', n_clusters_list=[2, 3, 4, 5, 6, 7, 8, 9, 10]):
    best_score = -1
    best_n_clusters = None
    for n_clusters in n_clusters_list:
        # Perform Agglomerative Clustering with the specified number of clusters
        clustering = AgglomerativeClustering(n_clusters=n_clusters, metric=metric, linkage=linkage)
        cluster_labels = clustering.fit_predict(X)
        # Calculate silhouette score
        score = silhouette_score(X, cluster_labels, metric=metric)
        print(f"Number of clusters: {n_clusters}, Silhouette Score: {score}")
        if score > best_score:
            best_score = score
            best_n_clusters = n_clusters
    print(f"Best silhouette score: {best_score} with {best_n_clusters} clusters.")
    return best_n_clusters

# Compute silhouette score for Euclidean Distance (4a) using 'ward' linkage
print("Silhouette Score for Euclidean Distance:")
best_clusters_euclidean = compute_silhouette(X_scaled, metric='euclidean', linkage='ward')

# Compute silhouette score for Minkowski Distance (4b) using 'complete' linkage and precomputed distance matrix
print("Silhouette Score for Minkowski Distance:")
best_clusters_minkowski = compute_silhouette(minkowski_dist, metric='precomputed', linkage='complete')

# Compute silhouette score for Cosine Distance (4c) using 'complete' linkage
print("Silhouette Score for Cosine Similarity (Converted to Distance):")
best_clusters_cosine = compute_silhouette(cosine_dist, metric='precomputed', linkage='complete')


# We already have the best clustering labels from 4(a) - Euclidean Distance with 2 clusters
# Let's use these labels to train a classifier as in (3) using k-fold cross-validation

# The clustering labels from Agglomerative Clustering with Euclidean distance
best_cluster_labels = agg_euclidean_labels  # This corresponds to 2 clusters

# We'll use these cluster labels as the new target (y) for classification
y_cluster = best_cluster_labels

# Now, we will perform k-fold cross-validation using RandomForestClassifier on these new labels
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Initialize the classifier
clf_cluster = RandomForestClassifier(random_state=42)

# Use 5-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation with the clustering labels
cv_cluster_scores = cross_val_score(clf_cluster, X_scaled, y_cluster, cv=skf)

# Print the cross-validation scores
print("Cross-validation scores using clustering labels: ", cv_cluster_scores)
print("Mean cross-validation score using clustering labels: ", cv_cluster_scores.mean())
