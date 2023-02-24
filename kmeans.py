import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the diabetes dataset
diabetes_df = pd.read_csv("diabetes.csv")

# Select the Glucose and BMI features for clustering
X = diabetes_df[["Glucose", "BMI"]]

# Find the optimal number of clusters using the silhouette score
silhouette_scores = []
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(X)
    silhouette_scores.append(silhouette_score(X, cluster_labels))

optimal_n_clusters = silhouette_scores.index(max(silhouette_scores)) + 2

# Perform K-means clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=0)
y_pred = kmeans.fit_predict(X)

# Plot the clusters
plt.figure(figsize=(10, 6))
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_pred, cmap="viridis")
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    marker="x",
    s=200,
    linewidths=3,
    color="r",
)
plt.xlabel("Glucose")
plt.ylabel("BMI")
plt.title("K-means clustering of Pima Indian Diabetes dataset")
plt.show()

# Plot the silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores)
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette score")
plt.title("Silhouette score vs number of clusters for K-means clustering")
plt.show()
