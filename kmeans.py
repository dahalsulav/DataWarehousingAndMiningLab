import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the diabetes dataset
diabetes_df = pd.read_csv("diabetes.csv")

# Select the features for clustering
X = diabetes_df[["Glucose", "BMI"]]

# Fit K-means clustering to the data
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(X)

# Get the cluster labels
labels = kmeans.labels_

# Evaluate the clustering performance using the Silhouette score
silhouette_avg = silhouette_score(X, labels)
print("Silhouette score:", silhouette_avg)
