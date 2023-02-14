from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from kmeans import KMeans

# Defining values to make the sample data
centroids = [(-50, -50), (50, 50)]
cluster_std = [20, 20]

# getting the values
X, y = make_blobs(n_samples=1000, cluster_std=cluster_std, centers=centroids, n_features=2, random_state=42)

km = KMeans(n_clusters=2, max_iter=100)
y_means = km.fit_predict(X)

# plotting the data to get a visual representation of data
# plt.scatter(X[:, 0], X[:, 1])
# plt.show()

# plotting to see the clustering
plt.scatter(X[y_means == 0, 0], X[y_means == 0, 1], color='red')
plt.scatter(X[y_means == 1, 0], X[y_means == 1, 1], color='blue')
plt.show()
