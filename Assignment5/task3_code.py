import numpy as np
from sklearn.cluster import DBSCAN

# Load data
data1 = np.load("dataset1.npy")
data2 = np.load("dataset2.npy")

# Use X,Y
X1 = data1[:, :2]
X2 = data2[:, :2]

# DBSCAN
labels1 = DBSCAN(eps=0.8, min_samples=5).fit_predict(X1)
labels2 = DBSCAN(eps=0.45, min_samples=5).fit_predict(X2)

# Function
def largest_cluster_bounds(X, labels):
    unique = set(labels)
    unique.discard(-1)

    largest_label = max(unique, key=lambda l: np.sum(labels == l))
    cluster = X[labels == largest_label]

    min_x = cluster[:,0].min()
    max_x = cluster[:,0].max()
    min_y = cluster[:,1].min()
    max_y = cluster[:,1].max()

    return largest_label, min_x, max_x, min_y, max_y

# Results
r1 = largest_cluster_bounds(X1, labels1)
r2 = largest_cluster_bounds(X2, labels2)

print("Dataset1")
print("Cluster:", r1[0])
print("min_x:", r1[1])
print("max_x:", r1[2])
print("min_y:", r1[3])
print("max_y:", r1[4])

print("\nDataset2")
print("Cluster:", r2[0])
print("min_x:", r2[1])
print("max_x:", r2[2])
print("min_y:", r2[3])
print("max_y:", r2[4])