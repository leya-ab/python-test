import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Load data
data1 = np.load("dataset1.npy")
data2 = np.load("dataset2.npy")

# Use X,Y only
X1 = data1[:, :2]
X2 = data2[:, :2]

# DBSCAN
db1 = DBSCAN(eps=0.8, min_samples=5).fit(X1)
labels1 = db1.labels_

db2 = DBSCAN(eps=0.45, min_samples=5).fit(X2)
labels2 = db2.labels_

# Plot dataset1
plt.figure(figsize=(10,6))
plt.scatter(X1[:,0], X1[:,1], c=labels1, s=1, cmap='tab20')
plt.title("Dataset1 DBSCAN Clusters")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# Plot dataset2
plt.figure(figsize=(10,6))
plt.scatter(X2[:,0], X2[:,1], c=labels2, s=1, cmap='tab20')
plt.title("Dataset2 DBSCAN Clusters")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()