import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

data1 = np.load("dataset1.npy")
data2 = np.load("dataset2.npy")

# Use only X,Y for clustering
X1 = data1[:, :2]
X2 = data2[:, :2]

# Dataset1
nbrs1 = NearestNeighbors(n_neighbors=5).fit(X1)
distances1, _ = nbrs1.kneighbors(X1)
distances1 = np.sort(distances1[:, 4])

plt.figure(figsize=(10,5))
plt.plot(distances1)
plt.title("Dataset1 - k-distance graph")
plt.xlabel("Points sorted")
plt.ylabel("5th Neighbor Distance")
plt.grid(True)
plt.show()

# Dataset2
nbrs2 = NearestNeighbors(n_neighbors=5).fit(X2)
distances2, _ = nbrs2.kneighbors(X2)
distances2 = np.sort(distances2[:, 4])

plt.figure(figsize=(10,5))
plt.plot(distances2)
plt.title("Dataset2 - k-distance graph")
plt.xlabel("Points sorted")
plt.ylabel("5th Neighbor Distance")
plt.grid(True)
plt.show()