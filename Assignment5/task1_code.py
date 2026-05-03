import numpy as np
import matplotlib.pyplot as plt

# Load datasets
data1 = np.load("dataset1.npy")
data2 = np.load("dataset2.npy")

print("Dataset1 shape:", data1.shape)
print("Dataset2 shape:", data2.shape)

# Extract Z values
z1 = data1[:, 2]
z2 = data2[:, 2]

# Plot histogram dataset1
plt.figure(figsize=(10,5))
plt.hist(z1, bins=100)
plt.title("Dataset1 - Z Value Histogram")
plt.xlabel("Height (Z)")
plt.ylabel("Count")
plt.grid(True)
plt.show()

# Plot histogram dataset2
plt.figure(figsize=(10,5))
plt.hist(z2, bins=100)
plt.title("Dataset2 - Z Value Histogram")
plt.xlabel("Height (Z)")
plt.ylabel("Count")
plt.grid(True)
plt.show()