# Assignment 5 - LiDAR Point Cloud Processing Techniques

## Author
Leya Abu Hamdh

## Description
This assignment focuses on processing LiDAR point cloud datasets using Python and machine learning methods. Two datasets were analyzed in order to detect railway-related structures and extract useful spatial information.

The work was divided into three main tasks:

1. Estimate ground level using histogram analysis of Z-values.
2. Use DBSCAN clustering and determine suitable epsilon values with k-distance graphs.
3. Detect the largest cluster representing railway infrastructure and calculate its boundaries.

---

## Files Included

- `dataset1.npy`
- `dataset2.npy`
- `share.py`
- `task1_code.py`
- `task2_code.py`
- `task2_clusters.py`
- `task3_code.py`
- `Assignment 5 (AI).pdf`

---

## Methods Used

- Python
- NumPy
- Matplotlib
- Scikit-learn
- DBSCAN
- Nearest Neighbors

---

## Main Results

### Ground Level

- Dataset1: approximately **61.2**
- Dataset2: approximately **61.2**

### DBSCAN Parameters

- Dataset1: **eps = 0.8**
- Dataset2: **eps = 0.45**

### Largest Cluster Boundaries

#### Dataset1

- min(x): 5.67
- max(x): 62.14
- min(y): 80.01
- max(y): 160.00

#### Dataset2

- min(x): 9.36
- max(x): 37.72
- min(y): 0.00
- max(y): 80.00

---

## Conclusion

The assignment demonstrated how LiDAR point cloud data can be analyzed using unsupervised machine learning techniques. Histogram analysis was useful for identifying ground level, while DBSCAN clustering successfully separated railway structures from surrounding noise.

---

## How to Run

```bash
python task1_code.py
python task2_code.py
python task2_clusters.py
python task3_code.py