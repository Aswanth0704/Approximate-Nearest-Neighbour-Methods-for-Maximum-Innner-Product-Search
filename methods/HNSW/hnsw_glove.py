import numpy as np
import h5py
import requests
import tempfile
import time
import hnswlib
import psutil
import os

# -----------------------------
# Utils
# -----------------------------
def mem_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024**2)

def recall_at_k(neighbors, gt, k):
    return np.mean([
        np.intersect1d(neighbors[i], gt[i]).size / k
        for i in range(len(gt))
    ])

# -----------------------------
# Load ANN-Benchmarks GloVe-100-angular
# -----------------------------
with tempfile.TemporaryDirectory() as tmp:
    url = "http://ann-benchmarks.com/glove-100-angular.hdf5"
    path = os.path.join(tmp, "glove100.hdf5")
    with open(path, "wb") as f:
        f.write(requests.get(url).content)

    f = h5py.File(path, "r")
    xb = f["train"][:]          # (1,183,514, 100)
    xq = f["test"][:]           # (10,000, 100)
    gt = f["neighbors"][:, :10] # GT@10

print("Base:", xb.shape, "Query:", xq.shape)

# -----------------------------
# Normalize (CRITICAL)
# Angular ⇔ dot product on unit vectors
# -----------------------------
xb /= np.linalg.norm(xb, axis=1, keepdims=True)
xq /= np.linalg.norm(xq, axis=1, keepdims=True)

# -----------------------------
# HNSW parameters
# -----------------------------
dim = xb.shape[1]
k = 10
M = 32
ef_construction = 200
ef_search = 100

# -----------------------------
# Build HNSW index
# -----------------------------
print("\nBuilding HNSW index...")
mem_before = mem_mb()
t0 = time.time()

index = hnswlib.Index(space="ip", dim=dim)
index.init_index(
    max_elements=xb.shape[0],
    ef_construction=ef_construction,
    M=M
)
index.add_items(xb)

t1 = time.time()
mem_after = mem_mb()

# -----------------------------
# Search
# -----------------------------
index.set_ef(ef_search)

print("Searching...")
t2 = time.time()
neighbors, distances = index.knn_query(xq, k=k)
t3 = time.time()

# -----------------------------
# Recall
# -----------------------------
recall = recall_at_k(neighbors, gt, k)

# -----------------------------
# Report
# -----------------------------
print("\n============================")
print("HNSW Results (GloVe-100-angular)")
print("============================")
print(f"Recall@10        : {recall:.4f}")
print(f"Build time (s)   : {t1 - t0:.2f}")
print(f"Search time (s)  : {t3 - t2:.2f}")
print(f"Avg latency (ms) : {1000 * (t3 - t2) / len(xq):.3f}")
print(f"Index memory (MB): {mem_after - mem_before:.2f}")
