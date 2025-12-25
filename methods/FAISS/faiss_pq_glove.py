"""
OPQ + PQ MIPS for GloVe (Mac-compatible FAISS)
Uses OPQMatrix instead of IndexOPQ.
"""

import numpy as np
import h5py, requests, tempfile
import faiss
import time, psutil, os

# -----------------------------
# Utils
# -----------------------------
def mem_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024**2)

def recall_at_k(I, gt, k):
    return np.mean([
        np.intersect1d(I[i], gt[i]).size / k
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
# -----------------------------
xb /= np.linalg.norm(xb, axis=1, keepdims=True)
xq /= np.linalg.norm(xq, axis=1, keepdims=True)

# -----------------------------
# OPQ + PQ parameters
# -----------------------------
d = xb.shape[1]   # 100
k = 10
M = 10            # number of PQ blocks (bytes per vector)
nbits = 8         # bits per sub-quantizer (256 centroids)

# -----------------------------
# Build OPQ + PQ index
# -----------------------------
print("\nBuilding OPQ + PQ index...")
mem_before = mem_mb()
t0 = time.time()

opq = faiss.OPQMatrix(d, M)
pq  = faiss.IndexPQ(d, M, nbits, faiss.METRIC_INNER_PRODUCT)
index = faiss.IndexPreTransform(opq, pq)

index.train(xb)   # trains OPQ + PQ
index.add(xb)

t1 = time.time()
mem_after = mem_mb()

# -----------------------------
# Search
# -----------------------------
print("Searching...")
t2 = time.time()
D, I = index.search(xq, k)
t3 = time.time()

# -----------------------------
# Recall
# -----------------------------
recall = recall_at_k(I, gt, k)

# -----------------------------
# Report
# -----------------------------
print("\n============================")
print("OPQ + PQ Results (GloVe-100-angular)")
print("============================")
print(f"Recall@10        : {recall:.4f}")
print(f"Build time (s)   : {t1 - t0:.2f}")
print(f"Search time (s)  : {t3 - t2:.2f}")
print(f"Avg latency (ms) : {1000 * (t3 - t2) / len(xq):.3f}")
print(f"Index memory (MB): {mem_after - mem_before:.2f}")
