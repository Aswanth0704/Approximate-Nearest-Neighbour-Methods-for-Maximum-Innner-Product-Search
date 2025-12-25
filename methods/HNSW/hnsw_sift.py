import numpy as np
import hnswlib
import time
import psutil, os

# -----------------------------
# Helpers
# -----------------------------
def read_fvecs(fname):
    with open(fname, "rb") as f:
        data = np.fromfile(f, dtype=np.int32)
        d = data[0]
        return data.reshape(-1, d + 1)[:, 1:].astype(np.float32)

def read_ivecs(fname):
    iv = np.fromfile(fname, dtype=np.int32)
    d = iv[0]
    return iv.reshape(-1, d + 1)[:, 1:]

def mem_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024**2)

# -----------------------------
# Load SIFT1M
# -----------------------------
DATA = "../datasets/sift_1M"

print("Loading SIFT1M...")
xb = read_fvecs(DATA + "/sift_base.fvecs")
xq = read_fvecs(DATA + "/sift_query.fvecs")
gt = read_ivecs(DATA + "/sift_groundtruth.ivecs")[:, :10]

# 🔴 CRITICAL FIX: normalize vectors
xb /= np.linalg.norm(xb, axis=1, keepdims=True)
xq /= np.linalg.norm(xq, axis=1, keepdims=True)

print("Base:", xb.shape, "Query:", xq.shape)

dim = xb.shape[1]
k = 10

# -----------------------------
# HNSW parameters (your values)
# -----------------------------
M = 16
ef_construction = 200
ef_search = 100

# -----------------------------
# Build HNSW (L2)
# -----------------------------
print("\nBuilding HNSW index (L2)...")
mem_before = mem_mb()
t0 = time.time()

index = hnswlib.Index(space="l2", dim=dim)
index.init_index(
    max_elements=xb.shape[0],
    ef_construction=ef_construction,
    M=M
)

index.add_items(xb)
index.set_ef(ef_search)

t1 = time.time()
mem_after = mem_mb()

# -----------------------------
# Search
# -----------------------------
print("Searching...")
t2 = time.time()
labels, _ = index.knn_query(xq, k=k)
t3 = time.time()

# -----------------------------
# Recall@10
# -----------------------------
recall = np.mean([
    np.intersect1d(labels[i], gt[i]).size / k
    for i in range(len(xq))
])

# -----------------------------
# Report
# -----------------------------
print("\n============================")
print("HNSW Results (SIFT1M - L2)")
print("============================")
print(f"Recall@10        : {recall:.4f}")
print(f"Build time (s)   : {t1 - t0:.2f}")
print(f"Search time (s)  : {t3 - t2:.2f}")
print(f"Avg latency (ms) : {1000*(t3 - t2)/len(xq):.3f}")
print(f"Index memory (MB): {mem_after - mem_before:.2f}")
print("============================")
