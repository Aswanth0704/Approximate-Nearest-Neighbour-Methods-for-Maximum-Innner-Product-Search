import numpy as np
import h5py
import requests
import tempfile
import time
import scann
import psutil
import os

# -----------------------------
# Helpers
# -----------------------------
def get_mem_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)

def recall_at_k(neighbors, gt, k):
    return np.mean([
        np.intersect1d(neighbors[i], gt[i]).size / k
        for i in range(len(gt))
    ])

# -----------------------------
# Download ANN-Benchmarks GloVe-100
# -----------------------------
with tempfile.TemporaryDirectory() as tmp:
    url = "http://ann-benchmarks.com/glove-100-angular.hdf5"
    path = os.path.join(tmp, "glove100.hdf5")
    with open(path, "wb") as f:
        f.write(requests.get(url).content)

    f = h5py.File(path, "r")
    xb = f["train"][:]          # (1,183,514, 100)
    xq = f["test"][:]           # (10,000, 100)
    gt = f["neighbors"][:, :10] # ground truth @10

print("Base:", xb.shape, "Query:", xq.shape)

# -----------------------------
# Normalize (IMPORTANT)
# -----------------------------
xb /= np.linalg.norm(xb, axis=1, keepdims=True)
xq /= np.linalg.norm(xq, axis=1, keepdims=True)

# -----------------------------
# ScaNN parameters
# -----------------------------
k = 10
num_leaves = 2000
num_leaves_to_search = 100
dimensions_per_block = 2
reorder_k = 100

# -----------------------------
# Build index
# -----------------------------
print("\nBuilding ScaNN index...")
mem_before = get_mem_mb()
t0 = time.time()

searcher = scann.scann_ops_pybind.builder(
    xb, k, "dot_product"
).tree(
    num_leaves=num_leaves,
    num_leaves_to_search=num_leaves_to_search,
    training_sample_size=250_000
).score_ah(
    dimensions_per_block=dimensions_per_block,
    anisotropic_quantization_threshold=0.2
).reorder(
    reorder_k
).build()

t1 = time.time()
mem_after = get_mem_mb()

# -----------------------------
# Search (10k queries)
# -----------------------------
print("Searching...")
t2 = time.time()
neighbors, distances = searcher.search_batched(
    xq, final_num_neighbors=k
)
t3 = time.time()

# -----------------------------
# Recall
# -----------------------------
recall = recall_at_k(neighbors, gt, k)

# -----------------------------
# Report
# -----------------------------
print("\n============================")
print("ScaNN Results (GloVe-100-angular)")
print("============================")
print(f"Recall@10        : {recall:.4f}")
print(f"Build time (s)   : {t1 - t0:.2f}")
print(f"Search time (s)  : {t3 - t2:.2f}")
print(f"Avg latency (ms) : {1000 * (t3 - t2) / len(xq):.3f}")
print(f"Index memory (MB): {mem_after - mem_before:.2f}")
