import numpy as np
import scann
import time
import psutil
import os

# =============================
# Helpers
# =============================
def read_fvecs(fname):
    with open(fname, "rb") as f:
        data = np.fromfile(f, dtype=np.int32)
        d = data[0]
        return data.reshape(-1, d + 1)[:, 1:].astype(np.float32)

def read_ivecs(fname):
    iv = np.fromfile(fname, dtype=np.int32)
    d = iv[0]
    return iv.reshape(-1, d + 1)[:, 1:]

def get_mem_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)

def compute_mips_ground_truth(xb, xq, k=10, batch_size=50):
    """
    Exact MIPS ground truth using dot product (batched).
    Returns array of shape (num_queries, k)
    """
    gt = []
    for i in range(0, len(xq), batch_size):
        q_batch = xq[i:i+batch_size]          # (B, D)
        scores = xb @ q_batch.T               # (N, B)
        topk = np.argsort(-scores, axis=0)[:k].T
        gt.append(topk)
    return np.vstack(gt)

# =============================
# Paths
# =============================
DATA_PATH = "/N/slate/mnanuva/scann/datasets/sift_1M"
GT_CACHE = "sift_mips_gt.npy"

# =============================
# Load SIFT1M
# =============================
print("Loading SIFT1M...")
xb = read_fvecs(os.path.join(DATA_PATH, "sift_base.fvecs"))    # (1M, 128)
xq = read_fvecs(os.path.join(DATA_PATH, "sift_query.fvecs"))   # (10k, 128)

print("Base:", xb.shape, "Query:", xq.shape)

# NOTE:
# Do NOT normalize — this is pure MIPS (dot product), NOT cosine.

# =============================
# Generate / Load MIPS Ground Truth
# =============================
if os.path.exists(GT_CACHE):
    print("Loading cached MIPS ground truth...")
    gt_mips = np.load(GT_CACHE)
else:
    print("Computing MIPS ground truth (this takes time)...")
    gt_mips = compute_mips_ground_truth(xb, xq, k=10, batch_size=50)
    np.save(GT_CACHE, gt_mips)
    print("Saved MIPS ground truth to disk.")

# =============================
# ScaNN Parameters (MIPS)
# =============================
num_leaves = 316
num_leaves_to_search = 40
dimensions_per_block = 2
reordering_num_neighbors = 200
k = 10

# =============================
# Build ScaNN Index
# =============================
print("\nBuilding ScaNN MIPS index...")

mem_before = get_mem_mb()
build_start = time.time()

searcher = scann.scann_ops_pybind.builder(
    xb, k, "dot_product"        # MIPS
).tree(
    num_leaves=num_leaves,
    num_leaves_to_search=num_leaves_to_search,
    training_sample_size=50000
).score_ah(
    dimensions_per_block=dimensions_per_block
).reorder(
    reordering_num_neighbors
).build()

build_end = time.time()
mem_after = get_mem_mb()

build_time = build_end - build_start
build_mem = mem_after - mem_before

# =============================
# Search (ALL 10k queries)
# =============================
print("Searching (10k queries)...")

search_start = time.time()
neighbors, distances = searcher.search_batched(
    xq,
    final_num_neighbors=k
)
search_end = time.time()

search_time = search_end - search_start

# =============================
# Recall@10 (MIPS-correct)
# =============================
recall = np.mean([
    np.intersect1d(neighbors[i], gt_mips[i]).size / k
    for i in range(len(xq))
])

# =============================
# Report
# =============================
print("\n============================")
print("ScaNN Results (SIFT1M – MIPS)")
print("============================")
print(f"Recall@10        : {recall:.4f}")
print(f"Build time (s)   : {build_time:.2f}")
print(f"Search time (s)  : {search_time:.2f}")
print(f"Avg latency (ms) : {1000 * search_time / len(xq):.3f}")
print(f"Index memory (MB): {build_mem:.2f}")
