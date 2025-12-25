"""
==================== QFIPS (Quantization-based Fast Inner Product Search) ====================

Goal:
    Approximate Maximum Inner Product Search (MIPS) using Product Quantization + LUT scoring.

Hyperparameters:
    D      = vector dimension (e.g., 128 for SIFT)
    M      = number of PQ subspaces  (paper: {4, 8, 16, 32})
    nbits  = bits per subspace       (paper: {4, 8})
    C      = number of centroids per subspace = 2^nbits
    b      = total bits per vector = M * nbits   (paper plots use b ∈ {64,128,256,512})

----------------------------------------------------------------------------------------------
Training Phase:
----------------------------------------------------------------------------------------------

1. Split each database vector x into M equal blocks:
       x → [x^(1), x^(2), ..., x^(M)]

2. For each block k = 1..M:
       Train a codebook U^(k) with C centroids using k-means.

3. Encode each vector x_i:
       For each block k:
            α_i^(k) = index of closest centroid in U^(k)
       Store PQ code:  Code(x_i) = [α_i^(1), ..., α_i^(M)]

----------------------------------------------------------------------------------------------
Query Phase:
----------------------------------------------------------------------------------------------

4. Split query q into M blocks: q^(1), ..., q^(M)

5. Build lookup tables (LUT):
       For each block k:
           For each centroid c in {1..C}:
               LUT[k][c] = dot( q^(k), U^(k)[c] )

6. Compute approximate inner product for each database vector:
       score(x_i) = sum over k of ( LUT[k][ α_i^(k) ] )

7. Return the Top-K database vectors by score.

==============================================================================================
"""


"""
==================== QFIPS (Quantization-based Fast Inner Product Search) ====================

Implements MIPS using Product Quantization (PQ) + lookup-table (LUT) scoring.
Metrics reported:
    - Recall@10
    - Build Time
    - Search Time
    - Latency/query
    - Memory Used (MB)
"""

import numpy as np
import faiss
import time, psutil, os

# -----------------------------------------------------
# Memory Utility
# -----------------------------------------------------
def memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


# -----------------------------------------------------
# Helper: Load SIFT dataset in fvecs/ivecs format
# -----------------------------------------------------
def read_fvecs(fname):
    a = np.fromfile(fname, dtype=np.int32)
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].view("float32")

def read_ivecs(fname):
    a = np.fromfile(fname, dtype=np.int32)
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:]


# -----------------------------------------------------
# Load SIFT1M dataset
# -----------------------------------------------------
print("Loading SIFT1M dataset...")
dataset_path = "../datasets/sift_1M/"

xb = read_fvecs(dataset_path + "sift_base.fvecs")
xq = read_fvecs(dataset_path + "sift_query.fvecs")
gt = read_ivecs(dataset_path + "sift_groundtruth.ivecs")[:, :10]

xb = np.ascontiguousarray(xb, dtype="float32")
xq = np.ascontiguousarray(xq, dtype="float32")

print("Base vectors:", xb.shape)
print("Query vectors:", xq.shape)
print("Groundtruth:", gt.shape)


# -----------------------------------------------------
# QFIPS Parameters
# -----------------------------------------------------
d = xb.shape[1]       # 128
M = 128               # Number of PQ subquantizers
nbits = 8             # Bits per subquantizer
k = 10                # Recall@10


print("\n====================================")
print("     Building QFIPS (PQ + Inner Product)")
print("====================================")

# -----------------------------------------------------
# Measure memory before PQ index
# -----------------------------------------------------
mem_before = memory_mb()

# -----------------------------------------------------
# Create PQ index
# -----------------------------------------------------
pq_index = faiss.IndexPQ(d, M, nbits, faiss.METRIC_INNER_PRODUCT)

# -----------------------------------------------------
# Train PQ codebooks
# -----------------------------------------------------
print("\nTraining PQ codebooks...")
t_train0 = time.time()
pq_index.train(xb)
t_train1 = time.time()

# -----------------------------------------------------
# Add vectors
# -----------------------------------------------------
print("Encoding + Adding base vectors...")
t_add0 = time.time()
pq_index.add(xb)
t_add1 = time.time()

# Compute real index memory by saving to disk
faiss.write_index(pq_index, "tmp_index.faiss")
index_mem = os.path.getsize("tmp_index.faiss") / (1024 * 1024)   # MB
os.remove("tmp_index.faiss")


# -----------------------------------------------------
# SEARCH
# -----------------------------------------------------
print("\nRunning QFIPS search...")
t_search0 = time.time()
D, I = pq_index.search(xq, k)
t_search1 = time.time()

search_time = t_search1 - t_search0
latency_per_query = search_time / len(xq)


# -----------------------------------------------------
# Recall@10
# -----------------------------------------------------
recall = (I == gt).sum() / (len(xq) * k)


# -----------------------------------------------------
# Print Results
# -----------------------------------------------------
print("\n============================")
print("        QFIPS METRICS")
print("============================")
print(f"Recall@10        = {recall:.4f}")
print(f"Build Time       = {t_add1 - t_train0:.4f} sec")
print(f"Search Time      = {search_time:.4f} sec for {len(xq)} queries")
print(f"Latency/query    = {latency_per_query*1000:.4f} ms")
print(f"Index Memory     = {index_mem:.2f} MB")
print("============================")
