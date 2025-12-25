'''
PAPER:
    "Similarity Estimation Techniques from Rounding Algorithms"
     (Moses Charikar, STOC 2002)
     — Core algorithm behind Signed Random Projections (SRP)
     — Used in Spotify's Annoy library for angular similarity search

LIBRARY:
    Annoy (Approximate Nearest Neighbors Oh Yeah)
    — Open-source C++/Python library used by Spotify
    — Implements forest of SRP-based binary trees

KEY PARAMETERS:
    n_trees (default: 10)
        • Number of random projection trees
        • More trees → higher recall, slower build, larger index
        • Fewer trees → faster but less accurate

    search_k (default: n_trees * k)
        • Number of nodes inspected during search
        • Higher → better recall, slower search
        • Lower → faster, but may miss neighbors

    metric ("angular", "euclidean", "manhattan")
        • Angular metric uses SRP hashing (cosine similarity)
        • Best performance = "angular"

ALGORITHM (SRP + Random Projection Trees):
    • SRP hashes each vector v into a binary signature:
        sign(v · r1), sign(v · r2), … sign(v · rd)
      where r_i are random hyperplanes.

    • Annoy builds multiple random-projection trees:
        - Each tree recursively splits the data using random directions
        - Leaves contain buckets of candidate vectors

    • During query:
        - Query is routed down all n_trees
        - Candidates from leaf nodes are collected
        - Final candidates re-ranked using actual distances

PSEUDOCODE:
    Load base vectors Xb, queries Xq, and GT
    Initialize Annoy: AnnoyIndex(dim, metric="angular")
    For each vector:
        add_item(index, vector)
    Build index with n_trees
    For each query:
        get_nns_by_vector(query, k, search_k)
    Compute metrics:
        - Recall@10
        - Precision@10
        - Build time
        - Search latency
        - Index size (memory footprint)
'''

import numpy as np
from annoy import AnnoyIndex
import time
import psutil
import os

# -------------------------------------------------------
# Helper: read SIFT .fvecs / .ivecs
# -------------------------------------------------------
def read_fvecs(path):
    a = np.fromfile(path, dtype=np.int32)
    d = a[0]
    return a.reshape(-1, d+1)[:, 1:].astype(np.float32)

def read_ivecs(path):
    a = np.fromfile(path, dtype=np.int32)
    d = a[0]
    return a.reshape(-1, d+1)[:, 1:]

# -------------------------------------------------------
# Load SIFT1M dataset
# -------------------------------------------------------
path = "../datasets/sift_1M/"

base = read_fvecs(path + "sift_base.fvecs")
query = read_fvecs(path + "sift_query.fvecs")
gt = read_ivecs(path + "sift_groundtruth.ivecs")

print("Loaded SIFT1M:")
print("Base:", base.shape, "Query:", query.shape)

# Normalize for angular distance
base = base / np.linalg.norm(base, axis=1, keepdims=True)
query = query / np.linalg.norm(query, axis=1, keepdims=True)

dim = base.shape[1]
ann = AnnoyIndex(dim, metric='angular')

# -------------------------------------------------------
# Build ANNoy index (no timing)
# -------------------------------------------------------
num_trees = 1000
for i, vec in enumerate(base):
    ann.add_item(i, vec)
ann.build(num_trees)

# -------------------------------------------------------
# Memory Usage
# -------------------------------------------------------
process = psutil.Process(os.getpid())
memory_used = process.memory_info().rss / (1024**2)

# -------------------------------------------------------
# Metric Functions
# -------------------------------------------------------
def recall_at_k(pred, truth, k=10):
    return len(set(pred[:k]) & set(truth[:k])) / k

def precision_at_k(pred, truth, k=10):
    return len(set(pred[:k]) & set(truth[:k])) / k

# -------------------------------------------------------
# Search Evaluation
# -------------------------------------------------------
k = 10
search_k = 5_000_000   # large → higher recall

recalls = []
precisions = []

t0 = time.time()

for qi in range(len(query)):
    retrieved = ann.get_nns_by_vector(query[qi], k, search_k=search_k)
    recalls.append(recall_at_k(retrieved, gt[qi], k))
    precisions.append(precision_at_k(retrieved, gt[qi], k))

t1 = time.time()

search_time = t1 - t0
latency_ms = (search_time / len(query)) * 1000

# -------------------------------------------------------
# Final Output (Clean)
# -------------------------------------------------------
print("\n============================")
print("        ANNOY METRICS")
print("============================")
print(f"Recall@10     = {np.mean(recalls):.4f}")
print(f"Precision@10  = {np.mean(precisions):.4f}")
print(f"Search Time   = {search_time:.4f} sec for {len(query)} queries")
print(f"Latency/query = {latency_ms:.4f} ms")
print(f"Memory Used   = {memory_used:.2f} MB")
print("============================\n")
