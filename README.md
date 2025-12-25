# ANN Methods for Maximum Inner Product Search (MIPS)

This directory contains reference implementations of representative **Approximate Nearest Neighbor (ANN)** methods used in our experimental study on **Maximum Inner Product Search (MIPS)**.  
The code is organized by ANN family and supports experiments on both **GloVe-100-angular** and **SIFT1M** benchmarks.

The implementations are designed to reproduce the results reported in our paper and facilitate systematic comparison across accuracy, latency, build time, and memory usage.

## 🔹 Method Descriptions

### 1. Annoy (Sign Random Projections)
**Folder:** `Annoy/`  
Implements Annoy with Sign Random Projection (SRP) hashing for approximate similarity search.

- `annoy_srp.py`: Generic SRP-based Annoy implementation  
- `annoy_srp_glove.py`: Annoy SRP configured for GloVe-100-angular (MIPS)

This method emphasizes fast indexing and simplicity but typically trades off recall.


### 2. FAISS (OPQ + PQ)
**Folder:** `FAISS/`  
Implements FAISS using Optimized Product Quantization (OPQ) followed by Product Quantization (PQ).

- `faiss_pq_glove.py`: FAISS OPQ+PQ for GloVe-100-angular  
- `faiss_pq_sift.py`: FAISS OPQ+PQ for SIFT1M (L2)

These scripts focus on **memory-efficient indexing**, reflecting classic accuracy–compression trade-offs.


### 3. HNSW (Hierarchical Navigable Small World Graphs)
**Folder:** `HNSW/`  
Graph-based ANN method using greedy search over proximity graphs.

- `hnsw_glove.py`: HNSW configured for inner product similarity (MIPS)  
- `hnsw_sift.py`: HNSW configured for Euclidean (L2) distance

HNSW prioritizes **low query latency**, often at the cost of higher memory usage.


### 4. ScaNN
**Folder:** `SCANN/`  
Implements ScaNN using hierarchical partitioning and quantization.

- `scann_glove.py`: ScaNN for MIPS on GloVe-100-angular  
- `scann_sift.py`: ScaNN for L2 search on SIFT1M

ScaNN provides strong **accuracy–efficiency trade-offs** for large-scale MIPS workloads.


## ⚙️ Reproducibility

Each script is self-contained and can be executed independently. Typical workflow:

1. Load dataset vectors and queries
2. Build ANN index
3. Perform approximate search
4. Report:
   - Recall@k
   - Index build time
   - Query latency
   - Memory usage

Example:
```bash
python methods/SCANN/scann_glove.py
```

### Notes

- Each script is dataset-specific to avoid metric mismatches.
- Inner Product search is used for MIPS benchmarks.
- Euclidean distance is used for SIFT benchmarks.
- This code is intended for **analysis and benchmarking**, not production deployment.

## Datasets

The scripts assume access to the following standard ANN benchmarks:

- **GloVe-100-angular**
  - 1.18M vectors, 100 dimensions
  - Metric: Inner Product (MIPS)
- **SIFT1M**
  - 1M vectors, 128 dimensions
  - Metric: Euclidean (L2)

Exact nearest-neighbor ground truth is required for Recall@k evaluation.


## Evaluation Metrics

All methods report the following metrics:

- **Recall@10**
- **Index build time (seconds)**
- **Average query latency (milliseconds)**
- **Total search time (seconds)**
- **Index memory footprint (MB)**

Metrics are computed under identical task definitions and similarity metrics per dataset to ensure fair comparison.


## Citation

If you use this code, please cite our paper and the original ANN methods:

- Malkov & Yashunin, *Efficient and Robust Approximate Nearest Neighbor Search Using HNSW*, TPAMI 2020  
- Guo et al., *Accelerating Large-Scale Inference with ScaNN*, ICML 2020  
- Johnson et al., *Billion-Scale Similarity Search with GPUs*, IEEE BigData 2017  
- Charikar, *Similarity Estimation Techniques from Rounding Algorithms*, STOC 2002


