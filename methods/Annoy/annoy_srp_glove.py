# import numpy as np
# from annoy import AnnoyIndex
# import random
# import time
# import psutil
# import os

# # ============================================================
# # STEP 1 — SAMPLE 100 WORD VECTORS FROM GLOVE FILE
# # ============================================================

# # GLOVE_FILE = "../datasets/glove.6B/glove.6B.50d.txt"
# # SAMPLED_FILE = "glove_sample_100.txt"
# # SAMPLE_SIZE = 100

# # # Read lines
# # with open(GLOVE_FILE, "r", encoding="utf8") as f:
# #     all_lines = f.readlines()

# # # Randomly sample 100 embeddings
# # random.seed(42)
# # sampled_lines = random.sample(all_lines, SAMPLE_SIZE)

# # # Save sample to new file
# # with open(SAMPLED_FILE, "w", encoding="utf8") as f:
# #     f.writelines(sampled_lines)

# # print(f"Sampled 100 embeddings → saved to {SAMPLED_FILE}")

# SAMPLED_FILE = "glove_sample_100.txt"


# # ============================================================
# # STEP 2 — LOAD SAMPLE
# # ============================================================

# def load_glove(file):
#     word_to_idx = {}
#     vectors = []

#     with open(file, "r", encoding="utf8") as f:
#         for line in f:
#             parts = line.strip().split()
#             word = parts[0]
#             vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)

#             word_to_idx[word] = len(word_to_idx)
#             vectors.append(vec)

#     vectors = np.vstack(vectors)
#     return word_to_idx, vectors


# word_to_idx, vectors = load_glove(SAMPLED_FILE)
# idx_to_word = {v: k for k, v in word_to_idx.items()}
# DIM = vectors.shape[1]

# print("Loaded sample:")
# print("Words:", len(word_to_idx))
# print("Vector dim:", DIM)


# # ============================================================
# # STEP 3 — Normalize (cosine similarity)
# # ============================================================

# vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


# # ============================================================
# # STEP 4 — Build Annoy index
# # ============================================================

# ann = AnnoyIndex(DIM, "angular")

# for i, vec in enumerate(vectors):
#     ann.add_item(i, vec)

# print("\nBuilding Annoy index...")
# ann.build(200)
# print("Index built.")


# # ============================================================
# # STEP 5 — Nearest Neighbor Query
# # ============================================================

# def neighbors(word, k=10):
#     if word not in word_to_idx:
#         return []

#     idx = word_to_idx[word]
#     ids = ann.get_nns_by_item(idx, k, search_k=2000)
#     return [idx_to_word[i] for i in ids]


# # ============================================================
# # STEP 6 — Compute Recall & Precision
# # ============================================================

# def exact_nn(query_vec, k=10):
#     sims = vectors @ query_vec
#     return np.argsort(-sims)[:k]


# def recall_at_k(pred, truth, k):
#     return len(set(pred[:k]) & set(truth[:k])) / k


# recalls = []
# precisions = []

# query_ids = random.sample(list(range(len(vectors))), 20)

# t0 = time.time()
# for qi in query_ids:
#     q = vectors[qi]

#     gt = exact_nn(q, 10)
#     pred = ann.get_nns_by_item(qi, 10, search_k=5000)

#     r = recall_at_k(pred, gt, 10)
#     recalls.append(r)
#     precisions.append(r)  # same for ANN
# t1 = time.time()

# latency = (t1 - t0) / len(query_ids) * 1000
# memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

# print("\n============================")
# print("      ANNOY SAMPLE METRICS")
# print("============================")
# print(f"Recall@10     = {np.mean(recalls):.4f}")
# print(f"Precision@10  = {np.mean(precisions):.4f}")
# print(f"Latency/query = {latency:.4f} ms")
# print(f"Memory Used   = {memory:.2f} MB")
# print("============================")


# # ============================================================
# # STEP 7 — Example word queries
# # ============================================================

# test_words = random.sample(list(word_to_idx.keys()), 5)
# print("\nRandom example neighbors:")
# for w in test_words:
#     print(f"{w}: {neighbors(w, 5)}")


import numpy as np
from annoy import AnnoyIndex
import time
import psutil, os
import random

# -------------------------------------------------------------------
# 1. LOAD SAMPLE GLOVE EMBEDDINGS
# -------------------------------------------------------------------
def load_glove_file(path):
    words = []
    vectors = []
    
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vec = np.array(list(map(float, parts[1:])), dtype=np.float32)

            words.append(word)
            vectors.append(vec)

    vectors = np.vstack(vectors)
    print(f"Loaded {len(words)} words, dim = {vectors.shape[1]}")
    return words, vectors


# -------------------------------------------------------------------
# 2. EXACT BRUTE-FORCE TOP-k NEIGHBORS  (GROUND TRUTH)
# -------------------------------------------------------------------
def exact_topk(vectors, q_idx, k=10):
    q = vectors[q_idx]
    sims = vectors @ q  # cosine similarity because vectors are normalized
    best = np.argsort(-sims)[1:k+1]  # skip itself
    return best


# -------------------------------------------------------------------
# 3. ANNOY EVALUATION
# -------------------------------------------------------------------
def annoy_eval(words, vectors, num_trees=50, search_k=5000, k=10):
    dim = vectors.shape[1]

    # Normalize for cosine similarity
    vectors_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    # Build index
    ann = AnnoyIndex(dim, "angular")

    for i, v in enumerate(vectors_norm):
        ann.add_item(i, v)

    print(f"\nBuilding Annoy index with {num_trees} trees...")
    ann.build(num_trees)
    print("Index built.\n")

    # Evaluate recall / precision
    sample_queries = list(range(len(words)))  # all words
    random.shuffle(sample_queries)
    sample_queries = sample_queries[:50]      # evaluate on 50 queries

    recalls = []
    precisions = []

    t0 = time.time()
    for qi in sample_queries:
        gt = set(exact_topk(vectors_norm, qi, k))

        pred = ann.get_nns_by_item(qi, k, search_k=search_k)
        pred = set(pred)

        recalls.append(len(pred & gt) / k)
        precisions.append(len(pred & gt) / len(pred))
    t1 = time.time()

    latency = (t1 - t0) * 1000 / len(sample_queries)
    mem = psutil.Process(os.getpid()).memory_info().rss / (1024*1024)

    print("============================")
    print("      ANNOY METRICS")
    print("============================")
    print("Recall@10:     ", round(np.mean(recalls), 4))
    print("Precision@10:  ", round(np.mean(precisions), 4))
    print("Latency/query: ", round(latency, 4), "ms")
    print("Memory Used:   ", round(mem, 2), "MB")
    print("============================\n")

    return ann


# -------------------------------------------------------------------
# 4. NEIGHBOR DEBUGGING (HUMAN SEMANTICS)
# -------------------------------------------------------------------
def show_neighbors(ann, words, query, k=10):
    if query not in words:
        print(f"'{query}' not found in vocabulary\n")
        return

    idx = words.index(query)
    ids = ann.get_nns_by_item(idx, k, search_k=-1)
    print(f"Nearest neighbors of '{query}':")
    for j in ids:
        print("  ", words[j])
    print()


# -------------------------------------------------------------------
# RUN EVERYTHING
# -------------------------------------------------------------------
words, vectors = load_glove_file("glove_sample_100.txt")

ann = annoy_eval(words, vectors,
                 num_trees=200,
                 search_k=20000,
                 k=10)

# Try semantic words from your sample
test_words = ["small"]

for w in test_words:
    show_neighbors(ann, words, w, k=10)
