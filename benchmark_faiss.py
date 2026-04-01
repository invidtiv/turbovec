"""
FAISS PQ benchmark for comparison with py-turboquant.

Uses IndexPQFastScan (SIMD-accelerated PQ) on the same datasets.
Tests both single-threaded and multi-threaded performance.

Usage:
  python3 benchmark_faiss.py openai-1536
"""

import os
import sys
import time

import faiss
import numpy as np

from benchmark import load_openai, load_glove, recall_at_1_at_k

DATASETS = {
    "glove": ("GloVe d=200 (100K vectors, 10K queries)", load_glove),
    "openai-1536": ("OpenAI DBpedia d=1536 (100K vectors, 1K queries)", lambda: load_openai(1536)),
    "openai-3072": ("OpenAI DBpedia d=3072 (100K vectors, 1K queries)", lambda: load_openai(3072)),
}


def bench_search(index, queries, k, n_runs=25):
    # Warmup
    index.search(queries[:1], k)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        scores, indices = index.search(queries, k)
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2], scores, indices


def run_faiss_benchmark(database, queries, label=""):
    n, dim = database.shape
    n_queries = len(queries)

    # Ground truth (brute force)
    true_top1 = np.argmax(queries @ database.T, axis=1)

    # Match TurboQuant compression ratios:
    # TQ 2-bit: 2 bits/dim = 384 bytes/vec → PQ768x4: 768 subs × 4 bits = 384 bytes ✓
    # TQ 4-bit: 4 bits/dim = 768 bytes/vec → PQ1536x4: 1536 subs × 4 bits = 768 bytes ✓
    # (FastScan needs m % 32 == 0; both 768 and 1536 qualify)
    configs = [
        ("PQ768x4 (=TQ 2-bit)", 768, 4),
        ("PQ1536x4 (=TQ 4-bit)", 1536, 4),
    ]

    k = 64

    for config_label, m, nbits in configs:
        print(f"\n  {config_label}: m={m}, nbits={nbits}")

        # Build index
        t0 = time.time()
        index = faiss.IndexPQFastScan(dim, m, nbits)
        index.train(database)
        index.add(database)
        build_time = time.time() - t0
        print(f"    Build: {build_time*1000:.0f}ms")

        # Multi-threaded (default)
        n_threads = faiss.omp_get_max_threads()
        search_time, scores, indices = bench_search(index, queries, k)
        ms_multi = search_time / n_queries * 1000
        print(f"    Multi-threaded ({n_threads} threads): {ms_multi:.3f}ms/q (median of 25 runs)")

        # Single-threaded
        faiss.omp_set_num_threads(1)
        search_time, _, _ = bench_search(index, queries, k)
        ms_single = search_time / n_queries * 1000
        print(f"    Single-threaded: {ms_single:.3f}ms/q (median of 25 runs)")
        faiss.omp_set_num_threads(n_threads)

        # Recall
        for rk in [1, 4, 16, 64]:
            r = recall_at_1_at_k(true_top1, indices, rk)
            print(f"    recall@1/{rk}: {r:.4f}")


if __name__ == "__main__":
    args = sys.argv[1:] if len(sys.argv) > 1 else ["openai-1536"]

    print("FAISS PQ benchmark")
    print("=" * 60)

    for name in args:
        if name not in DATASETS:
            print(f"Unknown dataset: {name}")
            continue

        label, loader = DATASETS[name]
        print(f"\nDataset: {label}")
        database, queries = loader()
        n, dim = database.shape
        print(f"Database: {n:,} x {dim}, Queries: {len(queries):,}")

        run_faiss_benchmark(database, queries, name)

    print("\nDone.")
