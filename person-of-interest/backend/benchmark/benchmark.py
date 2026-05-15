"""End-to-end latency benchmark for POI Re-identification System.

Measures:
  1. FAISS add latency
  2. FAISS search latency (varying index sizes)
  3. Matching pipeline latency (cache miss + cache hit)
  4. Full event processing latency

Usage:
  cd backend && python -m backend.benchmark.benchmark
"""

from __future__ import annotations

import os
import statistics
import sys
import tempfile
import time
from unittest.mock import MagicMock, patch

import numpy as np

DIM = 256
WARMUP_ROUNDS = 5
BENCHMARK_ROUNDS = 100


def _setup():
    """Patch config and create a fresh FAISS repo for benchmarking."""
    tmp = tempfile.mkdtemp()
    cfg = MagicMock()
    cfg.faiss_dimension = DIM
    cfg.faiss_index_path = os.path.join(tmp, "bench.index")
    cfg.faiss_id_map_path = os.path.join(tmp, "bench_map.json")
    cfg.similarity_threshold = 0.6
    cfg.search_top_k = 10
    cfg.object_cache_ttl = 300
    cfg.benchmark_latency = False

    return cfg, tmp


def bench_faiss_add(sizes=(100, 500, 1000, 5000)):
    """Benchmark FAISS add latency at various index sizes."""
    print("\n=== FAISS Add Latency ===")
    print(f"{'Vectors':>10} {'Batch':>8} {'Mean (ms)':>12} {'Std (ms)':>10}")
    print("-" * 50)

    for size in sizes:
        with patch("backend.infrastructure.faiss.repository.get_config") as mock_cfg:
            cfg, _ = _setup()
            mock_cfg.return_value = cfg
            from backend.infrastructure.faiss.repository import FAISSRepository
            FAISSRepository._instance = None
            repo = FAISSRepository()

        batch_size = 10
        vectors = [np.random.randn(DIM).astype(np.float32) for _ in range(size)]

        # Add in batches and time each
        durations = []
        for i in range(0, size, batch_size):
            batch = vectors[i : i + batch_size]
            t0 = time.perf_counter()
            repo.add(f"poi-bench-{i}", batch)
            elapsed = (time.perf_counter() - t0) * 1000
            durations.append(elapsed)

        FAISSRepository._instance = None
        print(
            f"{size:>10} {batch_size:>8} {statistics.mean(durations):>12.3f} {statistics.stdev(durations) if len(durations) > 1 else 0:>10.3f}"
        )


def bench_faiss_search(sizes=(100, 500, 1000, 5000, 10000)):
    """Benchmark FAISS search latency at various index sizes."""
    print("\n=== FAISS Search Latency ===")
    print(f"{'Index Size':>12} {'top_k':>6} {'Mean (ms)':>12} {'P95 (ms)':>10} {'P99 (ms)':>10}")
    print("-" * 60)

    for size in sizes:
        with patch("backend.infrastructure.faiss.repository.get_config") as mock_cfg:
            cfg, _ = _setup()
            mock_cfg.return_value = cfg
            from backend.infrastructure.faiss.repository import FAISSRepository
            FAISSRepository._instance = None
            repo = FAISSRepository()

        # Populate index
        vectors = np.random.randn(size, DIM).astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vectors /= norms
        ids = np.arange(size, dtype=np.int64)
        repo._index.add_with_ids(vectors, ids)
        for i in range(size):
            repo._id_map[i] = f"poi-{i}"
        repo._next_id = size

        # Warmup
        for _ in range(WARMUP_ROUNDS):
            q = np.random.randn(DIM).astype(np.float32)
            repo.search(q, top_k=10)

        # Benchmark
        durations = []
        for _ in range(BENCHMARK_ROUNDS):
            q = np.random.randn(DIM).astype(np.float32)
            t0 = time.perf_counter()
            repo.search(q, top_k=10)
            elapsed = (time.perf_counter() - t0) * 1000
            durations.append(elapsed)

        durations.sort()
        p95 = durations[int(0.95 * len(durations))]
        p99 = durations[int(0.99 * len(durations))]

        FAISSRepository._instance = None
        print(
            f"{size:>12} {10:>6} {statistics.mean(durations):>12.3f} {p95:>10.3f} {p99:>10.3f}"
        )


def bench_matching_pipeline():
    """Benchmark the full matching pipeline: FAISS search + cache-aside."""
    print("\n=== Matching Pipeline Latency ===")

    with patch("backend.infrastructure.faiss.repository.get_config") as mock_cfg:
        cfg, _ = _setup()
        mock_cfg.return_value = cfg
        from backend.infrastructure.faiss.repository import FAISSRepository
        FAISSRepository._instance = None
        repo = FAISSRepository()

    # Populate with 1000 POIs
    for i in range(1000):
        v = np.random.randn(DIM).astype(np.float32)
        repo.add(f"poi-{i}", [v])

    from backend.strategy.matching import CosineSimilarityStrategy

    strategy = CosineSimilarityStrategy(repo)

    # Mock cache
    cache = MagicMock()
    cache.get_poi_for_object.return_value = None  # Always miss

    with patch("backend.service.matching_service.get_config") as mock_cfg2:
        mock_cfg2.return_value = cfg
        from backend.service.matching_service import MatchingService
        service = MatchingService(strategy, cache)

    # Warmup
    for _ in range(WARMUP_ROUNDS):
        q = np.random.randn(DIM).astype(np.float32).tolist()
        service.match_object(f"warmup-{_}", q)

    # Cache MISS benchmark
    durations_miss = []
    for i in range(BENCHMARK_ROUNDS):
        q = np.random.randn(DIM).astype(np.float32).tolist()
        t0 = time.perf_counter()
        service.match_object(f"bench-miss-{i}", q)
        elapsed = (time.perf_counter() - t0) * 1000
        durations_miss.append(elapsed)

    # Cache HIT benchmark
    cache.get_poi_for_object.return_value = "poi-0"
    durations_hit = []
    for i in range(BENCHMARK_ROUNDS):
        q = np.random.randn(DIM).astype(np.float32).tolist()
        t0 = time.perf_counter()
        service.match_object(f"bench-hit-{i}", q)
        elapsed = (time.perf_counter() - t0) * 1000
        durations_hit.append(elapsed)

    durations_miss.sort()
    durations_hit.sort()

    print(f"  Cache MISS (FAISS search):")
    print(f"    Mean: {statistics.mean(durations_miss):.3f} ms")
    print(f"    P95:  {durations_miss[int(0.95 * len(durations_miss))]:.3f} ms")
    print(f"    P99:  {durations_miss[int(0.99 * len(durations_miss))]:.3f} ms")
    print(f"  Cache HIT (skip FAISS):")
    print(f"    Mean: {statistics.mean(durations_hit):.3f} ms")
    print(f"    P99:  {durations_hit[int(0.99 * len(durations_hit))]:.3f} ms")

    FAISSRepository._instance = None


def main():
    print("=" * 60)
    print("  POI Re-identification System — Latency Benchmark")
    print("=" * 60)

    bench_faiss_add()
    bench_faiss_search()
    bench_matching_pipeline()

    print("\n" + "=" * 60)
    print("  Benchmark complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
