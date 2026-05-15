"""Stream density benchmark — measures throughput under concurrent object load.

Simulates multiple cameras producing person events simultaneously and
measures the matching pipeline's ability to keep up.

Usage:
  cd backend && python -m backend.benchmark.stream_density
"""

from __future__ import annotations

import os
import statistics
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import numpy as np

DIM = 256


def _setup_faiss(num_pois=1000):
    """Create a populated FAISS repo for benchmarking."""
    tmp = tempfile.mkdtemp()
    cfg = MagicMock()
    cfg.faiss_dimension = DIM
    cfg.faiss_index_path = os.path.join(tmp, "stream.index")
    cfg.faiss_id_map_path = os.path.join(tmp, "stream_map.json")
    cfg.similarity_threshold = 0.6
    cfg.search_top_k = 10
    cfg.object_cache_ttl = 300
    cfg.benchmark_latency = False

    with patch("backend.infrastructure.faiss.repository.get_config") as mock_cfg:
        mock_cfg.return_value = cfg
        from backend.infrastructure.faiss.repository import FAISSRepository
        FAISSRepository._instance = None
        repo = FAISSRepository()

    for i in range(num_pois):
        v = np.random.randn(DIM).astype(np.float32)
        repo.add(f"poi-{i}", [v])

    return repo, cfg


def bench_stream_density(camera_counts=(1, 4, 8, 16), objects_per_camera=50):
    """Simulate concurrent camera streams and measure throughput."""
    print("\n=== Stream Density Benchmark ===")
    print(f"Objects per camera: {objects_per_camera}")
    print(f"POIs in index: 1000")
    print(f"{'Cameras':>10} {'Total Objs':>12} {'Total (s)':>10} {'Throughput':>14} {'Avg (ms)':>10}")
    print("-" * 65)

    repo, cfg = _setup_faiss(num_pois=1000)

    from backend.strategy.matching import CosineSimilarityStrategy

    strategy = CosineSimilarityStrategy(repo)

    cache = MagicMock()
    cache.get_poi_for_object.return_value = None

    with patch("backend.service.matching_service.get_config") as mock_cfg:
        mock_cfg.return_value = cfg
        from backend.service.matching_service import MatchingService
        service = MatchingService(strategy, cache)

    def process_objects(camera_id: int, count: int):
        durations = []
        for j in range(count):
            emb = np.random.randn(DIM).astype(np.float32).tolist()
            t0 = time.perf_counter()
            service.match_object(f"cam{camera_id}-obj{j}", emb)
            elapsed = (time.perf_counter() - t0) * 1000
            durations.append(elapsed)
        return durations

    for num_cameras in camera_counts:
        total_objects = num_cameras * objects_per_camera

        t_start = time.perf_counter()
        all_durations = []

        with ThreadPoolExecutor(max_workers=num_cameras) as executor:
            futures = []
            for cam_id in range(num_cameras):
                futures.append(executor.submit(process_objects, cam_id, objects_per_camera))

            for f in futures:
                all_durations.extend(f.result())

        t_total = time.perf_counter() - t_start
        throughput = total_objects / t_total

        print(
            f"{num_cameras:>10} {total_objects:>12} {t_total:>10.3f} {throughput:>12.1f}/s {statistics.mean(all_durations):>10.3f}"
        )

    from backend.infrastructure.faiss.repository import FAISSRepository
    FAISSRepository._instance = None


def main():
    print("=" * 65)
    print("  POI Re-identification System — Stream Density Benchmark")
    print("=" * 65)

    bench_stream_density()

    print("\n" + "=" * 65)
    print("  Benchmark complete.")
    print("=" * 65)


if __name__ == "__main__":
    main()
