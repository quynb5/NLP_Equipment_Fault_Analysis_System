"""
TASK 05 — Latency Measurement (CPU)
====================================
Đo thời gian xử lý trung bình cho mỗi câu mô tả trên CPU.
Hỗ trợ multi-engine.
"""
import time
import numpy as np
from backend.core.engine_factory import get_engine


def measure_latency(texts: list[str], engine_name: str = "phobert") -> dict:
    """
    Đo latency cho từng câu trong danh sách texts.

    Args:
        texts: Danh sách các mô tả cần đo
        engine_name: "phobert" hoặc "tfidf"

    Returns:
        dict chứa mean, min, max, p95 latency (ms) và danh sách latency từng câu
    """
    engine = get_engine(engine_name)
    latencies = []

    for text in texts:
        start = time.perf_counter()
        engine.analyze(equipment="Thiết bị", description=text)
        end = time.perf_counter()
        latency_ms = (end - start) * 1000  # Convert to milliseconds
        latencies.append(latency_ms)

    latencies_arr = np.array(latencies)

    stats = {
        "mean_ms": round(float(np.mean(latencies_arr)), 2),
        "min_ms": round(float(np.min(latencies_arr)), 2),
        "max_ms": round(float(np.max(latencies_arr)), 2),
        "p95_ms": round(float(np.percentile(latencies_arr, 95)), 2),
        "total_samples": len(latencies),
        "all_latencies_ms": [round(l, 2) for l in latencies],
    }

    return stats


def print_latency_stats(stats: dict):
    """In latency stats ra console."""
    print("\n" + "=" * 50)
    print("  ⏱️  LATENCY STATS (CPU)")
    print("=" * 50)
    print(f"  Mean:    {stats['mean_ms']:.2f} ms")
    print(f"  Min:     {stats['min_ms']:.2f} ms")
    print(f"  Max:     {stats['max_ms']:.2f} ms")
    print(f"  P95:     {stats['p95_ms']:.2f} ms")
    print(f"  Samples: {stats['total_samples']}")
    print("=" * 50)
