"""Inference latency, throughput, and system resource profiler."""

import time
from collections.abc import Callable
from typing import Any

import numpy as np


class LatencyProfiler:
    """Profiles TTFT, throughput (tokens/sec), and latency percentiles."""

    def __init__(self, warmup_runs: int = 3):
        self.warmup_runs = warmup_runs

    def profile_function(
        self,
        func: Callable[[str], str],
        test_prompts: list[str],
        token_counter: Callable[[str], int] | None = None,
    ) -> dict[str, Any]:
        """
        Runs repeated executions of func across test_prompts and computes latency distribution.
        """
        if not test_prompts:
            return {}

        # Default token counter: rough whitespace/4 chars approximation if tokenizer not passed
        count_tokens = token_counter or (lambda txt: max(1, len(txt.split()) * 4 // 3))

        # Warmup
        for i in range(min(self.warmup_runs, len(test_prompts))):
            func(test_prompts[i])

        latencies_ms: list[float] = []
        token_counts: list[int] = []

        for prompt in test_prompts:
            t0 = time.perf_counter()
            output = func(prompt)
            t1 = time.perf_counter()

            latency_ms = (t1 - t0) * 1000.0
            latencies_ms.append(latency_ms)
            token_counts.append(count_tokens(output))

        latencies_arr = np.array(latencies_ms)
        total_tokens = sum(token_counts)
        total_time_sec = sum(latencies_ms) / 1000.0

        throughput_tps = total_tokens / total_time_sec if total_time_sec > 0 else 0.0

        return {
            "total_requests": len(test_prompts),
            "total_tokens_generated": total_tokens,
            "throughput_tokens_per_sec": round(float(throughput_tps), 2),
            "latency_mean_ms": round(float(np.mean(latencies_arr)), 2),
            "latency_p50_ms": round(float(np.percentile(latencies_arr, 50)), 2),
            "latency_p90_ms": round(float(np.percentile(latencies_arr, 90)), 2),
            "latency_p95_ms": round(float(np.percentile(latencies_arr, 95)), 2),
            "latency_p99_ms": round(float(np.percentile(latencies_arr, 99)), 2),
            "latency_min_ms": round(float(np.min(latencies_arr)), 2),
            "latency_max_ms": round(float(np.max(latencies_arr)), 2),
        }
