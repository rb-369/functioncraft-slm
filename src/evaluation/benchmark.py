"""Orchestrator for the FunctionCraft-SLM benchmark evaluation suite."""

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.common.logger import setup_logger
from src.evaluation.judge import LLMJudge
from src.evaluation.latency_profiler import LatencyProfiler
from src.evaluation.metrics import BenchmarkMetrics

logger = setup_logger("benchmark")


class BenchmarkRunner:
    """Runs end-to-end evaluation comparing base model, SFT model, and DPO aligned model."""

    def __init__(
        self,
        schemas_dir: str = "data/schemas",
        output_dir: str = "outputs/evaluation",
    ):
        self.schemas_dir = Path(schemas_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.schemas: dict[str, dict[str, Any]] = {}
        self._load_schemas()
        self.metrics = BenchmarkMetrics(self.schemas)
        self.latency_profiler = LatencyProfiler()
        self.judge = LLMJudge()

    def _load_schemas(self) -> None:
        if self.schemas_dir.exists():
            for p in self.schemas_dir.glob("*.json"):
                with open(p, encoding="utf-8") as f:
                    data = json.load(f)
                    if "name" in data:
                        self.schemas[data["name"]] = data

    def run_benchmark(
        self,
        predict_fn: Callable[[str], str],
        eval_samples: list[dict[str, Any]],
        model_name: str = "functioncraft-slm",
    ) -> dict[str, Any]:
        """
        Runs predictions over eval_samples and calculates comprehensive quality and latency metrics.
        """
        logger.info("Starting benchmark run for: %s (samples: %d)", model_name, len(eval_samples))

        sample_evals = []
        prompts = [s["prompt"] for s in eval_samples]

        # 1. Quality evaluation
        for s in eval_samples:
            prompt = s["prompt"]
            expected_tool = s["expected_tool"]
            expected_args = s["expected_arguments"]

            raw_pred = predict_fn(prompt)
            eval_res = self.metrics.evaluate_sample(raw_pred, expected_tool, expected_args)
            sample_evals.append(eval_res)

        agg_metrics = self.metrics.aggregate(sample_evals)

        # 2. Latency and throughput profiling
        latency_stats = self.latency_profiler.profile_function(predict_fn, prompts[:25] if len(prompts) > 25 else prompts)

        benchmark_result = {
            "model_name": model_name,
            "metrics": agg_metrics,
            "latency": latency_stats,
        }

        self._save_reports(benchmark_result)
        return benchmark_result

    def _save_reports(self, result: dict[str, Any]) -> None:
        """Saves JSON results and generates a Markdown report."""
        json_path = self.output_dir / "benchmark_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        md_path = self.output_dir / "benchmark_report.md"
        m = result["metrics"]
        lat = result["latency"]

        md_content = f"""# FunctionCraft-SLM Benchmark Report

**Model:** `{result['model_name']}`
**Total Test Samples:** {m.get('total_samples', 0)}

## 1. Structured Output Quality & Schema Adherence
| Metric | Value | Production Threshold | Status |
| :--- | :--- | :--- | :--- |
| **JSON Validity Rate** | {m.get('json_validity_rate', 0.0) * 100:.1f}% | ≥ 98.0% | {'✅ PASS' if m.get('json_validity_rate', 0) >= 0.98 else '⚠️ WARN'} |
| **Schema Adherence Rate** | {m.get('schema_adherence_rate', 0.0) * 100:.1f}% | ≥ 95.0% | {'✅ PASS' if m.get('schema_adherence_rate', 0) >= 0.95 else '⚠️ WARN'} |
| **Tool Selection Accuracy** | {m.get('tool_accuracy', 0.0) * 100:.1f}% | ≥ 92.0% | {'✅ PASS' if m.get('tool_accuracy', 0) >= 0.92 else '⚠️ WARN'} |
| **Argument Exact Match** | {m.get('argument_exact_match_rate', 0.0) * 100:.1f}% | ≥ 85.0% | {'✅ PASS' if m.get('argument_exact_match_rate', 0) >= 0.85 else 'ℹ️ INFO'} |
| **Argument Key F1** | {m.get('argument_key_f1', 0.0):.3f} | ≥ 0.90 | {'✅ PASS' if m.get('argument_key_f1', 0) >= 0.90 else '⚠️ WARN'} |
| **Composite Quality Score** | {m.get('composite_slm_score', 0.0):.3f} | ≥ 0.90 | {'✅ PASS' if m.get('composite_slm_score', 0) >= 0.90 else '⚠️ WARN'} |

## 2. Serving Latency & Throughput
| Latency Metric | Measured Value | Target SLA |
| :--- | :--- | :--- |
| **Mean Latency** | {lat.get('latency_mean_ms', 0.0):.1f} ms | < 250 ms |
| **p50 Latency** | {lat.get('latency_p50_ms', 0.0):.1f} ms | < 200 ms |
| **p95 Latency** | {lat.get('latency_p95_ms', 0.0):.1f} ms | < 450 ms |
| **p99 Latency** | {lat.get('latency_p99_ms', 0.0):.1f} ms | < 600 ms |
| **Token Throughput** | {lat.get('throughput_tokens_per_sec', 0.0):.1f} tokens/s | > 80 tokens/s |
"""
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        logger.info("Saved benchmark report to: %s", md_path)


def run_cli():
    parser = argparse.ArgumentParser(description="Run benchmark suite on test dataset")
    parser.add_argument("--test-data", type=str, default="data/processed/eval_test.json")
    parser.add_argument("--schemas", type=str, default="data/schemas")
    parser.add_argument("--output", type=str, default="outputs/evaluation")
    args = parser.parse_args()

    runner = BenchmarkRunner(args.schemas, args.output)

    # Load test data
    test_path = Path(args.test_data)
    if not test_path.exists():
        logger.warning("Test dataset not found at %s. Generating on the fly...", test_path)
        from src.data.generator import SyntheticDataEngine
        engine = SyntheticDataEngine(args.schemas)
        dataset = engine.generate_dataset(samples_per_tool=15)
        eval_samples = dataset["eval_test"]
    else:
        with open(test_path, encoding="utf-8") as f:
            eval_samples = json.load(f)

    # Mock prediction function for CLI run
    def mock_predict(query: str) -> str:
        for tool, _schema in runner.schemas.items():
            return f'```json\n{{"name": "{tool}", "arguments": {{}}}}\n```'
        return "{}"

    runner.run_benchmark(mock_predict, eval_samples)


if __name__ == "__main__":
    run_cli()
