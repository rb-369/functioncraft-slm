"""Tests for evaluation metrics, latency profiler, and benchmark runner."""

import pytest

from src.evaluation.benchmark import BenchmarkRunner
from src.evaluation.judge import LLMJudge
from src.evaluation.latency_profiler import LatencyProfiler
from src.evaluation.metrics import BenchmarkMetrics


@pytest.fixture
def dummy_schemas():
    return {
        "dummy_tool": {
            "name": "dummy_tool",
            "parameters": {
                "type": "object",
                "properties": {
                    "param_a": {"type": "string"},
                    "count": {"type": "integer"},
                },
                "required": ["param_a", "count"],
            },
        }
    }


def test_benchmark_metrics_calculation(dummy_schemas):
    metrics = BenchmarkMetrics(dummy_schemas)

    raw_output = '{"name": "dummy_tool", "arguments": {"param_a": "test", "count": 10}}'
    res = metrics.evaluate_sample(
        raw_output=raw_output,
        expected_tool="dummy_tool",
        expected_args={"param_a": "test", "count": 10},
    )

    assert res["valid_json"] is True
    assert res["valid_schema"] is True
    assert res["tool_correct"] is True
    assert res["arg_exact_match"] is True
    assert res["arg_key_f1"] == 1.0

    aggregated = metrics.aggregate([res])
    assert aggregated["json_validity_rate"] == 1.0
    assert aggregated["schema_adherence_rate"] == 1.0
    assert aggregated["composite_slm_score"] == 1.0


def test_latency_profiler():
    profiler = LatencyProfiler(warmup_runs=1)

    def dummy_inference(prompt: str) -> str:
        return '{"result": "ok"}'

    prompts = ["p1", "p2", "p3", "p4", "p5"]
    stats = profiler.profile_function(dummy_inference, prompts)

    assert stats["total_requests"] == 5
    assert stats["latency_mean_ms"] >= 0.0
    assert "latency_p50_ms" in stats
    assert "latency_p99_ms" in stats
    assert "throughput_tokens_per_sec" in stats


def test_llm_judge_heuristic():
    judge = LLMJudge()
    expected = {"name": "dummy_tool", "arguments": {"param_a": "hello"}}
    generated = '{"name": "dummy_tool", "arguments": {"param_a": "hello"}}'

    score_dict = judge.evaluate_pair("test query", expected, generated)
    assert score_dict["score"] == 5
    assert "Exact match" in score_dict["reasoning"]


def test_benchmark_runner(tmp_path):
    runner = BenchmarkRunner(schemas_dir="data/schemas", output_dir=str(tmp_path))

    samples = [
        {
            "prompt": "Run SQL query on analytics",
            "expected_tool": "execute_sql_query",
            "expected_arguments": {
                "query": "SELECT * FROM analytics;",
                "database": "analytics",
            },
        }
    ]

    def mock_predict(p: str) -> str:
        return '```json\n{"name": "execute_sql_query", "arguments": {"query": "SELECT * FROM analytics;", "database": "analytics"}}\n```'

    results = runner.run_benchmark(mock_predict, samples, model_name="test-model")
    assert "metrics" in results
    assert (tmp_path / "benchmark_summary.json").exists()
    assert (tmp_path / "benchmark_report.md").exists()
