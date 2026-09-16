"""Evaluation metrics for tool calling, schema adherence, and argument extraction."""

from typing import Any

from src.data.validator import ToolCallValidator


class BenchmarkMetrics:
    """Calculates granular evaluation metrics for structured function-calling models."""

    def __init__(self, schemas: dict[str, dict[str, Any]] | None = None):
        self.validator = ToolCallValidator(schemas)

    def evaluate_sample(
        self,
        raw_output: str,
        expected_tool: str,
        expected_args: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Evaluates a single model output against gold expected tool and arguments.
        """
        val_result = self.validator.parse_and_validate(raw_output, expected_tool=expected_tool)
        is_json = val_result["is_valid_json"]
        is_schema = val_result["is_valid_schema"]
        pred_tool = val_result["tool_name"]
        pred_args = val_result["arguments"] or {}

        tool_correct = (pred_tool == expected_tool) if pred_tool else False

        # Argument evaluation
        arg_em = False
        key_precision = 0.0
        key_recall = 0.0
        arg_f1 = 0.0

        if tool_correct and isinstance(pred_args, dict):
            # Exact Match
            arg_em = (pred_args == expected_args)

            # Key-level overlap
            gold_keys = set(expected_args.keys())
            pred_keys = set(pred_args.keys())

            if gold_keys and pred_keys:
                common_keys = gold_keys.intersection(pred_keys)
                key_precision = len(common_keys) / len(pred_keys)
                key_recall = len(common_keys) / len(gold_keys)
                if key_precision + key_recall > 0:
                    arg_f1 = 2 * (key_precision * key_recall) / (key_precision + key_recall)
            elif not gold_keys and not pred_keys:
                key_precision = 1.0
                key_recall = 1.0
                arg_f1 = 1.0

        return {
            "valid_json": is_json,
            "valid_schema": is_schema,
            "tool_correct": tool_correct,
            "arg_exact_match": arg_em,
            "arg_key_f1": arg_f1,
            "predicted_tool": pred_tool,
            "validation_error": val_result["error"],
        }

    def aggregate(self, sample_results: list[dict[str, Any]]) -> dict[str, float]:
        """Aggregates per-sample metric dicts into summary benchmark percentages."""
        if not sample_results:
            return {}

        n = len(sample_results)
        valid_json_count = sum(1 for r in sample_results if r["valid_json"])
        valid_schema_count = sum(1 for r in sample_results if r["valid_schema"])
        tool_correct_count = sum(1 for r in sample_results if r["tool_correct"])
        arg_em_count = sum(1 for r in sample_results if r["arg_exact_match"])
        avg_arg_f1 = sum(r["arg_key_f1"] for r in sample_results) / n

        schema_adherence_rate = valid_schema_count / n
        tool_accuracy = tool_correct_count / n
        arg_exact_match_rate = arg_em_count / n

        # Composite SLM quality index (weighted score)
        composite_score = (
            0.35 * schema_adherence_rate
            + 0.35 * tool_accuracy
            + 0.20 * arg_exact_match_rate
            + 0.10 * avg_arg_f1
        )

        return {
            "total_samples": n,
            "json_validity_rate": round(valid_json_count / n, 4),
            "schema_adherence_rate": round(schema_adherence_rate, 4),
            "tool_accuracy": round(tool_accuracy, 4),
            "argument_exact_match_rate": round(arg_exact_match_rate, 4),
            "argument_key_f1": round(avg_arg_f1, 4),
            "composite_slm_score": round(composite_score, 4),
        }
