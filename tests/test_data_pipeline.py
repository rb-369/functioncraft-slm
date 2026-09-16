"""Tests for the data pipeline: validation, perturbation, and dataset generation."""

from pathlib import Path

import pytest

from src.data.generator import SyntheticDataEngine
from src.data.perturb import SchemaPerturbator
from src.data.validator import ToolCallValidator


@pytest.fixture
def sample_schema():
    return {
        "name": "test_refund",
        "description": "Test refund tool",
        "parameters": {
            "type": "object",
            "properties": {
                "transaction_id": {"type": "string"},
                "amount": {"type": "number", "minimum": 1.0},
            },
            "required": ["transaction_id", "amount"],
            "additionalProperties": False,
        },
    }


def test_validator_valid_json(sample_schema):
    validator = ToolCallValidator({"test_refund": sample_schema})
    raw_output = '```json\n{"name": "test_refund", "arguments": {"transaction_id": "txn_123", "amount": 25.0}}\n```'

    res = validator.parse_and_validate(raw_output)
    assert res["is_valid_json"] is True
    assert res["is_valid_schema"] is True
    assert res["tool_name"] == "test_refund"
    assert res["arguments"]["amount"] == 25.0


def test_validator_invalid_schema(sample_schema):
    validator = ToolCallValidator({"test_refund": sample_schema})
    # Missing required 'amount' and has undeclared property 'hacked'
    raw_output = '{"name": "test_refund", "arguments": {"transaction_id": "txn_123", "hacked": true}}'

    res = validator.parse_and_validate(raw_output)
    assert res["is_valid_json"] is True
    assert res["is_valid_schema"] is False
    assert res["error"] is not None


def test_validator_malformed_syntax(sample_schema):
    validator = ToolCallValidator({"test_refund": sample_schema})
    raw_output = '{"name": "test_refund", "arguments": { invalid_json '

    res = validator.parse_and_validate(raw_output)
    assert res["is_valid_json"] is False


def test_perturbator_strategies(sample_schema):
    perturbator = SchemaPerturbator({"test_refund": sample_schema}, seed=42)
    gold = {"name": "test_refund", "arguments": {"transaction_id": "txn_123", "amount": 50.0}}

    for strategy in [
        "hallucinated_param",
        "syntax_corruption",
        "wrong_tool_selection",
        "missing_required_param",
        "type_mismatch",
    ]:
        rejected_text, chosen_strategy = perturbator.perturb(gold, strategy=strategy)
        assert chosen_strategy == strategy
        assert isinstance(rejected_text, str)


def test_synthetic_engine_generation(tmp_path):
    schemas_dir = Path("data/schemas")
    engine = SyntheticDataEngine(schemas_dir=str(schemas_dir), seed=42)

    assert len(engine.schemas) >= 4
    sample = engine.generate_single_sample("execute_sql_query")
    assert "prompt" in sample
    assert sample["gold_tool_call"]["name"] == "execute_sql_query"

    dataset = engine.generate_dataset(samples_per_tool=5)
    assert "sft_train" in dataset
    assert "dpo_train" in dataset
    assert "eval_test" in dataset
    assert len(dataset["sft_train"]) > 0
    assert len(dataset["dpo_train"]) > 0
