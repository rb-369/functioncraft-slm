"""Dataset serialization, loading, and HuggingFace formatting."""

import json
from pathlib import Path
from typing import Any


def save_dataset_splits(
    splits: dict[str, list[dict[str, Any]]], output_dir: str = "data/processed"
) -> dict[str, str]:
    """Saves generated dataset splits to json files in the output directory."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    file_map = {}

    for split_name, records in splits.items():
        file_path = out_path / f"{split_name}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        file_map[split_name] = str(file_path)

    return file_map


def load_dataset_split(file_path: str) -> list[dict[str, Any]]:
    """Loads a JSON dataset split from disk."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset split not found: {file_path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def format_for_hf_dpo(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    """
    Formats standard DPO records into HuggingFace trl.DPOTrainer input dicts:
    {"prompt": ..., "chosen": ..., "rejected": ...}
    """
    formatted = []
    for r in records:
        full_prompt = f"{r['system']}\n\nUser: {r['prompt']}\n\nAssistant:"
        formatted.append({
            "prompt": full_prompt,
            "chosen": r["chosen"],
            "rejected": r["rejected"],
        })
    return formatted
