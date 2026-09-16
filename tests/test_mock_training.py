"""Tests for training pipelines in mock/dry-run mode."""

from pathlib import Path

from src.training.dpo_trainer import train_dpo
from src.training.quantize import merge_and_export
from src.training.sft_trainer import train_sft


def test_sft_mock_training(tmp_path):
    sft_cfg = tmp_path / "sft_cfg.yaml"
    import yaml

    cfg_dict = {
        "model": {"base_model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct"},
        "training": {"output_dir": str(tmp_path / "sft_ckpt")},
    }
    with open(sft_cfg, "w") as f:
        yaml.dump(cfg_dict, f)

    res = train_sft(config_path=str(sft_cfg), dry_run=True)
    assert res["status"] == "mock_completed"
    assert (tmp_path / "sft_ckpt" / "adapter_config.json").exists()


def test_dpo_mock_training(tmp_path):
    dpo_cfg = tmp_path / "dpo_cfg.yaml"
    import yaml

    cfg_dict = {
        "model": {"base_model_name": "Qwen/Qwen2.5-1.5B-Instruct"},
        "dpo": {"beta": 0.1},
        "training": {"output_dir": str(tmp_path / "dpo_ckpt")},
    }
    with open(dpo_cfg, "w") as f:
        yaml.dump(cfg_dict, f)

    res = train_dpo(config_path=str(dpo_cfg), dry_run=True)
    assert res["status"] == "mock_completed"
    assert res["reward_margin"] > 0
    assert (tmp_path / "dpo_ckpt" / "dpo_metadata.json").exists()


def test_merge_and_export(tmp_path):
    out_dir = tmp_path / "merged_model"
    res_path = merge_and_export(
        base_model_name="Qwen/Qwen2.5-1.5B-Instruct",
        adapter_path="non_existent_mock_path",
        output_path=str(out_dir),
    )
    assert Path(res_path).exists()
    assert (out_dir / "merge_manifest.json").exists()
