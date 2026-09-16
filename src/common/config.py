"""Configuration loaders and Pydantic validation schemas."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class BaseModelConfig(BaseModel):
    name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    alt_name: str | None = "meta-llama/Llama-3.2-1B-Instruct"
    context_length: int = 2048
    torch_dtype: str = "bfloat16"


class PathsConfig(BaseModel):
    schemas_dir: str = "data/schemas"
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    checkpoints_dir: str = "checkpoints"
    output_dir: str = "outputs"
    eval_results_dir: str = "outputs/evaluation"


class ServingConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    timeout_keep_alive: int = 30
    max_model_len: int = 2048


class AppConfig(BaseModel):
    project_name: str = "functioncraft-slm"
    version: str = "0.1.0"
    seed: int = 42
    base_model: BaseModelConfig = Field(default_factory=BaseModelConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)


class LoraConfigModel(BaseModel):
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    task_type: str = "CAUSAL_LM"


class SftTrainingConfig(BaseModel):
    output_dir: str = "checkpoints/sft_output"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2.0e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_seq_length: int = 2048
    bf16: bool = True
    fp16: bool = False
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 100


class DpoTrainingConfig(BaseModel):
    output_dir: str = "checkpoints/dpo_output"
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5.0e-6
    beta: float = 0.1
    max_length: int = 2048
    max_prompt_length: int = 1024


def load_yaml(file_path: str | Path) -> dict[str, Any]:
    """Loads a YAML file into a dictionary."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_app_config(config_path: str = "configs/config.yaml") -> AppConfig:
    """Loads and validates the main application configuration."""
    raw = load_yaml(config_path)
    return AppConfig(**raw)
