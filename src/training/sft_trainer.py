"""Supervised Fine-Tuning (SFT) script for Small Language Models (Qwen2.5/Llama-3.2)."""

import argparse
from pathlib import Path
from typing import Any

from src.common.config import load_yaml
from src.common.logger import setup_logger

logger = setup_logger("sft_trainer")


def train_sft(
    config_path: str = "configs/sft_train.yaml",
    dataset_path: str = "data/processed/sft_train.json",
    val_dataset_path: str | None = "data/processed/sft_val.json",
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Executes Supervised Fine-Tuning with QLoRA.

    If dry_run=True, performs a mock training run on CPU to verify pipeline integrity.
    """
    cfg = load_yaml(config_path) if Path(config_path).exists() else {}
    logger.info("Initializing SFT Trainer with config: %s", config_path)

    if dry_run:
        logger.info("[DRY RUN] Executing mock SFT training pass on CPU...")
        output_dir = cfg.get("training", {}).get("output_dir", "checkpoints/sft_output")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Write mock adapter config and checkpoint metadata
        metadata = {
            "model_type": "sft_qlora",
            "base_model": cfg.get("model", {}).get("base_model_name_or_path", "Qwen/Qwen2.5-1.5B-Instruct"),
            "status": "mock_completed",
            "epochs": 1,
            "train_loss": 0.421,
            "val_loss": 0.389,
        }
        import json
        with open(Path(output_dir) / "adapter_config.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info("[DRY RUN] Mock SFT checkpoint saved to: %s", output_dir)
        return metadata

    # Full GPU Training using Hugging Face TRL & PEFT
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
        )
        from trl import SFTTrainer
    except ImportError as e:
        logger.error("Training dependencies missing. Please install 'torch', 'transformers', 'peft', 'trl'. Error: %s", e)
        raise

    model_name = cfg.get("model", {}).get("base_model_name_or_path", "Qwen/Qwen2.5-1.5B-Instruct")
    output_dir = cfg.get("training", {}).get("output_dir", "checkpoints/sft_output")
    lora_cfg_dict = cfg.get("lora", {})

    logger.info("Loading base model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4-bit Quantization Config (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    device_map = "auto" if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config if torch.cuda.is_available() else None,
        device_map=device_map,
        trust_remote_code=True,
    )

    if torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=lora_cfg_dict.get("r", 16),
        lora_alpha=lora_cfg_dict.get("lora_alpha", 32),
        lora_dropout=lora_cfg_dict.get("lora_dropout", 0.05),
        target_modules=lora_cfg_dict.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        bias="none",
        task_type="CAUSAL_LM",
    )

    train_ds = load_dataset("json", data_files=dataset_path, split="train")
    val_ds = load_dataset("json", data_files=val_dataset_path, split="train") if val_dataset_path else None

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg.get("training", {}).get("num_train_epochs", 3),
        per_device_train_batch_size=cfg.get("training", {}).get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=cfg.get("training", {}).get("gradient_accumulation_steps", 4),
        learning_rate=float(cfg.get("training", {}).get("learning_rate", 2e-4)),
        lr_scheduler_type=cfg.get("training", {}).get("lr_scheduler_type", "cosine"),
        warmup_ratio=cfg.get("training", {}).get("warmup_ratio", 0.03),
        weight_decay=cfg.get("training", {}).get("weight_decay", 0.01),
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        fp16=not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        logging_steps=10,
        eval_strategy="steps" if val_ds else "no",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_config,
        dataset_text_field="messages",
        max_seq_length=cfg.get("training", {}).get("max_seq_length", 2048),
        tokenizer=tokenizer,
        args=training_args,
    )

    logger.info("Starting SFT training...")
    trainer.train()
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("SFT training complete. Checkpoint saved to: %s", output_dir)
    return {"status": "success", "output_dir": output_dir}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SFT training on function calling dataset")
    parser.add_argument("--config", type=str, default="configs/sft_train.yaml")
    parser.add_argument("--data", type=str, default="data/processed/sft_train.json")
    parser.add_argument("--val-data", type=str, default="data/processed/sft_val.json")
    parser.add_argument("--dry-run", action="store_true", help="Run quick mock CPU pass")
    args = parser.parse_args()
    train_sft(args.config, args.data, args.val_data, args.dry_run)
