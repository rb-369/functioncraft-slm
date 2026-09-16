"""Direct Preference Optimization (DPO) alignment script."""

import argparse
from pathlib import Path
from typing import Any

from src.common.config import load_yaml
from src.common.logger import setup_logger

logger = setup_logger("dpo_trainer")


def train_dpo(
    config_path: str = "configs/dpo_train.yaml",
    dataset_path: str = "data/processed/dpo_train.json",
    val_dataset_path: str | None = "data/processed/dpo_val.json",
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Executes DPO alignment to heavily penalize malformed JSON, hallucinated parameters,
    and incorrect tool routing.
    """
    cfg = load_yaml(config_path) if Path(config_path).exists() else {}
    logger.info("Initializing DPO Trainer with config: %s", config_path)

    if dry_run:
        logger.info("[DRY RUN] Executing mock DPO alignment pass on CPU...")
        output_dir = cfg.get("training", {}).get("output_dir", "checkpoints/dpo_output")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        metadata = {
            "model_type": "dpo_aligned_qlora",
            "base_checkpoint": cfg.get("model", {}).get("sft_model_checkpoint", "checkpoints/sft_output"),
            "status": "mock_completed",
            "epochs": 1,
            "beta": cfg.get("dpo", {}).get("beta", 0.1),
            "chosen_reward_mean": 1.84,
            "rejected_reward_mean": -2.31,
            "reward_margin": 4.15,
        }
        import json
        with open(Path(output_dir) / "dpo_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info("[DRY RUN] Mock DPO checkpoint saved to: %s", output_dir)
        return metadata

    try:
        import torch
        from datasets import load_dataset
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from trl import DPOTrainer
    except ImportError as e:
        logger.error("Training dependencies missing. Error: %s", e)
        raise

    base_model_name = cfg.get("model", {}).get("base_model_name", "Qwen/Qwen2.5-1.5B-Instruct")
    sft_checkpoint = cfg.get("model", {}).get("sft_model_checkpoint", "checkpoints/sft_output")
    output_dir = cfg.get("training", {}).get("output_dir", "checkpoints/dpo_output")

    logger.info("Loading tokenizer and base model: %s", base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model to be trained
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    if Path(sft_checkpoint).exists():
        logger.info("Loading SFT adapter weights from: %s", sft_checkpoint)
        model = PeftModel.from_pretrained(model, sft_checkpoint, is_trainable=True)

    # Reference model (frozen) for DPO KL penalty
    ref_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if Path(sft_checkpoint).exists():
        ref_model = PeftModel.from_pretrained(ref_model, sft_checkpoint)
    ref_model.eval()

    train_ds = load_dataset("json", data_files=dataset_path, split="train")
    val_ds = load_dataset("json", data_files=val_dataset_path, split="train") if val_dataset_path else None

    dpo_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg.get("training", {}).get("num_train_epochs", 2),
        per_device_train_batch_size=cfg.get("training", {}).get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=cfg.get("training", {}).get("gradient_accumulation_steps", 8),
        learning_rate=float(cfg.get("training", {}).get("learning_rate", 5e-6)),
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        logging_steps=10,
        eval_strategy="steps" if val_ds else "no",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        beta=cfg.get("dpo", {}).get("beta", 0.1),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        args=dpo_args,
        max_length=cfg.get("dpo", {}).get("max_length", 2048),
        max_prompt_length=cfg.get("dpo", {}).get("max_prompt_length", 1024),
    )

    logger.info("Starting DPO preference alignment...")
    trainer.train()
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("DPO alignment completed. Checkpoint saved to: %s", output_dir)
    return {"status": "success", "output_dir": output_dir}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DPO alignment for function calling")
    parser.add_argument("--config", type=str, default="configs/dpo_train.yaml")
    parser.add_argument("--data", type=str, default="data/processed/dpo_train.json")
    parser.add_argument("--val-data", type=str, default="data/processed/dpo_val.json")
    parser.add_argument("--dry-run", action="store_true", help="Run quick mock CPU pass")
    args = parser.parse_args()
    train_dpo(args.config, args.data, args.val_data, args.dry_run)
