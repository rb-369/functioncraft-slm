"""Inference engine supporting HuggingFace local models and fast mock test mode."""

import json
from pathlib import Path
from typing import Any

from src.common.logger import setup_logger

logger = setup_logger("inference_engine")


class BaseInferenceEngine:
    """Abstract interface for model inference."""

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        raise NotImplementedError


class MockInferenceEngine(BaseInferenceEngine):
    """
    Intelligent mock inference engine for fast local testing, development, and CI.
    Recognizes queries and returns valid tool call JSON responses.
    """

    def __init__(self, schemas: dict[str, dict[str, Any]]):
        self.schemas = schemas

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        prompt_lower = prompt.lower()

        if any(w in prompt_lower for w in ["sql", "query", "select", "customer", "inventory", "transactions", "warehouse"]):
            tool_call = {
                "name": "execute_sql_query",
                "arguments": {
                    "query": "SELECT customer_id, name, lifetime_value FROM customers WHERE signup_month = 'March' ORDER BY lifetime_value DESC LIMIT 10;",
                    "database": "analytics",
                    "max_rows": 50,
                    "timeout_seconds": 30,
                },
            }
        elif any(w in prompt_lower for w in ["ticket", "issue", "support", "dashboard", "403", "outage", "bug"]):
            tool_call = {
                "name": "create_support_ticket",
                "arguments": {
                    "customer_id": "CUST-48291",
                    "subject": "System Access Incident - Dashboard Authentication Error",
                    "category": "technical_outage" if "outage" in prompt_lower else "account_access",
                    "priority": "critical" if "emergency" in prompt_lower else "high",
                    "tags": ["automated_triage", "production"],
                },
            }
        elif any(w in prompt_lower for w in ["refund", "charge", "payment", "fraud", "dollar", "txn_"]):
            tool_call = {
                "name": "process_payment_refund",
                "arguments": {
                    "transaction_id": "txn_984129841209",
                    "amount": 149.99,
                    "currency": "USD",
                    "reason": "duplicate_charge" if "twice" in prompt_lower or "duplicate" in prompt_lower else "customer_request",
                    "notify_customer": True,
                },
            }
        elif any(w in prompt_lower for w in ["schedule", "meeting", "calendar", "event", "sync", "call"]):
            tool_call = {
                "name": "schedule_calendar_event",
                "arguments": {
                    "title": "Strategy & Technical Architecture Sync",
                    "start_time": "2026-10-15T14:00:00Z",
                    "duration_minutes": 30,
                    "attendees": ["lead.engineer@enterprise.io"],
                    "location": "Virtual Google Meet",
                },
            }
        else:
            # Default fallback tool
            tool_call = {
                "name": "execute_sql_query",
                "arguments": {
                    "query": "SELECT * FROM analytics LIMIT 10;",
                    "database": "analytics",
                },
            }

        return f"```json\n{json.dumps(tool_call, indent=2)}\n```"


class HuggingFaceInferenceEngine(BaseInferenceEngine):
    """Loads and generates with HuggingFace pipeline / CausalLM model."""

    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        self.device = device
        logger.info("Initializing HuggingFace engine from path: %s", model_path)

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=device,
                trust_remote_code=True,
            )
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
            )
        except Exception as e:
            logger.error("Failed to load HuggingFace model: %s", e)
            raise

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        if hasattr(self.tokenizer, "apply_chat_template"):
            formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            formatted_prompt = f"{system_prompt or ''}\n\nUser: {prompt}\n\nAssistant:"

        outputs = self.generator(
            formatted_prompt,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=False,
            return_full_text=False,
        )
        return outputs[0]["generated_text"]


def create_engine(
    engine_type: str = "mock",
    model_path: str | None = None,
    schemas: dict[str, dict[str, Any]] | None = None,
) -> BaseInferenceEngine:
    """Factory function for creating an inference engine."""
    schemas = schemas or {}
    if engine_type == "hf" and model_path and Path(model_path).exists():
        return HuggingFaceInferenceEngine(model_path)
    return MockInferenceEngine(schemas)
