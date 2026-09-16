"""Synthetic distillation data generator for SFT and DPO training."""

import json
import random
from pathlib import Path
from typing import Any

from src.data.perturb import SchemaPerturbator
from src.data.validator import ToolCallValidator


class SyntheticDataEngine:
    """Generates realistic enterprise user queries and corresponding tool calls."""

    SQL_TEMPLATES = [
        (
            "Show me the top {n} customers who signed up in {month} sorted by lifetime value.",
            "SELECT customer_id, name, lifetime_value FROM customers WHERE signup_month = '{month}' ORDER BY lifetime_value DESC LIMIT {n};",
            "customers",
        ),
        (
            "Find all pending transactions exceeding ${amount} in the last 24 hours.",
            "SELECT transaction_id, customer_id, amount, status FROM transactions WHERE status = 'pending' AND amount > {amount} AND created_at >= NOW() - INTERVAL '24 HOURS';",
            "transactions",
        ),
        (
            "What is the average inventory count for warehouse '{warehouse}' across all SKUs?",
            "SELECT warehouse_id, AVG(stock_count) as avg_stock FROM inventory WHERE warehouse_id = '{warehouse}' GROUP BY warehouse_id;",
            "inventory",
        ),
        (
            "Get daily active user count for our mobile app over the past {days} days.",
            "SELECT event_date, COUNT(DISTINCT user_id) as dau FROM analytics WHERE event_date >= CURRENT_DATE - INTERVAL '{days} DAYS' GROUP BY event_date ORDER BY event_date ASC;",
            "analytics",
        ),
    ]

    TICKET_TEMPLATES = [
        (
            "Customer {cust_id} reports they cannot log in and get error 403 on the dashboard.",
            "Dashboard Authentication Failure - Error 403",
            "account_access",
            "high",
            ["auth", "dashboard", "escalated"],
        ),
        (
            "Client {cust_id} says they were double charged on their annual renewal invoice.",
            "Duplicate Billing on Annual Renewal",
            "billing",
            "high",
            ["billing", "invoice", "duplicate"],
        ),
        (
            "User {cust_id} requests dark mode support in the mobile iOS application.",
            "Feature Request: Dark Mode Support on iOS",
            "feature_request",
            "low",
            ["mobile", "ui", "enhancement"],
        ),
        (
            "Emergency: Production API gateway returning 502 Bad Gateway for customer {cust_id}.",
            "Production Outage: 502 Bad Gateway on API Gateway",
            "technical_outage",
            "critical",
            ["outage", "infra", "p0"],
        ),
    ]

    REFUND_TEMPLATES = [
        (
            "Process a refund of ${amount} for transaction {txn_id} because the customer was charged twice.",
            "duplicate_charge",
            "USD",
        ),
        (
            "Refund €{amount} on txn {txn_id}. The client was dissatisfied with our premium consultation.",
            "service_dissatisfaction",
            "EUR",
        ),
        (
            "Fraud alert triggered on {txn_id}. Void and refund £{amount} immediately.",
            "fraudulent",
            "GBP",
        ),
        (
            "Customer requested cancellation for order with transaction {txn_id} totaling ${amount}.",
            "customer_request",
            "USD",
        ),
    ]

    CALENDAR_TEMPLATES = [
        (
            "Schedule a {mins}-minute sprint retrospective with {email} for {iso_date}.",
            "Sprint Retrospective",
            30,
        ),
        (
            "Book a quarterly business review with {email} on {iso_date} for {mins} minutes in Boardroom B.",
            "Quarterly Business Review (QBR)",
            60,
        ),
        (
            "Set up a quick 1:1 sync with {email} on {iso_date} at 10 AM UTC.",
            "Weekly 1:1 Sync",
            30,
        ),
        (
            "Arrange an urgent security incident debrief with {email} at {iso_date} for {mins} minutes.",
            "Security Incident Post-Mortem",
            45,
        ),
    ]

    def __init__(self, schemas_dir: str = "data/schemas", seed: int = 42):
        self.schemas_dir = Path(schemas_dir)
        self.seed = seed
        random.seed(seed)
        self.schemas: dict[str, dict[str, Any]] = {}
        self.load_schemas()
        self.validator = ToolCallValidator(self.schemas)
        self.perturbator = SchemaPerturbator(self.schemas, seed=seed)

    def load_schemas(self) -> None:
        """Loads all JSON schema files from schemas directory."""
        if not self.schemas_dir.exists():
            return
        for file in self.schemas_dir.glob("*.json"):
            with open(file, encoding="utf-8") as f:
                data = json.load(f)
                name = data.get("name")
                if name:
                    self.schemas[name] = data

    def generate_single_sample(self, tool_name: str) -> dict[str, Any]:
        """Generates a single prompt and gold tool call pair for a given tool."""
        if tool_name == "execute_sql_query":
            template, sql_pattern, db = random.choice(self.SQL_TEMPLATES)
            n = random.choice([5, 10, 25, 50])
            month = random.choice(["January", "March", "July", "November"])
            amount = random.choice([500, 1000, 2500, 5000])
            warehouse = f"WH-{random.randint(10, 99)}"
            days = random.choice([7, 14, 30, 90])

            query_str = sql_pattern.format(
                n=n, month=month, amount=amount, warehouse=warehouse, days=days
            )
            prompt = template.format(
                n=n, month=month, amount=amount, warehouse=warehouse, days=days
            )

            gold_call = {
                "name": "execute_sql_query",
                "arguments": {
                    "query": query_str,
                    "database": db,
                    "max_rows": n if "{n}" in template else 100,
                    "timeout_seconds": 30,
                },
            }

        elif tool_name == "create_support_ticket":
            template, subj, cat, prio, tags = random.choice(self.TICKET_TEMPLATES)
            cust_id = f"CUST-{random.randint(10000, 99999)}"
            prompt = template.format(cust_id=cust_id)
            gold_call = {
                "name": "create_support_ticket",
                "arguments": {
                    "customer_id": cust_id,
                    "subject": subj,
                    "category": cat,
                    "priority": prio,
                    "tags": tags,
                },
            }

        elif tool_name == "process_payment_refund":
            template, reason, curr = random.choice(self.REFUND_TEMPLATES)
            txn_id = f"txn_{random.randint(100000000000, 999999999999)}"
            amount = round(random.uniform(15.0, 950.0), 2)
            prompt = template.format(amount=amount, txn_id=txn_id)
            gold_call = {
                "name": "process_payment_refund",
                "arguments": {
                    "transaction_id": txn_id,
                    "amount": amount,
                    "currency": curr,
                    "reason": reason,
                    "notify_customer": True,
                },
            }

        elif tool_name == "schedule_calendar_event":
            template, title, default_mins = random.choice(self.CALENDAR_TEMPLATES)
            mins = random.choice([15, 30, 45, 60])
            email = f"lead.engineer.{random.randint(1, 99)}@enterprise.io"
            day = random.randint(10, 28)
            hour = random.randint(9, 17)
            iso_date = f"2026-10-{day:02d}T{hour:02d}:00:00Z"
            prompt = template.format(mins=mins, email=email, iso_date=iso_date)
            gold_call = {
                "name": "schedule_calendar_event",
                "arguments": {
                    "title": title,
                    "start_time": iso_date,
                    "duration_minutes": mins,
                    "attendees": [email],
                    "location": "Virtual Google Meet",
                },
            }
        else:
            raise ValueError(f"Unsupported tool name: {tool_name}")

        return {"prompt": prompt, "gold_tool_call": gold_call}

    def format_system_prompt(self) -> str:
        """Constructs the standard system prompt containing tool descriptions."""
        tool_specs = [
            {"name": k, "description": v.get("description", ""), "parameters": v.get("parameters", {})}
            for k, v in self.schemas.items()
        ]
        return (
            "You are an expert AI tool-calling engine. Given the available tools and user intent, "
            "respond ONLY with a valid JSON object matching the chosen tool's schema.\n\n"
            f"Available Tools:\n{json.dumps(tool_specs, indent=2)}\n\n"
            "Format response as:\n"
            '```json\n{\n  "name": "<tool_name>",\n  "arguments": { ... }\n}\n```'
        )

    def generate_dataset(
        self, samples_per_tool: int = 100
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Generates full train/val/test splits with SFT pairs, DPO preference pairs, and evaluation sets.
        """
        raw_samples = []
        for tool_name in self.schemas:
            for _ in range(samples_per_tool):
                sample = self.generate_single_sample(tool_name)
                raw_samples.append(sample)

        random.shuffle(raw_samples)

        sft_data = []
        dpo_data = []
        eval_data = []

        system_prompt = self.format_system_prompt()

        for idx, sample in enumerate(raw_samples):
            prompt = sample["prompt"]
            gold = sample["gold_tool_call"]
            chosen_text = f"```json\n{json.dumps(gold, indent=2)}\n```"

            # Create negative sample for DPO
            rejected_raw, strategy = self.perturbator.perturb(gold)
            rejected_text = f"```json\n{rejected_raw}\n```"

            # SFT Format
            sft_item = {
                "id": f"sft_{idx:05d}",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": chosen_text},
                ],
            }
            sft_data.append(sft_item)

            # DPO Format
            dpo_item = {
                "id": f"dpo_{idx:05d}",
                "system": system_prompt,
                "prompt": prompt,
                "chosen": chosen_text,
                "rejected": rejected_text,
                "perturbation_strategy": strategy,
            }
            dpo_data.append(dpo_item)

            # Eval Format
            eval_item = {
                "id": f"eval_{idx:05d}",
                "prompt": prompt,
                "expected_tool": gold["name"],
                "expected_arguments": gold["arguments"],
            }
            eval_data.append(eval_item)

        total = len(sft_data)
        train_idx = int(0.8 * total)
        val_idx = int(0.9 * total)

        return {
            "sft_train": sft_data[:train_idx],
            "sft_val": sft_data[train_idx:val_idx],
            "sft_test": sft_data[val_idx:],
            "dpo_train": dpo_data[:train_idx],
            "dpo_val": dpo_data[train_idx:val_idx],
            "eval_test": eval_data[val_idx:],
        }
