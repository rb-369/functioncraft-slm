"""LLM-as-a-Judge semantic quality evaluator."""

import json
import os
from typing import Any

from src.common.logger import setup_logger

logger = setup_logger("eval_judge")


class LLMJudge:
    """Evaluates semantic accuracy and context adherence of tool calls using a judge model."""

    JUDGE_PROMPT_TEMPLATE = """You are an impartial evaluator assessing the accuracy of an AI function calling model.

User Query: {query}
Expected Tool Call: {expected}
Model Generated Tool Call: {generated}

Score the generated tool call on a scale of 1 to 5:
5 - Perfect: Correct tool selected, all parameters semantically faithful and well-typed.
4 - Good: Correct tool and necessary parameters present, minor harmless discrepancies.
3 - Borderline: Right tool but missing one optional parameter or slightly suboptimal value.
2 - Poor: Incorrect tool selection or severe parameter hallucination.
1 - Failure: Completely unparsable or nonsensical output.

Respond strictly in JSON format:
{{
  "score": <integer from 1 to 5>,
  "reasoning": "<concise explanation>"
}}
"""

    def __init__(self, model_name: str = "gpt-4o-mini", api_key: str | None = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

    def evaluate_pair(
        self, query: str, expected: dict[str, Any], generated_raw: str
    ) -> dict[str, Any]:
        """Judges a generated tool call against expected ground truth."""
        # If API key is available, call OpenAI API
        if self.api_key:
            try:
                import requests
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": self.model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": self.JUDGE_PROMPT_TEMPLATE.format(
                                query=query,
                                expected=json.dumps(expected),
                                generated=generated_raw,
                            ),
                        }
                    ],
                    "temperature": 0.0,
                    "response_format": {"type": "json_object"},
                }
                res = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=15)
                res.raise_for_status()
                content = res.json()["choices"][0]["message"]["content"]
                return json.loads(content)
            except Exception as e:
                logger.warning("LLM Judge API request failed: %s. Falling back to heuristic judge.", e)

        # Deterministic heuristic judge fallback (offline / local development)
        return self._heuristic_judge(expected, generated_raw)

    def _heuristic_judge(self, expected: dict[str, Any], generated_raw: str) -> dict[str, Any]:
        """Rule-based heuristic evaluator when API key is not configured."""
        try:
            from src.data.validator import ToolCallValidator
            validator = ToolCallValidator()
            json_str, err = validator.extract_json_block(generated_raw)
            if not json_str:
                return {"score": 1, "reasoning": f"Failed to parse JSON: {err}"}

            parsed = json.loads(json_str)
            tool = parsed.get("name") or parsed.get("tool")
            args = parsed.get("arguments") or parsed.get("parameters") or {}

            expected_tool = expected.get("name")
            expected_args = expected.get("arguments", {})

            if tool != expected_tool:
                return {"score": 2, "reasoning": f"Tool mismatch: predicted '{tool}', expected '{expected_tool}'"}

            if args == expected_args:
                return {"score": 5, "reasoning": "Exact match on tool selection and all arguments."}

            # Check key overlap
            exp_keys = set(expected_args.keys())
            pred_keys = set(args.keys())
            if exp_keys.issubset(pred_keys):
                extra = pred_keys - exp_keys
                return {"score": 4, "reasoning": f"All expected arguments matched with harmless extra keys: {list(extra)}"}
            elif pred_keys.intersection(exp_keys):
                return {"score": 3, "reasoning": "Partial argument match."}
            else:
                return {"score": 2, "reasoning": "No overlapping arguments found."}

        except Exception as e:
            return {"score": 1, "reasoning": f"Heuristic evaluation exception: {e}"}
