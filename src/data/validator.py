"""Validation utilities for tool call syntax and JSON schema compliance."""

import json
import re
from typing import Any

import jsonschema
from jsonschema import ValidationError


class ToolCallValidator:
    """Validates tool call strings and dictionaries against OpenAPI/JSON schemas."""

    def __init__(self, schemas: dict[str, dict[str, Any]] | None = None):
        """
        Args:
            schemas: Mapping from tool name to its JSON schema definition.
        """
        self.schemas = schemas or {}

    def register_schema(self, tool_name: str, schema: dict[str, Any]) -> None:
        """Registers a new schema for validation."""
        self.schemas[tool_name] = schema

    @staticmethod
    def extract_json_block(text: str) -> tuple[str | None, str | None]:
        """
        Extracts tool name and JSON string from common LLM outputs:
        - Markdown blocks ```json ... ```
        - Direct JSON objects {"name": "...", "arguments": {...}}
        - XML tags <tool_call>...</tool_call>
        """
        text = text.strip()

        # Check for XML tags
        xml_match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
        if xml_match:
            text = xml_match.group(1).strip()

        # Check for markdown code fence
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()

        # Fallback: find outermost curly braces
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_candidate = text[first_brace : last_brace + 1]
            return json_candidate, None

        return None, "No JSON block structure identified in model output"

    def parse_and_validate(
        self, raw_output: str, expected_tool: str | None = None
    ) -> dict[str, Any]:
        """
        Parses raw text and validates syntax and schema adherence.

        Returns:
            Dict with keys:
            - is_valid_json (bool)
            - is_valid_schema (bool)
            - tool_name (Optional[str])
            - arguments (Optional[dict])
            - error (Optional[str])
        """
        result = {
            "is_valid_json": False,
            "is_valid_schema": False,
            "tool_name": None,
            "arguments": None,
            "error": None,
        }

        json_str, err = self.extract_json_block(raw_output)
        if not json_str:
            result["error"] = err or "Could not find JSON object"
            return result

        try:
            parsed = json.loads(json_str)
            result["is_valid_json"] = True
        except json.JSONDecodeError as exc:
            result["error"] = f"JSONDecodeError at line {exc.lineno}: {exc.msg}"
            return result

        if not isinstance(parsed, dict):
            result["error"] = f"Parsed JSON must be an object, got {type(parsed).__name__}"
            return result

        # Detect tool name and arguments fields
        tool_name = parsed.get("name") or parsed.get("tool") or parsed.get("function")
        arguments = parsed.get("arguments") or parsed.get("parameters") or parsed.get("args")

        # If parsed directly as arguments dict
        if tool_name is None and expected_tool:
            tool_name = expected_tool
            arguments = parsed

        if not tool_name:
            result["error"] = "Tool call missing 'name' or 'tool' field"
            return result

        if arguments is None or not isinstance(arguments, dict):
            # If arguments was encoded as string JSON, attempt parse
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except Exception:
                    result["error"] = "Arguments field is not a valid JSON object string"
                    return result
            else:
                result["error"] = "Tool call missing valid 'arguments' dictionary"
                return result

        result["tool_name"] = tool_name
        result["arguments"] = arguments

        # Schema validation
        schema = self.schemas.get(tool_name)
        if not schema:
            result["error"] = f"Unknown tool '{tool_name}' (not in registered schemas)"
            return result

        param_schema = schema.get("parameters", schema)
        try:
            jsonschema.validate(instance=arguments, schema=param_schema)
            result["is_valid_schema"] = True
        except ValidationError as val_err:
            result["error"] = f"Schema ValidationError: {val_err.message} (path: {list(val_err.path)})"

        return result
