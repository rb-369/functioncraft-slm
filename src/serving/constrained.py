"""Grammar-guided constrained decoding, JSON repair, and schema sanitization."""

import re
from typing import Any

from src.data.validator import ToolCallValidator


class ConstrainedDecoder:
    """Repairs and enforces schema adherence on raw LLM tool calling outputs."""

    def __init__(self, schemas: dict[str, dict[str, Any]]):
        self.schemas = schemas
        self.validator = ToolCallValidator(schemas)

    def repair_json_string(self, raw: str) -> str | None:
        """Attempts deterministic repairs on common small model JSON syntax defects."""
        s = raw.strip()
        # Strip code blocks
        s = re.sub(r"^```(?:json)?", "", s, flags=re.MULTILINE)
        s = re.sub(r"```$", "", s, flags=re.MULTILINE).strip()

        # Fix single quotes
        if "'" in s and '"' not in s:
            s = s.replace("'", '"')

        # Fix trailing commas before closing braces/brackets
        s = re.sub(r",\s*(\}|\])", r"\1", s)

        # Attempt balance of missing closing brackets
        open_braces = s.count("{") - s.count("}")
        if open_braces > 0:
            s = s + ("}" * open_braces)

        return s

    def decode_and_enforce(
        self,
        raw_output: str,
        preferred_tool: str | None = None,
    ) -> tuple[bool, dict[str, Any], str | None]:
        """
        Extracts, repairs, and strictly validates against schema.

        Returns:
            Tuple: (is_valid, parsed_tool_call_dict, error_message)
        """
        # First attempt: direct validation
        val_res = self.validator.parse_and_validate(raw_output, expected_tool=preferred_tool)
        if val_res["is_valid_schema"]:
            return True, {"name": val_res["tool_name"], "arguments": val_res["arguments"]}, None

        # Second attempt: heuristic repair
        repaired = self.repair_json_string(raw_output)
        if repaired:
            val_repaired = self.validator.parse_and_validate(repaired, expected_tool=preferred_tool)
            if val_repaired["is_valid_schema"]:
                return True, {"name": val_repaired["tool_name"], "arguments": val_repaired["arguments"]}, None

        # Fallback if valid JSON but missing optional parameters or failed strict check
        if val_res["is_valid_json"] and val_res["tool_name"]:
            return False, {"name": val_res["tool_name"], "arguments": val_res["arguments"]}, val_res["error"]

        return False, {}, val_res["error"] or "Failed to decode and enforce valid tool call"
