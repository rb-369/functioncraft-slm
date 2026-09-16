"""Perturbation engine for generating hard negative rejected samples for DPO alignment."""

import copy
import json
import random
from typing import Any


class SchemaPerturbator:
    """Generates realistic negative examples from valid tool calls for DPO preference pairs."""

    HALLUCINATED_PARAMS = [
        ("dry_run", True),
        ("authorization_token", "Bearer test_tok_99182"),
        ("bypass_cache", True),
        ("priority_override", "urgent"),
        ("metadata_tags", ["auto_generated", "v2"]),
        ("user_locale", "en-US"),
        ("async_dispatch", False),
    ]

    def __init__(self, schemas: dict[str, dict[str, Any]], seed: int = 42):
        self.schemas = schemas
        self.tool_names = list(schemas.keys())
        random.seed(seed)

    def perturb(
        self, gold_tool_call: dict[str, Any], strategy: str = "random"
    ) -> tuple[str, str]:
        """
        Perturbs a valid gold tool call dict into a rejected negative sample.

        Returns:
            Tuple of (rejected_raw_text, strategy_used)
        """
        tool_name = gold_tool_call.get("name")
        args = copy.deepcopy(gold_tool_call.get("arguments", {}))

        valid_strategies = [
            "hallucinated_param",
            "syntax_corruption",
            "wrong_tool_selection",
            "missing_required_param",
            "type_mismatch",
        ]

        chosen_strategy = random.choice(valid_strategies) if strategy == "random" else strategy

        if chosen_strategy == "hallucinated_param":
            fake_key, fake_val = random.choice(self.HALLUCINATED_PARAMS)
            args[fake_key] = fake_val
            rejected_dict = {"name": tool_name, "arguments": args}
            return json.dumps(rejected_dict, indent=2), "hallucinated_param"

        elif chosen_strategy == "syntax_corruption":
            # Valid dict dumped to json, then intentionally corrupted
            clean_json = json.dumps({"name": tool_name, "arguments": args}, indent=2)
            corruption_type = random.choice(["trailing_comma", "unclosed_brace", "single_quotes"])
            if corruption_type == "trailing_comma":
                lines = clean_json.split("\n")
                if len(lines) > 3:
                    lines[-2] = lines[-2] + ","
                return "\n".join(lines), "syntax_corruption"
            elif corruption_type == "unclosed_brace":
                return clean_json.rstrip("}\n "), "syntax_corruption"
            else:
                return clean_json.replace('"', "'"), "syntax_corruption"

        elif chosen_strategy == "wrong_tool_selection":
            other_tools = [t for t in self.tool_names if t != tool_name]
            wrong_tool = random.choice(other_tools) if other_tools else "generic_search"
            rejected_dict = {"name": wrong_tool, "arguments": args}
            return json.dumps(rejected_dict, indent=2), "wrong_tool_selection"

        elif chosen_strategy == "missing_required_param":
            schema = self.schemas.get(tool_name, {})
            param_spec = schema.get("parameters", {})
            required_keys = param_spec.get("required", [])
            available_to_drop = [k for k in required_keys if k in args]

            if available_to_drop:
                key_to_drop = random.choice(available_to_drop)
                del args[key_to_drop]
            elif args:
                # Drop any existing key
                del args[random.choice(list(args.keys()))]

            rejected_dict = {"name": tool_name, "arguments": args}
            return json.dumps(rejected_dict, indent=2), "missing_required_param"

        elif chosen_strategy == "type_mismatch":
            # Convert an int/float to string or boolean to string
            modified = False
            for k, v in args.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    args[k] = f"INVALID_NUM_{v}"
                    modified = True
                    break
                elif isinstance(v, bool):
                    args[k] = "maybe"
                    modified = True
                    break

            if not modified and args:
                k = random.choice(list(args.keys()))
                args[k] = 12345  # Injected integer for string field

            rejected_dict = {"name": tool_name, "arguments": args}
            return json.dumps(rejected_dict, indent=2), "type_mismatch"

        # Fallback: simple malformed JSON
        return '{"name": "' + str(tool_name) + '", "arguments": { ... }', "syntax_corruption"
