"""Batch evaluation tool schema builder.

make_batch_tool_schema() wraps any single-item eval tool schema into a batch
version that accepts an array of evaluations, each tagged with a listing_id
for result correlation.

Usage in an app:
    from screening_core.batch import make_batch_tool_schema
    BATCH_EVAL_TOOL_SCHEMA = make_batch_tool_schema(EVAL_TOOL_SCHEMA)
"""
from __future__ import annotations

BATCH_RAW_CHARS_PER_ITEM = 1500   # raw_text truncation per item in batch (vs 3000 for single)
BATCH_MAX_OUTPUT_TOKENS = 8000    # safe upper bound for batch response


def make_batch_tool_schema(single_tool: dict, name: str = "submit_batch_evaluation") -> dict:
    """Wrap a single-item eval tool schema into a batch version with listing_id correlation.

    The returned schema has a single top-level 'evaluations' array. Each element
    mirrors the single_tool schema plus a required 'listing_id' field.
    """
    per_item_props = dict(single_tool["input_schema"]["properties"])
    per_item_props["listing_id"] = {
        "type": "string",
        "description": (
            "The listing_id of the item being evaluated "
            "(return exactly as given in the input)."
        ),
    }
    per_item_required = ["listing_id"] + list(single_tool["input_schema"]["required"])

    return {
        "name": name,
        "description": (
            "Submit structured evaluation scores and recommended actions for multiple "
            "items in a single call."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "evaluations": {
                    "type": "array",
                    "description": (
                        "Evaluations for each item in the same order as the input. "
                        "Each element's listing_id must exactly match the value provided."
                    ),
                    "items": {
                        "type": "object",
                        "properties": per_item_props,
                        "required": per_item_required,
                    },
                },
            },
            "required": ["evaluations"],
        },
    }
