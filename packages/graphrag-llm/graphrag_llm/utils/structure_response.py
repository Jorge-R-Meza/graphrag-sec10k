# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Structure response as pydantic base model."""

import json
import logging
import re
from typing import Any, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel, covariant=True)

# Matches a ```json ... ``` or ``` ... ``` fenced block
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)
# Matches the outermost JSON object {...} anywhere in the string
_EMBEDDED_JSON_RE = re.compile(r"(\{.*\})", re.DOTALL)
# Fixes keys where the LLM uses mixed quotes: "key': → "key":
_MIXED_QUOTE_KEY_RE = re.compile(r'"([^"\n]+)\'\s*:')


def _fix_mixed_quote_keys(json_str: str) -> str:
    """Fix LLM output where JSON keys use mixed quotes: "key': → "key":"""
    fixed, count = _MIXED_QUOTE_KEY_RE.subn(lambda m: f'"{m.group(1)}":', json_str)
    if count:
        logger.debug("Fixed %d mixed-quote key(s) in LLM JSON response.", count)
    return fixed


def _extract_json(response: str) -> str:
    """Return the JSON string to parse, trying fenced block then embedded object."""
    match = _FENCED_JSON_RE.search(response)
    if match:
        return match.group(1).strip()
    match = _EMBEDDED_JSON_RE.search(response)
    if match:
        return match.group(1).strip()
    return response.strip()


def structure_completion_response(response: str, model: type[T]) -> T:
    """Structure completion response as pydantic base model.

    Args
    ----
        response: str
            The completion response as a JSON string.
        model: type[T]
            The pydantic base model type to structure the response into.

    Returns
    -------
        The structured response as a pydantic base model.
    """
    json_str = _fix_mixed_quote_keys(_extract_json(response))
    try:
        parsed_dict: dict[str, Any] = json.loads(json_str)
    except json.JSONDecodeError as exc:
        logger.warning(
            "Failed to parse LLM response as JSON for %s. raw=%r error=%s",
            model.__name__, response[:300], exc,
        )
        raise
    return model.model_validate(parsed_dict)
