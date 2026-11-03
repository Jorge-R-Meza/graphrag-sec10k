# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Graph extraction helpers that return tabular data."""

import logging
import re
import traceback
from typing import TYPE_CHECKING, Any

import pandas as pd
from graphrag_llm.circuit_breaker import PipelineHalted
from graphrag_llm.utils import (
    CompletionMessagesBuilder,
)

from graphrag.index.typing.error_handler import ErrorHandlerFn
from graphrag.index.utils.string import clean_str

if TYPE_CHECKING:
    from graphrag_llm.completion import LLMCompletion
    from graphrag_llm.types import LLMCompletionResponse

INPUT_TEXT_KEY = "input_text"
RECORD_DELIMITER_KEY = "record_delimiter"
COMPLETION_DELIMITER_KEY = "completion_delimiter"
ENTITY_TYPES_KEY = "entity_types"
EXTRACTED_ENTITIES_KEY = "extracted_entities"
ENTITY_LIST_KEY = "entity_list"
TUPLE_DELIMITER = "<|>"
RECORD_DELIMITER = "##"
COMPLETION_DELIMITER = "<|COMPLETE|>"
RELATIONSHIP_CONTINUE_PROMPT = (
    "MANY relationships were missed, or your previous answer was cut off before "
    "you finished. Continue extracting relationships from the SAME passage and "
    "the SAME known-entity list. Do NOT repeat any relationship you already "
    'emitted. Use the exact ("relationship"<|>SOURCE<|>TARGET<|>DESCRIPTION<|>'
    "STRENGTH) format with ## as the record delimiter, and output "
    "<|COMPLETE|> only when there are genuinely no more relationships to add.\n"
)
_COMPLETE_REMINDER = (
    "\n\nIMPORTANT: Your very last output must be exactly <|COMPLETE|> on its own line after "
    "all entity and relationship records."
)

_TRAILING_BRACKET_RE = re.compile(r"\s*\[[^\]]*\]\s*$")

logger = logging.getLogger(__name__)


class GraphExtractor:
    """Unipartite graph extractor class definition."""

    _model: "LLMCompletion"
    _extraction_prompt: str
    _gleaning_prompt: str
    _relationship_prompt: str
    _max_gleanings: int
    _on_error: ErrorHandlerFn
    _relationship_max_tokens: int | None

    def __init__(
        self,
        model: "LLMCompletion",
        prompt: str,
        gleaning_prompt: str,
        relationship_prompt: str,
        max_gleanings: int,
        on_error: ErrorHandlerFn | None = None,
        relationship_max_tokens: int | None = None,
    ):
        """Init method definition."""
        self._model = model
        self._extraction_prompt = prompt
        self._gleaning_prompt = gleaning_prompt
        self._relationship_prompt = relationship_prompt
        self._max_gleanings = max_gleanings
        self._on_error = on_error or (lambda _e, _s, _d: None)
        self._relationship_max_tokens = relationship_max_tokens

    async def __call__(
        self, text: str, entity_types: list[str], source_id: str,
        entity_reference: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame] | None:
        """Extract entities and relationships from the supplied text.

        When *entity_reference* is provided the extractor runs in phase-2 mode:
        entity extraction is skipped and the relationship prompt is called
        directly with the supplied reference list as the entity_list context.
        """
        try:
            if entity_reference is not None:
                rel_raw = (
                    await self._extract_relationships_raw(text, entity_reference)
                    if entity_reference.strip()
                    else ""
                )
                result = rel_raw
            else:
                entity_raw = await self._extract_entities_raw(text, entity_types)
                entity_list = _parse_entity_list_with_types(entity_raw)
                rel_raw = await self._extract_relationships_raw(text, entity_list) if entity_list else ""
                result = entity_raw + rel_raw
        except PipelineHalted:
            raise
        except Exception as e:  # pragma: no cover - defensive logging
            logger.exception("error extracting graph")
            self._on_error(
                e,
                traceback.format_exc(),
                {
                    "source_id": source_id,
                    "text": text,
                },
            )
            return None

        return self._process_result(
            result,
            source_id,
            TUPLE_DELIMITER,
            RECORD_DELIMITER,
        )

    async def extract_entities(
        self, text: str, entity_types: list[str], source_id: str
    ) -> pd.DataFrame | None:
        """Extract entities only (no relationship extraction)."""
        try:
            result = await self._extract_entities_raw(text, entity_types)
        except PipelineHalted:
            raise
        except Exception as e:  # pragma: no cover
            logger.exception("error extracting entities")
            self._on_error(e, traceback.format_exc(), {"source_id": source_id, "text": text})
            return None
        entities_df, _ = self._process_result(result, source_id, TUPLE_DELIMITER, RECORD_DELIMITER, expect_relationships=False)
        return entities_df

    async def extract_relationships(
        self, text: str, entity_list: str, source_id: str
    ) -> pd.DataFrame:
        """Extract relationships given a pre-built entity list string."""
        if not entity_list:
            return _empty_relationships_df()
        try:
            result = await self._extract_relationships_raw(text, entity_list)
        except PipelineHalted:
            raise
        except Exception as e:  # pragma: no cover
            logger.exception("error extracting relationships")
            self._on_error(e, traceback.format_exc(), {"source_id": source_id, "text": text})
            return _empty_relationships_df()
        _, relationships_df = self._process_result(result, source_id, TUPLE_DELIMITER, RECORD_DELIMITER)
        return relationships_df

    async def _extract_entities_raw(self, text: str, entity_types: list[str]) -> str:
        # Call 1: Full initial extraction (entities + any relationships from main prompt)
        messages_builder = CompletionMessagesBuilder().add_user_message(
            self._extraction_prompt.format(**{
                INPUT_TEXT_KEY: text,
                ENTITY_TYPES_KEY: ",".join(entity_types),
            })
        )
        response: LLMCompletionResponse = await self._model.completion_async(
            messages=messages_builder.build(),
        )  # type: ignore
        results = response.content
        if self._check_truncation(response, "initial extraction"):
            logger.debug("Retrying initial extraction with completion reminder (source: missing <|COMPLETE|>)")
            retry_builder = CompletionMessagesBuilder().add_user_message(
                (self._extraction_prompt + _COMPLETE_REMINDER).format(**{
                    INPUT_TEXT_KEY: text,
                    ENTITY_TYPES_KEY: ",".join(entity_types),
                })
            )
            response = await self._model.completion_async(
                messages=retry_builder.build(),
            )  # type: ignore
            results = response.content
            self._check_truncation(response, "initial extraction (retry)")

        # Calls 2..N: Entity gleaning — each is a fresh conversation
        if self._max_gleanings > 0:
            for i in range(self._max_gleanings):
                entity_names = _parse_entity_names(results)
                if not entity_names:
                    break
                extracted_entities = "\n".join(f"- {name}" for name in entity_names)
                gleaning_builder = CompletionMessagesBuilder().add_user_message(
                    self._gleaning_prompt.format(**{
                        ENTITY_TYPES_KEY: ",".join(entity_types),
                        # Backward-compatible alias for custom prompts that expect
                        # relationship-style entity list variable names.
                        ENTITY_LIST_KEY: extracted_entities,
                        EXTRACTED_ENTITIES_KEY: extracted_entities,
                        INPUT_TEXT_KEY: text,
                    })
                )
                response = await self._model.completion_async(
                    messages=gleaning_builder.build(),
                )  # type: ignore
                gleaning_text = response.content
                self._check_truncation(response, f"gleaning {i + 1}")
                results += gleaning_text

        return results

    async def _extract_relationships_raw(self, text: str, entity_list: str) -> str:
        rel_builder = CompletionMessagesBuilder().add_user_message(
            self._relationship_prompt.format(**{
                ENTITY_LIST_KEY: entity_list,
                INPUT_TEXT_KEY: text,
            })
        )

        call_kwargs: dict[str, Any] = {}
        if self._relationship_max_tokens is not None:
            call_kwargs["max_tokens"] = self._relationship_max_tokens

        response = await self._model.completion_async(
            messages=rel_builder.build(),
            **call_kwargs,
        )  # type: ignore
        if self._check_truncation(response, "relationship extraction"):  # type: ignore[arg-type]
            logger.debug("Retrying relationship extraction with completion reminder (source: missing <|COMPLETE|>)")
            retry_rel_builder = CompletionMessagesBuilder().add_user_message(
                (self._relationship_prompt + _COMPLETE_REMINDER).format(**{
                    ENTITY_LIST_KEY: entity_list,
                    INPUT_TEXT_KEY: text,
                })
            )
            response = await self._model.completion_async(
                messages=retry_rel_builder.build(),
                **call_kwargs,
            )  # type: ignore
            self._check_truncation(response, "relationship extraction (retry)")  # type: ignore[arg-type]
        results = response.content  # type: ignore[union-attr]

        for i in range(self._max_gleanings):
            if COMPLETION_DELIMITER in response.content:  # type: ignore[union-attr]
                break

            rel_builder.add_assistant_message(response.content)  # type: ignore[union-attr]
            rel_builder.add_user_message(RELATIONSHIP_CONTINUE_PROMPT)

            response = await self._model.completion_async(
                messages=rel_builder.build(),
                **call_kwargs,
            )  # type: ignore
            self._check_truncation(response, f"relationship gleaning {i + 1}")  # type: ignore[arg-type]

            if not response.content.strip():  # type: ignore[union-attr]
                break
            results += response.content  # type: ignore[union-attr]

        return results

    def _check_truncation(
        self, response: "LLMCompletionResponse", phase: str
    ) -> bool:
        """Log a warning when the LLM response appears truncated.

        Truncation signals:
        1. ``finish_reason`` is ``"length"`` (token limit hit).
        2. The ``<|COMPLETE|>`` completion delimiter is absent from the
           response content (the extraction prompt instructs the model to
           emit this marker when it has finished).
        """
        finish_reason = (
            response.choices[0].finish_reason
            if response.choices
            else None
        )
        content = response.content

        if finish_reason == "length":
            logger.warning(
                "LLM response TRUNCATED during %s — finish_reason='length'. "
                "Response length: %d chars. The model hit its max-token limit; "
                "entities/relationships may be missing. Consider increasing "
                "max_tokens or reducing chunk size.",
                phase,
                len(content),
            )
            return False
        elif COMPLETION_DELIMITER not in content and content.strip():
            logger.warning(
                "LLM response may be TRUNCATED during %s — the expected "
                "completion delimiter '%s' is missing from the response "
                "(length: %d chars, last 80 chars: '%.80s'). "
                "Entities/relationships may be incomplete.",
                phase,
                COMPLETION_DELIMITER,
                len(content),
                content[-80:] if content else "",
            )
            return True
        return False

    def _process_result(
        self,
        result: str,
        source_id: str,
        tuple_delimiter: str,
        record_delimiter: str,
        expect_relationships: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Parse the result string into entity and relationship data frames."""
        entities: list[dict[str, Any]] = []
        relationships: list[dict[str, Any]] = []

        records = [r.strip() for r in result.split(record_delimiter)]

        for raw_record in records:
            record = re.sub(r"^\(|\)$", "", raw_record.strip())
            if not record or record == COMPLETION_DELIMITER:
                continue

            # LLMs sometimes drop the trailing `>` of the `<|>` tuple delimiter
            # (emitting `<|`), which merges the surrounding fields after split.
            # Recover by normalizing any `<|` not followed by `>` — except the
            # `<|COMPLETE|>` completion marker — back into `<|>`.
            record = re.sub(r"<\|(?!>)(?!COMPLETE\|>)", tuple_delimiter, record)

            record_attributes = record.split(tuple_delimiter)
            record_type = record_attributes[0]

            if record_type == '"entity"' and len(record_attributes) >= 4:
                entity_name = clean_str(
                    _strip_placeholder_brackets(record_attributes[1]).upper()
                )
                entity_type = clean_str(
                    _strip_placeholder_brackets(record_attributes[2]).upper()
                )
                entity_description = clean_str(
                    _strip_placeholder_brackets(record_attributes[3])
                )
                if "|" in entity_name or "|" in entity_type:
                    continue  # residual delimiter — two fields merged into one after mis-split
                if not entity_name:  # blank name produces empty node IDs downstream
                    continue
                entities.append({
                    "title": entity_name,
                    "type": entity_type,
                    "description": entity_description,
                    "source_id": source_id,
                })

            if record_type == '"relationship"' and len(record_attributes) >= 5:
                source = clean_str(
                    _strip_placeholder_brackets(record_attributes[1]).upper()
                )
                target = clean_str(
                    _strip_placeholder_brackets(record_attributes[2]).upper()
                )
                edge_description = clean_str(
                    _strip_placeholder_brackets(record_attributes[3])
                )
                if "|" in source or "|" in target:
                    continue  # residual delimiter — endpoint name is garbage
                if not source or not target:  # blank endpoint produces empty edge source/target
                    continue
                try:
                    weight = float(record_attributes[-1])
                except ValueError:
                    weight = 1.0

                relationships.append({
                    "source": source,
                    "target": target,
                    "description": edge_description,
                    "source_id": source_id,
                    "weight": weight,
                })

        entities_df = pd.DataFrame(entities) if entities else _empty_entities_df()
        relationships_df = (
            pd.DataFrame(relationships) if relationships else _empty_relationships_df()
        )

        if expect_relationships and entities and not relationships:
            logger.warning(
                "Extraction for source '%s' produced %d entities but 0 "
                "relationships — the LLM response was likely truncated before "
                "emitting relationship records.",
                source_id,
                len(entities),
            )

        return entities_df, relationships_df


def _parse_entity_names(result: str) -> list[str]:
    """Extract a deduplicated list of entity names from a raw extraction result string."""
    names = []
    seen: set[str] = set()
    for raw_record in result.split(RECORD_DELIMITER):
        record = re.sub(r"^\(|\)$", "", raw_record.strip())
        if not record or record == COMPLETION_DELIMITER:
            continue
        parts = record.split(TUPLE_DELIMITER)
        if parts and parts[0] == '"entity"' and len(parts) >= 2:
            name = clean_str(_strip_placeholder_brackets(parts[1]).upper())
            if name and name not in seen:
                seen.add(name)
                names.append(name)
    return names


def _parse_entity_list_with_types(result: str) -> str:
    """Return a formatted string of 'NAME (TYPE)' lines from a raw extraction result."""
    lines = []
    seen: set[str] = set()
    for raw_record in result.split(RECORD_DELIMITER):
        record = re.sub(r"^\(|\)$", "", raw_record.strip())
        if not record or record == COMPLETION_DELIMITER:
            continue
        parts = record.split(TUPLE_DELIMITER)
        if parts and parts[0] == '"entity"' and len(parts) >= 3:
            name = clean_str(_strip_placeholder_brackets(parts[1]).upper())
            etype = clean_str(_strip_placeholder_brackets(parts[2]).upper())
            if name and name not in seen:
                seen.add(name)
                lines.append(f"- {name} ({etype})")
    return "\n".join(lines)


def _strip_placeholder_brackets(value: str) -> str:
    # Strip three categories of parser noise:
    # 1. Leading < from angle-bracket placeholders copied from the prompt spec.
    # 2. Trailing [TYPE] annotations copied from the salience list (e.g. "FOO [TERMINOLOGY]").
    # 3. Stray <> [] | characters at either end left by partial mis-splits.
    v = value.strip().lstrip("<").rstrip(">").strip()
    v = _TRAILING_BRACKET_RE.sub("", v)  # strip trailing [TYPE] annotation
    return v.strip("<>[]|").strip()  # strip any remaining stray delimiter chars


def _empty_entities_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["title", "type", "description", "source_id"])


def _empty_relationships_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["source", "target", "weight", "description", "source_id"]
    )
