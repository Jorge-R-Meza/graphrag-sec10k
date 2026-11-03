# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Parameterization settings for the default configuration."""

from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from graphrag.config.defaults import graphrag_config_defaults
from graphrag.prompts.index.extract_graph import (
    ENTITY_GLEANING_PROMPT,
    GRAPH_EXTRACTION_PROMPT,
    RELATIONSHIP_EXTRACTION_PROMPT,
    SECOND_PHASE_PROMPT,
)


@dataclass
class ExtractGraphPrompts:
    """Graph extraction prompt templates."""

    extraction_prompt: str
    gleaning_prompt: str
    relationship_prompt: str


class ExtractGraphConfig(BaseModel):
    """Configuration section for entity extraction."""

    completion_model_id: str = Field(
        description="The model ID to use for text embeddings.",
        default=graphrag_config_defaults.extract_graph.completion_model_id,
    )
    model_instance_name: str = Field(
        description="The model singleton instance name. This primarily affects the cache storage partitioning.",
        default=graphrag_config_defaults.extract_graph.model_instance_name,
    )
    prompt: str | None = Field(
        description="The entity extraction prompt to use.",
        default=graphrag_config_defaults.extract_graph.prompt,
    )
    entity_types: list[str] = Field(
        description="The entity extraction entity types to use.",
        default=graphrag_config_defaults.extract_graph.entity_types,
    )
    max_gleanings: int = Field(
        description="The maximum number of entity gleanings to use.",
        default=graphrag_config_defaults.extract_graph.max_gleanings,
    )
    gleaning_prompt: str | None = Field(
        description="Path to a custom entity gleaning prompt file. Falls back to built-in prompt if not set.",
        default=None,
    )
    relationship_prompt: str | None = Field(
        description="Path to a custom relationship extraction prompt file. Falls back to built-in prompt if not set.",
        default=None,
    )
    two_phase: bool = Field(
        description="Enable two-phase extraction: phase 1 extracts entities globally; phase 2 extracts relationships with the entity list injected as salience context.",
        default=graphrag_config_defaults.extract_graph.two_phase,
    )
    relationship_passes: int = Field(
        description="Maximum number of relationship extraction passes in phase 2. The loop stops early when no new entity titles appear.",
        default=graphrag_config_defaults.extract_graph.relationship_passes,
    )
    variant: str = Field(
        description="Orphan-endpoint handling strategy: 'v5' drops orphans, 'v4_middle_path' classifies and keeps successes, 'v4_clean_inferred' classifies and mints residual as UNKNOWN.",
        default=graphrag_config_defaults.extract_graph.variant,
    )
    relationship_entity_budget: int | None = Field(
        description=(
            "Two-phase extraction only. If set, each chunk's relationship pass "
            "receives only the top-N most relevant known entities (lexical matches "
            "in the chunk, then nearest by embedding) instead of the full "
            "document-wide list — keeping the prompt within the context window and "
            "cutting token cost on long documents. None (default) injects the full "
            "entity list, preserving current behavior. Ignored when the merged "
            "entity count is already <= N. Reuses the embed_text embedding model."
        ),
        default=graphrag_config_defaults.extract_graph.relationship_entity_budget,
    )
    relationship_max_tokens: int | None = Field(
        description="Max output tokens for the relationship call. None inherits the model's call_args.max_tokens.",
        default=None,
    )
    relationship_gleanings: int = Field(
        description="Phase-2 relationship recovery turns (0 disables).",
        default=1,
    )
    inline_endpoint_classification: bool = Field(
        description="After each chunk's phase-2 relationship pass, immediately classify any relationship endpoints that are not yet declared entities, using the chunk as context. Adds at most one extra LLM call per chunk per pass.",
        default=False,
    )
    phase1_entities_only: bool = Field(
        description="In phase 1, suppress relationship output and use the entity-only continuation prompt. Measured +36%% phase-1 entity records vs combined extraction.",
        default=False,
    )
    pass_entity_list: bool = Field(
        description="LOAD-BEARING. Inject the accumulated entity list as salience context into each phase-2 chunk prompt. Must stay True for normal operation: ablation shows relationships −41%%, orphan share 8.7%%→35.8%%, rel/entity 2.0→0.8.",
        default=True,
    )
    filer_alias_file: str | None = Field(
        description=(
            "Path to a JSON mapping document TITLE -> canonical filer ENTITY name; renames "
            "generic self-references (THE COMPANY, THE REGISTRANT) before the cross-chunk merge."
        ),
        default=graphrag_config_defaults.extract_graph.filer_alias_file,
    )

    def resolved_prompts(self) -> ExtractGraphPrompts:
        """Get the resolved graph extraction prompts."""
        if self.relationship_prompt:
            rel_prompt = Path(self.relationship_prompt).read_text(encoding="utf-8")
        elif self.two_phase:
            rel_prompt = SECOND_PHASE_PROMPT
        else:
            rel_prompt = RELATIONSHIP_EXTRACTION_PROMPT
        return ExtractGraphPrompts(
            extraction_prompt=Path(self.prompt).read_text(encoding="utf-8")
            if self.prompt
            else GRAPH_EXTRACTION_PROMPT,
            gleaning_prompt=Path(self.gleaning_prompt).read_text(encoding="utf-8")
            if self.gleaning_prompt
            else ENTITY_GLEANING_PROMPT,
            relationship_prompt=rel_prompt,
        )
