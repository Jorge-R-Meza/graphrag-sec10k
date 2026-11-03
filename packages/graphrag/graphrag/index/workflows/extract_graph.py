# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing run_workflow method definition."""

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING
import json

import pandas as pd
from graphrag_llm.completion import create_completion
from graphrag_llm.circuit_breaker import PipelineHalted
from graphrag_llm.embedding import create_embedding

from graphrag.cache.cache_key_creator import cache_key_creator
from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
from graphrag.config.enums import AsyncType
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.data_model.data_reader import DataReader
from graphrag.index.operations.extract_graph.extract_graph import (
    extract_graph as extractor,
)
from graphrag.index.operations.summarize_descriptions.summarize_descriptions import (
    summarize_descriptions,
    summarize_descriptions_scoped,
)
from graphrag.index.operations.summarize_descriptions.scoped_summary_extractor import (
    sanitize_scope_label,
)
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.typing.workflow import WorkflowFunctionOutput
from graphrag.index.utils.derive_from_rows import PipelineHaltedError

if TYPE_CHECKING:
    from graphrag_llm.completion import LLMCompletion
    from graphrag_llm.embedding import LLMEmbedding

logger = logging.getLogger(__name__)


async def run_workflow(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """All the steps to create the base entity graph."""
    logger.info("Workflow started: extract_graph")
    reader = DataReader(context.output_table_provider)
    text_units = await reader.text_units()
    documents = await reader.documents()

    # Build text_unit_id -> document title mapping for scoped facts. Falling
    # back to document_id keeps behavior stable when title is unavailable.
    doc_id_to_title: dict[str, str] = {}
    if not documents.empty and "id" in documents.columns:
        for row in documents.itertuples(index=False):
            doc_id = str(getattr(row, "id", "") or "")
            if not doc_id:
                continue
            title = getattr(row, "title", None)
            doc_id_to_title[doc_id] = str(title).strip() if title is not None else doc_id
    else:
        msg = (
            "ALARM scoped_facts document mapping unavailable: documents table is "
            "empty or missing id column. Stopping indexing."
        )
        logger.error(msg)
        raise ValueError(msg)

    text_unit_to_document: dict[str, str] = {}
    missing_document_id: list[str] = []
    unresolved_document_id: list[str] = []

    scoped_cfg = config.scoped_facts

    def _scope_label(title: str) -> str:
        if scoped_cfg.enabled and scoped_cfg.label_pattern:
            title = re.sub(scoped_cfg.label_pattern, scoped_cfg.label_replacement, str(title))
        return sanitize_scope_label(title)

    for row in text_units.itertuples(index=False):
        tu_id = str(getattr(row, "id", "") or "")
        if not tu_id:
            continue
        doc_id = str(getattr(row, "document_id", "") or "")
        if not doc_id:
            missing_document_id.append(tu_id)
            continue
        if doc_id not in doc_id_to_title:
            unresolved_document_id.append(tu_id)
            continue
        text_unit_to_document[tu_id] = _scope_label(doc_id_to_title.get(doc_id, doc_id))

    logger.info(
        "scoped_facts: %d distinct scope labels: %s",
        len(set(text_unit_to_document.values())),
        sorted(set(text_unit_to_document.values())),
    )

    if missing_document_id:
        sample = ", ".join(missing_document_id[:5])
        msg = (
            "ALARM scoped_facts missing document_id on text_units: "
            f"count={len(missing_document_id)} sample=[{sample}]. "
            "Stopping indexing."
        )
        logger.error(msg)
        raise ValueError(msg)

    if unresolved_document_id:
        sample = ", ".join(unresolved_document_id[:5])
        msg = (
            "ALARM scoped_facts unresolved document_id references from text_units: "
            f"count={len(unresolved_document_id)} sample=[{sample}]. "
            "Stopping indexing."
        )
        logger.error(msg)
        raise ValueError(msg)

    # Build text_unit_id -> canonical filer entity name for self-reference de-weld.
    self_reference_map: dict[str, str] | None = None
    if config.extract_graph.filer_alias_file and "document_id" in text_units.columns:
        alias_map: dict[str, str] = json.loads(
            Path(config.extract_graph.filer_alias_file).read_text(encoding="utf-8")
        )
        doc_canonical = {
            doc_id: alias_map[str(title)]
            for doc_id, title in zip(documents["id"], documents["title"], strict=True)
            if str(title) in alias_map
        }
        self_reference_map = {
            str(tu): doc_canonical[doc]
            for tu, doc in zip(text_units["id"], text_units["document_id"], strict=True)
            if doc in doc_canonical
        }
        logger.info(
            "extract_graph: filer_alias_file mapped %d/%d text units to %d filer names",
            len(self_reference_map),
            len(text_units),
            len(set(self_reference_map.values())),
        )

    extraction_model_config = config.get_completion_model_config(
        config.extract_graph.completion_model_id
    )
    extraction_prompts = config.extract_graph.resolved_prompts()
    extraction_model = create_completion(
        extraction_model_config,
        cache=context.cache.child(config.extract_graph.model_instance_name),
        cache_key_creator=cache_key_creator,
    )

    summarization_model_config = config.get_completion_model_config(
        config.summarize_descriptions.completion_model_id
    )
    summarization_prompts = config.summarize_descriptions.resolved_prompts()
    summarization_model = create_completion(
        summarization_model_config,
        cache=context.cache.child(config.summarize_descriptions.model_instance_name),
        cache_key_creator=cache_key_creator,
    )

    # Create embedding model for top-k entity selection if enabled.
    embedding_model: LLMEmbedding | None = None
    if config.extract_graph.two_phase and config.extract_graph.relationship_entity_budget:
        embedding_model_config = config.get_embedding_model_config(
            config.embed_text.embedding_model_id
        )
        embedding_model = create_embedding(
            embedding_model_config,
            cache=context.cache.child(config.embed_text.model_instance_name),
            cache_key_creator=cache_key_creator,
        )

    try:
        entities, relationships, raw_entities, raw_relationships, covariates_df = await extract_graph(
            text_units=text_units,
            callbacks=context.callbacks,
            extraction_model=extraction_model,
            extraction_prompt=extraction_prompts.extraction_prompt,
            extraction_gleaning_prompt=extraction_prompts.gleaning_prompt,
            extraction_relationship_prompt=extraction_prompts.relationship_prompt,
            entity_types=config.extract_graph.entity_types,
            max_gleanings=config.extract_graph.max_gleanings,
            extraction_num_threads=config.concurrent_requests,
            extraction_async_type=config.async_mode,
            two_phase=config.extract_graph.two_phase,
            relationship_passes=config.extract_graph.relationship_passes,
            variant=config.extract_graph.variant,
            embedding_model=embedding_model,
            relationship_entity_budget=config.extract_graph.relationship_entity_budget,
            relationship_max_tokens=config.extract_graph.relationship_max_tokens,
            relationship_gleanings=config.extract_graph.relationship_gleanings,
            inline_endpoint_classification=config.extract_graph.inline_endpoint_classification,
            phase1_entities_only=config.extract_graph.phase1_entities_only,
            pass_entity_list=config.extract_graph.pass_entity_list,
            summarization_model=summarization_model,
            max_summary_length=config.summarize_descriptions.max_length,
            max_input_tokens=config.summarize_descriptions.max_input_tokens,
            summarization_max_tokens=config.summarize_descriptions.max_tokens,
            summarization_prompt=summarization_prompts.summarize_prompt,
            summarization_num_threads=config.concurrent_requests,
            scoped_facts_enabled=config.scoped_facts.enabled,
            scoped_facts_emit_claims=config.scoped_facts.emit_claims,
            scoped_facts_derive_scope=config.scoped_facts.derive_scope,
            scoped_facts_context_relationships=config.scoped_facts.context_relationships,
            text_unit_to_document=text_unit_to_document,
        )
    except (PipelineHaltedError, PipelineHalted) as e:
        reason = e.reason if isinstance(e, PipelineHalted) else e.reason
        done = getattr(e, "done", 0)
        total = getattr(e, "total", 0)
        logger.warning(
            "Workflow extract_graph paused: %s (%d/%d items this batch). "
            "Re-run 'graphrag index' to resume.",
            reason, done, total,
        )
        context.state["paused"] = {
            "workflow": "extract_graph",
            "reason": reason,
            "done": done,
            "total": total,
        }
        return WorkflowFunctionOutput(result=None, stop=True)

    # Map each entity's text_unit_ids -> unique sorted document_ids
    tid_to_doc = dict(zip(text_units["id"], text_units["document_id"]))
    entities["document_ids"] = entities["text_unit_ids"].apply(
        lambda ids: sorted({tid_to_doc[tid] for tid in ids if tid in tid_to_doc})
    )

    # Mirror document_ids onto relationships (both merged and raw snapshots).
    for _df in (relationships, raw_relationships):
        if "text_unit_ids" in _df.columns:
            _df["document_ids"] = _df["text_unit_ids"].apply(
                lambda ids: sorted({tid_to_doc[tid] for tid in ids if tid in tid_to_doc})
            )

    await context.output_table_provider.write_dataframe("entities", entities)
    await context.output_table_provider.write_dataframe("relationships", relationships)

    if covariates_df is not None and len(covariates_df) > 0:
        from uuid import uuid4

        from graphrag.data_model.schemas import COVARIATES_FINAL_COLUMNS

        covariates_df = covariates_df.copy()
        covariates_df["id"] = [str(uuid4()) for _ in range(len(covariates_df))]
        covariates_df["human_readable_id"] = range(len(covariates_df))
        # Fill any columns required by COVARIATES_FINAL_COLUMNS that are absent.
        for _col in COVARIATES_FINAL_COLUMNS:
            if _col not in covariates_df.columns:
                covariates_df[_col] = None
        covariates_df = covariates_df.loc[:, COVARIATES_FINAL_COLUMNS]
        await context.output_table_provider.write_dataframe("covariates", covariates_df)

    if config.snapshots.raw_graph:
        await context.output_table_provider.write_dataframe(
            "raw_entities", raw_entities
        )
        await context.output_table_provider.write_dataframe(
            "raw_relationships", raw_relationships
        )

    logger.info("Workflow completed: extract_graph")
    return WorkflowFunctionOutput(
        result={
            "entities": entities,
            "relationships": relationships,
        }
    )


async def extract_graph(
    text_units: pd.DataFrame,
    callbacks: WorkflowCallbacks,
    extraction_model: "LLMCompletion",
    extraction_prompt: str,
    extraction_gleaning_prompt: str,
    extraction_relationship_prompt: str,
    entity_types: list[str],
    max_gleanings: int,
    extraction_num_threads: int,
    extraction_async_type: AsyncType,
    two_phase: bool,
    relationship_passes: int,
    variant: str,
    embedding_model: "LLMEmbedding | None",
    relationship_entity_budget: int | None,
    relationship_max_tokens: int | None,
    relationship_gleanings: int,
    summarization_model: "LLMCompletion",
    max_summary_length: int,
    max_input_tokens: int,
    summarization_max_tokens: int | None = None,
    summarization_prompt: str = "",
    summarization_num_threads: int = 4,
    inline_endpoint_classification: bool = False,
    phase1_entities_only: bool = False,
    pass_entity_list: bool = True,
    scoped_facts_enabled: bool = False,
    scoped_facts_emit_claims: bool = True,
    scoped_facts_derive_scope: bool = False,
    scoped_facts_context_relationships: int = 5,
    text_unit_to_document: dict[str, str] | None = None,
    self_reference_map: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """All the steps to create the base entity graph."""
    # this returns a graph for each text unit, to be merged later
    extracted_entities, extracted_relationships = await extractor(
        text_units=text_units,
        callbacks=callbacks,
        text_column="text",
        id_column="id",
        model=extraction_model,
        prompt=extraction_prompt,
        gleaning_prompt=extraction_gleaning_prompt,
        relationship_prompt=extraction_relationship_prompt,
        entity_types=entity_types,
        max_gleanings=max_gleanings,
        num_threads=extraction_num_threads,
        async_type=extraction_async_type,
        two_phase=two_phase,
        relationship_passes=relationship_passes,
        variant=variant,
        embedding_model=embedding_model,
        relationship_entity_budget=relationship_entity_budget,
        relationship_max_tokens=relationship_max_tokens,
        relationship_gleanings=relationship_gleanings,
        inline_endpoint_classification=inline_endpoint_classification,
        phase1_entities_only=phase1_entities_only,
        pass_entity_list=pass_entity_list,
        self_reference_map=self_reference_map,
    )

    if len(extracted_entities) == 0:
        error_msg = "Graph Extraction failed. No entities detected during extraction."
        logger.error(error_msg)
        raise ValueError(error_msg)

    if len(extracted_relationships) == 0:
        error_msg = (
            "Graph Extraction failed. No relationships detected during extraction."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    # copy these as is before any summarization
    raw_entities = extracted_entities.copy()
    raw_relationships = extracted_relationships.copy()

    if scoped_facts_enabled:
        entities, relationships, covariates_df = await get_summarized_entities_relationships_scoped(
            extracted_entities=extracted_entities,
            extracted_relationships=extracted_relationships,
            callbacks=callbacks,
            model=summarization_model,
            max_summary_length=max_summary_length,
            max_input_tokens=max_input_tokens,
            max_tokens=summarization_max_tokens,
            summarization_prompt=summarization_prompt,
            num_threads=summarization_num_threads,
            text_unit_to_document=text_unit_to_document or {},
            emit_claims=scoped_facts_emit_claims,
            derive_scope=scoped_facts_derive_scope,
            context_relationships=scoped_facts_context_relationships,
        )
    else:
        entities, relationships = await get_summarized_entities_relationships(
            extracted_entities=extracted_entities,
            extracted_relationships=extracted_relationships,
            callbacks=callbacks,
            model=summarization_model,
            max_summary_length=max_summary_length,
            max_input_tokens=max_input_tokens,
            summarization_prompt=summarization_prompt,
            num_threads=summarization_num_threads,
        )
        covariates_df = None

    return (entities, relationships, raw_entities, raw_relationships, covariates_df)


async def get_summarized_entities_relationships(
    extracted_entities: pd.DataFrame,
    extracted_relationships: pd.DataFrame,
    callbacks: WorkflowCallbacks,
    model: "LLMCompletion",
    max_summary_length: int,
    max_input_tokens: int,
    summarization_prompt: str,
    num_threads: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize the entities and relationships."""
    entity_summaries, relationship_summaries = await summarize_descriptions(
        entities_df=extracted_entities,
        relationships_df=extracted_relationships,
        callbacks=callbacks,
        model=model,
        max_summary_length=max_summary_length,
        max_input_tokens=max_input_tokens,
        prompt=summarization_prompt,
        num_threads=num_threads,
    )

    relationships = extracted_relationships.drop(columns=["description"]).merge(
        relationship_summaries, on=["source", "target"], how="left"
    )

    extracted_entities.drop(columns=["description"], inplace=True)
    entities = extracted_entities.merge(entity_summaries, on="title", how="left")
    return entities, relationships


async def get_summarized_entities_relationships_scoped(
    extracted_entities: pd.DataFrame,
    extracted_relationships: pd.DataFrame,
    callbacks: WorkflowCallbacks,
    model: "LLMCompletion",
    max_summary_length: int,
    max_input_tokens: int,
    summarization_prompt: str,
    num_threads: int,
    text_unit_to_document: dict[str, str],
    max_tokens: int | None = None,
    emit_claims: bool = True,
    derive_scope: bool = False,
    context_relationships: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Scope-aware summarization returning (entities, relationships, covariates)."""
    entity_summaries, relationship_summaries, covariates_df = await summarize_descriptions_scoped(
        entities_df=extracted_entities,
        relationships_df=extracted_relationships,
        callbacks=callbacks,
        model=model,
        max_summary_length=max_summary_length,
        max_input_tokens=max_input_tokens,
        max_tokens=max_tokens,
        prompt=summarization_prompt,
        num_threads=num_threads,
        text_unit_to_document=text_unit_to_document,
        emit_claims=emit_claims,
        derive_scope=derive_scope,
        context_relationships=context_relationships,
    )

    relationships = extracted_relationships.drop(columns=["description"]).merge(
        relationship_summaries, on=["source", "target"], how="left"
    )
    extracted_entities.drop(columns=["description"], inplace=True)
    entities = extracted_entities.merge(entity_summaries, on="title", how="left")
    return entities, relationships, covariates_df
