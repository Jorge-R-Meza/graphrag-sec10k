# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Workflow for deterministic entity alias resolution.

Runs after extract_graph and before finalize_graph.
Only active when resolve_entity_aliases.enabled = True in config.
"""

import logging

from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.data_model.data_reader import DataReader
from graphrag.index.operations.resolve_entity_aliases.resolve_entity_aliases import (
    apply_alias_mapping,
    load_alias_mapping,
)
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.typing.workflow import WorkflowFunctionOutput

logger = logging.getLogger(__name__)


async def run_workflow(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """Apply entity alias resolution: merge alias nodes into canonical nodes."""
    logger.info("Workflow started: resolve_entity_aliases")

    if not config.resolve_entity_aliases.enabled:
        logger.info("Workflow resolve_entity_aliases: disabled — skipping.")
        return WorkflowFunctionOutput(result=None)

    mapping_file = config.resolve_entity_aliases.mapping_file
    if not mapping_file:
        logger.warning(
            "resolve_entity_aliases.enabled=True but no mapping_file specified. "
            "Skipping alias resolution."
        )
        return WorkflowFunctionOutput(result=None)

    alias_to_canonical = load_alias_mapping(mapping_file)
    if not alias_to_canonical:
        logger.info("resolve_entity_aliases: mapping is empty — nothing to merge.")
        return WorkflowFunctionOutput(result=None)

    reader = DataReader(context.output_table_provider)
    entities = await reader.entities()
    relationships = await reader.relationships()

    merged_entities, merged_relationships = apply_alias_mapping(
        entities=entities,
        relationships=relationships,
        alias_to_canonical=alias_to_canonical,
    )

    await context.output_table_provider.write_dataframe("entities", merged_entities)
    await context.output_table_provider.write_dataframe("relationships", merged_relationships)

    logger.info(
        "Workflow completed: resolve_entity_aliases (%d entities, %d relationships after merge).",
        len(merged_entities),
        len(merged_relationships),
    )
    return WorkflowFunctionOutput(result={
        "entities": merged_entities,
        "relationships": merged_relationships,
    })
