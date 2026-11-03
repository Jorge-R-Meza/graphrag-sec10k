# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing run_workflow method definition."""

import logging
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any, cast
from uuid import uuid4

import numpy as np
import pandas as pd
from graphrag_storage.tables.table import Table

from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.data_model.data_reader import DataReader
from graphrag.data_model.schemas import COMMUNITIES_FINAL_COLUMNS
from graphrag.index.operations.cluster_graph import cluster_graph
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.typing.workflow import WorkflowFunctionOutput

logger = logging.getLogger(__name__)


async def run_workflow(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """All the steps to transform final communities."""
    logger.info("Workflow started: create_communities")
    reader = DataReader(context.output_table_provider)
    relationships = await reader.relationships()

    max_cluster_size = config.cluster_graph.max_cluster_size
    use_lcc = config.cluster_graph.use_lcc
    seed = config.cluster_graph.seed

    # Document-scoped community detection
    node_groups: dict[str, str] | None = None
    if config.document_scoped_communities.enabled:
        text_units = await reader.text_units()
        documents = await reader.documents()

        # Build doc_id -> group label (apply group_pattern if configured)
        grp_pattern = config.document_scoped_communities.group_pattern
        grp_replacement = config.document_scoped_communities.group_replacement
        doc_id_to_title: dict[str, str] = dict(
            zip(documents["id"].astype(str), documents["title"].astype(str))
        )

        def _group_label(doc_id: str) -> str:
            title = doc_id_to_title.get(doc_id, doc_id)
            if grp_pattern:
                title = re.sub(grp_pattern, grp_replacement, title)
            return title

        # Build text_unit_id -> group label mapping
        tu_to_doc: dict[str, str] = {
            str(tu_id): _group_label(str(doc_id))
            for tu_id, doc_id in zip(
                text_units["id"].astype(str),
                text_units["document_id"].astype(str),
            )
        }
        exclude_patterns = config.document_scoped_communities.exclude_documents

        # Collect entity text_unit_ids from the entities table
        entity_text_units: dict[str, list[str]] = {}
        async with context.output_table_provider.open("entities") as entities_table:
            async for row in entities_table:
                title = str(row.get("title", ""))
                tus = row.get("text_unit_ids", None)
                if tus is None:
                    tus = []
                if isinstance(tus, str):
                    import json
                    try:
                        tus = json.loads(tus)
                    except Exception:
                        tus = [tus]
                entity_text_units[title] = [str(t) for t in tus]

        node_groups = _dominant_document_groups(
            entity_text_units=entity_text_units,
            text_unit_to_document=tu_to_doc,
            exclude_patterns=exclude_patterns,
        )
        logger.info(
            "Document-scoped communities: assigned %d/%d entities to a document group.",
            len(node_groups), len(entity_text_units),
        )

    async with (
        context.output_table_provider.open("entities") as entities_table,
        context.output_table_provider.open("communities") as communities_table,
    ):
        sample_rows = await create_communities(
            communities_table,
            entities_table,
            relationships,
            max_cluster_size=max_cluster_size,
            use_lcc=use_lcc,
            seed=seed,
            node_groups=node_groups,
            stamp_document_ids=config.document_scoped_communities.enabled,
        )

    logger.info("Workflow completed: create_communities")
    return WorkflowFunctionOutput(result=sample_rows)


async def create_communities(
    communities_table: Table,
    entities_table: Table,
    relationships: pd.DataFrame,
    max_cluster_size: int,
    use_lcc: bool,
    seed: int | None = None,
    node_groups: dict[str, str] | None = None,
    stamp_document_ids: bool = False,
) -> list[dict[str, Any]]:
    """Build communities from clustered relationships and stream rows to the table.

    Args
    ----
        communities_table: Table
            Output table to write community rows to.
        entities_table: Table
            Table containing entity rows.
        relationships: pd.DataFrame
            Relationships DataFrame with source, target, weight,
            text_unit_ids columns.
        max_cluster_size: int
            Maximum cluster size for hierarchical Leiden.
        use_lcc: bool
            Whether to restrict to the largest connected component.
        seed: int | None
            Random seed for deterministic clustering.
        node_groups: dict[str, str] | None
            When set, run document-scoped clustering (Leiden per group).
        stamp_document_ids: bool
            When True, carry document_ids onto community rows.

    Returns
    -------
        list[dict[str, Any]]
            Sample of up to 5 community rows for logging.
    """
    clusters = cluster_graph(
        relationships,
        max_cluster_size,
        use_lcc,
        seed=seed,
        node_groups=node_groups,
    )

    title_to_entity_id: dict[str, str] = {}
    async for row in entities_table:
        title_to_entity_id[row["title"]] = row["id"]

    communities = pd.DataFrame(
        clusters, columns=pd.Index(["level", "community", "parent", "title"])
    ).explode("title")
    communities["community"] = communities["community"].astype(int)

    # aggregate entity ids for each community
    entity_map = communities[["community", "title"]].copy()
    entity_map["entity_id"] = entity_map["title"].map(title_to_entity_id)
    entity_ids = (
        entity_map
        .dropna(subset=["entity_id"])
        .groupby("community")
        .agg(entity_ids=("entity_id", list))
        .reset_index()
    )

    # aggregate relationship ids per community, limited to
    # intra-community edges (source and target in the same community).
    # Process one hierarchy level at a time to keep intermediate
    # DataFrames small, then concat the grouped results once at the end.
    level_results = []
    for level in communities["level"].unique():
        level_comms = communities[communities["level"] == level]
        with_source = relationships.merge(
            level_comms, left_on="source", right_on="title", how="inner"
        )
        with_both = with_source.merge(
            level_comms, left_on="target", right_on="title", how="inner"
        )
        intra = with_both[with_both["community_x"] == with_both["community_y"]]
        if intra.empty:
            continue
        grouped = (
            intra
            .explode("text_unit_ids")
            .groupby(["community_x", "parent_x"])
            .agg(
                relationship_ids=("id", list),
                text_unit_ids=("text_unit_ids", list),
            )
            .reset_index()
        )
        grouped["level"] = level
        level_results.append(grouped)

    all_grouped = pd.concat(level_results, ignore_index=True).rename(
        columns={
            "community_x": "community",
            "parent_x": "parent",
        }
    )

    # deduplicate the lists
    all_grouped["relationship_ids"] = all_grouped["relationship_ids"].apply(
        lambda x: sorted(set(x))
    )
    all_grouped["text_unit_ids"] = all_grouped["text_unit_ids"].apply(
        lambda x: sorted(set(x))
    )

    # join it all up and add some new fields
    final_communities = all_grouped.merge(entity_ids, on="community", how="inner")
    final_communities["id"] = [str(uuid4()) for _ in range(len(final_communities))]
    final_communities["human_readable_id"] = final_communities["community"]
    final_communities["title"] = "Community " + final_communities["community"].astype(
        str
    )
    final_communities["parent"] = final_communities["parent"].astype(int)
    # collect the children so we have a tree going both ways
    parent_grouped = cast(
        "pd.DataFrame",
        final_communities.groupby("parent").agg(children=("community", "unique")),
    )
    final_communities = final_communities.merge(
        parent_grouped,
        left_on="community",
        right_on="parent",
        how="left",
    )
    # replace NaN children with empty list
    final_communities["children"] = final_communities["children"].apply(
        lambda x: x if isinstance(x, np.ndarray) else []  # type: ignore
    )
    # add fields for incremental update tracking
    final_communities["period"] = datetime.now(timezone.utc).date().isoformat()
    final_communities["size"] = final_communities.loc[:, "entity_ids"].apply(len)

    # Stamp document_ids per community when document-scoped clustering is active.
    if stamp_document_ids and node_groups is not None:
        final_communities["document_ids"] = final_communities["entity_ids"].apply(
            lambda eids: _build_community_doc_labels(eids, title_to_entity_id, node_groups)
        )

    output = final_communities.loc[:, COMMUNITIES_FINAL_COLUMNS]
    rows = output.to_dict("records")
    sample_rows: list[dict[str, Any]] = []
    for row in rows:
        row = _sanitize_row(row)
        await communities_table.write(row)
        if len(sample_rows) < 5:
            sample_rows.append(row)
    return sample_rows


def _sanitize_row(row: dict[str, Any]) -> dict[str, Any]:
    """Convert numpy types to native Python types for table serialization."""
    sanitized = {}
    for key, value in row.items():
        if isinstance(value, np.ndarray):
            sanitized[key] = value.tolist()
        elif isinstance(value, np.integer):
            sanitized[key] = int(value)
        elif isinstance(value, np.floating):
            sanitized[key] = float(value)
        else:
            sanitized[key] = value
    return sanitized


def _dominant_document_groups(
    entity_text_units: dict[str, list[str]],
    text_unit_to_document: dict[str, str],
    exclude_patterns: list[str] | None = None,
) -> dict[str, str]:
    """Assign each entity to its dominant source document (most text units).

    Entities tied in text-unit count are assigned alphabetically by document
    title. Entities that only appear in excluded documents are dropped (they
    will not join any community).

    Parameters
    ----------
    entity_text_units : dict[str, list[str]]
        Mapping of entity_title -> list of text_unit_ids.
    text_unit_to_document : dict[str, str]
        Mapping of text_unit_id -> document_title.
    exclude_patterns : list[str] | None
        Regex patterns matched against document titles. Matching documents
        are excluded from scoring.
    """
    compiled = [re.compile(p) for p in (exclude_patterns or [])]

    def is_excluded(doc_title: str) -> bool:
        return any(p.search(doc_title) for p in compiled)

    groups: dict[str, str] = {}
    for title, text_unit_ids in entity_text_units.items():
        counts: Counter[str] = Counter()
        for tu in text_unit_ids:
            doc = text_unit_to_document.get(tu)
            if doc is not None and not is_excluded(doc):
                counts[doc] += 1
        if not counts:
            continue
        # Sort by (-count, title) for deterministic tie-breaking
        dominant = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        groups[title] = dominant
    return groups


def _build_community_doc_labels(
    entity_ids: list,
    title_to_entity_id: dict[str, str],
    node_groups: dict[str, str],
) -> list[str]:
    """Return sorted unique document labels for all entities in a community."""
    # entity_ids is a list of UUIDs; we need to reverse-map to titles
    entity_id_to_title = {v: k for k, v in title_to_entity_id.items()}
    docs: set[str] = set()
    for eid in (entity_ids if isinstance(entity_ids, list) else list(entity_ids)):
        title = entity_id_to_title.get(str(eid))
        if title and title in node_groups:
            docs.add(node_groups[title])
    return sorted(docs)

