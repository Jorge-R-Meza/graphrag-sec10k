# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing cluster_graph method definition."""

import logging
from collections import defaultdict

import pandas as pd

from graphrag.graphs.hierarchical_leiden import hierarchical_leiden
from graphrag.graphs.stable_lcc import stable_lcc

Communities = list[tuple[int, int, int, list[str]]]


logger = logging.getLogger(__name__)


def cluster_graph(
    edges: pd.DataFrame,
    max_cluster_size: int,
    use_lcc: bool,
    seed: int | None = None,
    node_groups: dict[str, str] | None = None,
) -> Communities:
    """Apply a hierarchical clustering algorithm to a relationships DataFrame.

    When *node_groups* is provided (a mapping of node_title -> group_key),
    clustering runs independently per group (document-scoped communities).
    Cross-group edges are excluded from clustering but remain in the
    relationships table.
    """
    if node_groups is not None:
        node_id_to_community_map, parent_mapping = _compute_grouped_leiden_communities(
            edges=edges,
            node_groups=node_groups,
            max_cluster_size=max_cluster_size,
            use_lcc=use_lcc,
            seed=seed,
        )
    else:
        node_id_to_community_map, parent_mapping = _compute_leiden_communities(
            edges=edges,
            max_cluster_size=max_cluster_size,
            use_lcc=use_lcc,
            seed=seed,
        )

    levels = sorted(node_id_to_community_map.keys())

    clusters: dict[int, dict[int, list[str]]] = {}
    for level in levels:
        result: dict[int, list[str]] = defaultdict(list)
        clusters[level] = result
        for node_id, community_id in node_id_to_community_map[level].items():
            result[community_id].append(node_id)

    results: Communities = []
    for level in clusters:
        for cluster_id, nodes in clusters[level].items():
            results.append((level, cluster_id, parent_mapping[cluster_id], nodes))
    return results


# Taken from graph_intelligence & adapted
def _compute_leiden_communities(
    edges: pd.DataFrame,
    max_cluster_size: int,
    use_lcc: bool,
    seed: int | None = None,
) -> tuple[dict[int, dict[str, int]], dict[int, int]]:
    """Return Leiden root communities and their hierarchy mapping."""
    edge_df = edges.copy()

    # Normalize edge direction and deduplicate (undirected graph), summing weights
    # for reversed pairs (A→B and B→A) rather than arbitrarily keeping one.
    lo = edge_df[["source", "target"]].min(axis=1)
    hi = edge_df[["source", "target"]].max(axis=1)
    edge_df = edge_df.copy()
    edge_df["source"] = lo
    edge_df["target"] = hi
    if "weight" in edge_df.columns:
        edge_df = (
            edge_df.groupby(["source", "target"], sort=False)
            .agg(weight=("weight", "sum"))
            .reset_index()
        )
    else:
        edge_df.drop_duplicates(subset=["source", "target"], inplace=True)

    if use_lcc:
        edge_df = stable_lcc(edge_df)

    weights = (
        edge_df["weight"].astype(float)
        if "weight" in edge_df.columns
        else pd.Series(1.0, index=edge_df.index)
    )
    edge_list: list[tuple[str, str, float]] = sorted(
        zip(
            edge_df["source"].astype(str),
            edge_df["target"].astype(str),
            weights,
            strict=True,
        )
    )

    community_mapping = hierarchical_leiden(
        edge_list, max_cluster_size=max_cluster_size, random_seed=seed
    )
    results: dict[int, dict[str, int]] = {}
    hierarchy: dict[int, int] = {}
    for partition in community_mapping:
        results[partition.level] = results.get(partition.level, {})
        results[partition.level][partition.node] = partition.cluster

        hierarchy[partition.cluster] = (
            partition.parent_cluster if partition.parent_cluster is not None else -1
        )

    return results, hierarchy


def _compute_grouped_leiden_communities(
    edges: pd.DataFrame,
    node_groups: dict[str, str],
    max_cluster_size: int,
    use_lcc: bool,
    seed: int | None = None,
) -> tuple[dict[int, dict[str, int]], dict[int, int]]:
    """Run Leiden independently per node group; merge with unique cluster ids.

    Cross-group edges are excluded from clustering (both endpoint nodes must
    belong to the same group). Root parent (-1) is preserved.
    """
    edge_df = edges.copy()
    source_group = edge_df["source"].map(node_groups)
    target_group = edge_df["target"].map(node_groups)
    same_group = source_group.notna() & (source_group == target_group)

    dropped = int((~same_group).sum())
    if dropped:
        logger.info(
            "document-scoped clustering: excluding %d cross-group edges (of %d total).",
            dropped,
            len(edge_df),
        )

    edge_df = edge_df[same_group].copy()
    edge_df["_group"] = source_group[same_group].values

    merged_results: dict[int, dict[str, int]] = {}
    merged_hierarchy: dict[int, int] = {}
    offset = 0

    for group in sorted(edge_df["_group"].unique()):
        group_edges = edge_df[edge_df["_group"] == group].drop(columns=["_group"])
        group_results, group_hierarchy = _compute_leiden_communities(
            edges=group_edges,
            max_cluster_size=max_cluster_size,
            use_lcc=use_lcc,
            seed=seed,
        )
        if not group_hierarchy:
            continue

        max_id = max(group_hierarchy.keys())

        for level, node_map in group_results.items():
            level_map = merged_results.setdefault(level, {})
            for node, cluster in node_map.items():
                level_map[node] = cluster + offset

        for cluster, parent in group_hierarchy.items():
            merged_hierarchy[cluster + offset] = (
                parent + offset if parent != -1 else -1
            )

        offset += max_id + 1

    return merged_results, merged_hierarchy
