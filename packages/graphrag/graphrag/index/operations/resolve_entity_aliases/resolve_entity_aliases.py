# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Deterministic entity alias resolution via a hand-curated YAML mapping.

Merges alias forms of entities into their canonical node.  No LLM calls —
purely deterministic text substitution.

Mapping file format:
    - canonical: "International Business Machines"
      aliases: ["IBM", "Big Blue"]
"""

import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_alias_mapping(mapping_file: str) -> dict[str, str]:
    """Load a YAML alias mapping file and return alias -> canonical dict.

    Parameters
    ----------
    mapping_file : str
        Path to the YAML file containing alias definitions.

    Returns
    -------
    dict[str, str]
        Mapping of ``alias_upper -> canonical_upper`` for all alias forms.
    """
    path = Path(mapping_file)
    if not path.exists():
        msg = f"Alias mapping file not found: {mapping_file}"
        raise FileNotFoundError(msg)

    with path.open(encoding="utf-8") as f:
        entries = yaml.safe_load(f) or []

    alias_to_canonical: dict[str, str] = {}
    for entry in entries:
        canonical = str(entry.get("canonical", "")).strip().upper()
        if not canonical:
            logger.warning("Alias mapping entry has no canonical name, skipping: %s", entry)
            continue
        for alias in entry.get("aliases", []):
            alias_upper = str(alias).strip().upper()
            if alias_upper and alias_upper != canonical:
                alias_to_canonical[alias_upper] = canonical
                logger.debug("Alias mapping: %s -> %s", alias_upper, canonical)

    logger.info("Loaded %d alias mappings from %s.", len(alias_to_canonical), mapping_file)
    return alias_to_canonical


def _append_aliases(description: str, aliases: list[str]) -> str:
    """Append an 'Also known as:' suffix so aliases remain visible for retrieval."""
    if not aliases:
        return description
    also_known = "Also known as: " + ", ".join(aliases)
    base = (description or "").strip()
    return f"{base}\n\n{also_known}" if base else also_known


def apply_alias_mapping(
    entities: pd.DataFrame,
    relationships: pd.DataFrame,
    alias_to_canonical: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge aliased entities into their canonical forms.

    Parameters
    ----------
    entities : pd.DataFrame
        Entity DataFrame with columns: title, type, description,
        text_unit_ids, frequency, document_ids (optional).
    relationships : pd.DataFrame
        Relationship DataFrame with columns: source, target, description,
        weight, text_unit_ids.
    alias_to_canonical : dict[str, str]
        Mapping of alias_upper -> canonical_upper (both already uppercased).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Merged (entities, relationships) DataFrames.
    """
    if not alias_to_canonical:
        return entities, relationships

    # Collect all aliases that appear in the alias_to_canonical map.
    affected_aliases = set(alias_to_canonical.keys())

    entities_out = _merge_entities(entities, alias_to_canonical, affected_aliases)
    relationships_out = _rewrite_relationships(relationships, alias_to_canonical)

    return entities_out, relationships_out


def _merge_entities(
    entities: pd.DataFrame,
    alias_to_canonical: dict[str, str],
    affected_aliases: set[str],
) -> pd.DataFrame:
    """Merge alias entity rows into their canonical row."""
    if entities.empty:
        return entities

    rows = entities.copy()
    rows["_canonical"] = rows["title"].str.upper().map(
        lambda t: alias_to_canonical.get(t, t)
    )

    # Collect aliases per canonical for description suffix
    canonical_to_aliases: dict[str, list[str]] = {}
    for alias_upper, canonical_upper in alias_to_canonical.items():
        # Only include aliases that actually appear as entity titles
        if (rows["title"].str.upper() == alias_upper).any():
            canonical_to_aliases.setdefault(canonical_upper, []).append(alias_upper)

    # Sort so canonical rows come first (they "win" for first-aggregated fields).
    rows["_is_alias"] = rows["title"].str.upper().isin(affected_aliases)
    rows = rows.sort_values("_is_alias")  # False (canonical) first

    def _agg_list_col(col: str) -> pd.Series:
        """Union-set-aggregate a list column."""
        def union_lists(x):
            merged: list = []
            seen: set = set()
            for item in x:
                if isinstance(item, list):
                    for v in item:
                        k = str(v)
                        if k not in seen:
                            seen.add(k)
                            merged.append(v)
                else:
                    k = str(item)
                    if k not in seen:
                        seen.add(k)
                        merged.append(item)
            return merged
        if col in rows.columns:
            return rows.groupby("_canonical")[col].apply(union_lists)
        return pd.Series(dtype=object)

    def _first(col: str) -> pd.Series:
        if col in rows.columns:
            return rows.groupby("_canonical")[col].first()
        return pd.Series(dtype=object)

    def _sum(col: str) -> pd.Series:
        if col in rows.columns:
            return rows.groupby("_canonical")[col].sum()
        return pd.Series(dtype=object)

    def _join_desc(x: pd.Series) -> str:
        parts = []
        for item in x:
            if isinstance(item, list):
                parts.extend(str(v) for v in item if str(v).strip())
            elif str(item).strip():
                parts.append(str(item))
        return "; ".join(dict.fromkeys(parts))

    description_merged = rows.groupby("_canonical")["description"].apply(_join_desc)
    type_first = _first("type")
    text_unit_ids_merged = _agg_list_col("text_unit_ids")
    document_ids_merged = _agg_list_col("document_ids") if "document_ids" in rows.columns else None
    frequency_sum = _sum("frequency") if "frequency" in rows.columns else None

    result = pd.DataFrame({
        "title": description_merged.index,
        "type": type_first,
        "description": description_merged,
    })

    if frequency_sum is not None:
        result["frequency"] = frequency_sum.values

    if not text_unit_ids_merged.empty:
        result["text_unit_ids"] = text_unit_ids_merged.values

    if document_ids_merged is not None and not document_ids_merged.empty:
        result["document_ids"] = document_ids_merged.values

    # Append alias names to canonical descriptions for retrieval
    result["description"] = result.apply(
        lambda row: _append_aliases(
            row["description"],
            [a.title() for a in canonical_to_aliases.get(row["title"].upper(), [])],
        ),
        axis=1,
    )

    result = result.reset_index(drop=True)
    logger.info(
        "Alias resolution: %d entities -> %d after merging %d alias mappings.",
        len(entities), len(result), len(alias_to_canonical),
    )
    return result


def _rewrite_relationships(
    relationships: pd.DataFrame,
    alias_to_canonical: dict[str, str],
) -> pd.DataFrame:
    """Rewrite relationship endpoints, drop self-loops, re-group by (source, target)."""
    if relationships.empty:
        return relationships

    rels = relationships.copy()
    rels["source"] = rels["source"].str.upper().map(
        lambda t: alias_to_canonical.get(t, t)
    )
    rels["target"] = rels["target"].str.upper().map(
        lambda t: alias_to_canonical.get(t, t)
    )

    # Drop self-loops introduced by alias merging
    self_loops = rels["source"] == rels["target"]
    if self_loops.any():
        logger.info("Alias resolution: dropped %d self-loop relationships.", self_loops.sum())
    rels = rels[~self_loops].copy()

    if rels.empty:
        return rels

    # Re-group by (source, target): aggregate description and sum weight
    def _agg_desc(x):
        parts = []
        for item in x:
            if isinstance(item, list):
                parts.extend(str(v) for v in item if str(v).strip())
            elif str(item).strip():
                parts.append(str(item))
        return "; ".join(dict.fromkeys(parts))

    def _agg_tus(x):
        merged: list = []
        seen: set = set()
        for item in x:
            vals = item if isinstance(item, list) else [item]
            for v in vals:
                k = str(v)
                if k not in seen:
                    seen.add(k)
                    merged.append(v)
        return merged

    agg: dict = {
        "weight": ("weight", "sum"),
        "description": ("description", _agg_desc),
    }
    if "text_unit_ids" in rels.columns:
        agg["text_unit_ids"] = ("text_unit_ids", _agg_tus)

    # Keep first occurrence of any other columns
    extra_cols = [c for c in rels.columns if c not in {"source", "target", "weight", "description", "text_unit_ids"}]
    for col in extra_cols:
        agg[col] = (col, "first")

    grouped = (
        rels
        .groupby(["source", "target"], sort=False)
        .agg(**agg)
        .reset_index()
    )

    return grouped
