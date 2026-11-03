# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing extract_graph method."""

import asyncio
import hashlib
import logging
import pickle
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from graphrag_llm.circuit_breaker import PipelineHalted
from graphrag_llm.utils import CompletionMessagesBuilder

from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
from graphrag.config.enums import AsyncType
from graphrag.index.operations.extract_graph.graph_extractor import (
    COMPLETION_DELIMITER,
    RECORD_DELIMITER,
    TUPLE_DELIMITER,
    GraphExtractor,
    _empty_entities_df,
    _empty_relationships_df,
    _strip_placeholder_brackets,
)
from graphrag.index.operations.extract_graph.utils import filter_orphan_relationships
from graphrag.index.utils.derive_from_rows import derive_from_rows
from graphrag.index.utils.string import clean_str
from graphrag.prompts.index.extract_graph import (
    CLASSIFY_CHUNK_ENDPOINTS_PROMPT,
    CLASSIFY_ORPHAN_ENDPOINTS_PROMPT,
    ENTITY_CONTINUE_PROMPT,
)

if TYPE_CHECKING:
    from graphrag_llm.completion import LLMCompletion
    from graphrag_llm.embedding import LLMEmbedding

logger = logging.getLogger(__name__)

_MAX_SNIPPETS_PER_ORPHAN = 5
_ORPHAN_CLASSIFY_BATCH_SIZE = 25
_ORPHAN_FALLBACK_TYPE = "UNKNOWN"

_SELF_REFERENCE_NAMES = frozenset({
    "THE COMPANY", "COMPANY", "THE REGISTRANT", "REGISTRANT",
    "OUR COMPANY", "THE CORPORATION", "THE FIRM", "WE", "US", "OUR",
})


def _rename_self_references(
    df: pd.DataFrame | None,
    columns: list[str],
    self_reference_map: dict[str, str] | None,
    chunk_id: str,
) -> None:
    """Rename generic self-references (THE COMPANY, etc.) to the canonical filer name.

    Mutates *df* in-place; safe to call when df is None or empty.
    """
    if not self_reference_map or df is None or df.empty:
        return
    canonical = self_reference_map.get(chunk_id)
    if not canonical:
        return
    for col in columns:
        if col not in df.columns:
            continue
        mask = df[col].astype(str).str.strip().str.upper().isin(_SELF_REFERENCE_NAMES)
        if mask.any():
            df.loc[mask, col] = canonical


async def extract_graph(
    text_units: pd.DataFrame,
    callbacks: WorkflowCallbacks,
    text_column: str,
    id_column: str,
    model: "LLMCompletion",
    prompt: str,
    gleaning_prompt: str,
    relationship_prompt: str,
    entity_types: list[str],
    max_gleanings: int,
    num_threads: int,
    async_type: AsyncType,
    two_phase: bool = True,
    relationship_passes: int = 2,
    variant: str = "v5",
    embedding_model: "LLMEmbedding | None" = None,
    relationship_entity_budget: int | None = None,
    relationship_max_tokens: int | None = None,
    relationship_gleanings: int = 1,
    inline_endpoint_classification: bool = True,
    phase1_entities_only: bool = True,
    pass_entity_list: bool = True,
    self_reference_map: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract a graph from a piece of text using a language model."""
    if two_phase and relationship_prompt:
        return await _extract_graph_two_phase(
            text_units=text_units,
            callbacks=callbacks,
            text_column=text_column,
            id_column=id_column,
            model=model,
            prompt=prompt,
            gleaning_prompt=gleaning_prompt,
            relationship_prompt=relationship_prompt,
            entity_types=entity_types,
            max_gleanings=max_gleanings,
            num_threads=num_threads,
            async_type=async_type,
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

    # Single-pass: each chunk extracts entities and relationships together.
    extractor = GraphExtractor(
        model=model,
        prompt=prompt,
        gleaning_prompt=gleaning_prompt,
        relationship_prompt=relationship_prompt,
        max_gleanings=max_gleanings,
        on_error=lambda e, s, d: logger.error(
            "Entity Extraction Error", exc_info=e, extra={"stack": s, "details": d}
        ),
        relationship_max_tokens=relationship_max_tokens,
    )

    async def run_extraction(row):
        return await extractor(
            row[text_column].strip(),
            entity_types,
            row[id_column],
        )

    results = await derive_from_rows(
        text_units,
        run_extraction,
        callbacks,
        num_threads=num_threads,
        async_type=async_type,
        progress_msg="extract graph: ",
    )

    entity_dfs = [
        r[0] for r in results if r is not None and r[0] is not None and len(r[0]) > 0
    ]
    rel_dfs = [
        r[1] for r in results if r is not None and r[1] is not None and len(r[1]) > 0
    ]

    entities = _merge_entities(entity_dfs) if entity_dfs else _empty_entities_df()
    relationships = (
        _merge_relationships(rel_dfs) if rel_dfs else _empty_relationships_df()
    )
    relationships = filter_orphan_relationships(relationships, entities)
    logger.info(
        "Single-pass extraction complete: %d entities, %d relationships.",
        len(entities), len(relationships),
    )
    return entities, relationships


async def _extract_graph_two_phase(
    text_units: pd.DataFrame,
    callbacks: WorkflowCallbacks,
    text_column: str,
    id_column: str,
    model: "LLMCompletion",
    prompt: str,
    gleaning_prompt: str,
    relationship_prompt: str,
    entity_types: list[str],
    max_gleanings: int,
    num_threads: int,
    async_type: AsyncType,
    relationship_passes: int,
    variant: str,
    embedding_model: "LLMEmbedding | None" = None,
    relationship_entity_budget: int | None = None,
    relationship_max_tokens: int | None = None,
    relationship_gleanings: int = 1,
    inline_endpoint_classification: bool = False,
    phase1_entities_only: bool = False,
    pass_entity_list: bool = True,
    self_reference_map: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two-phase extraction: entity pass then multi-pass relationship extraction."""
    # Config fingerprint: changing entity_types, variant, max_gleanings, or prompts
    # should produce a fresh checkpoint directory so stale cached results are never
    # silently reused after a config change.
    _cfg_fp = hashlib.md5(  # nosec B324
        "|".join([
            "|".join(sorted(entity_types)),
            variant,
            str(max_gleanings),
            str(relationship_gleanings),
            str(relationship_passes),
            str(inline_endpoint_classification),
            str(phase1_entities_only),
            str(pass_entity_list),
            hashlib.md5(prompt.encode()).hexdigest()[:8],  # nosec B324
            hashlib.md5(relationship_prompt.encode()).hexdigest()[:8],  # nosec B324
            # Fold self_reference_map so adding/changing aliases invalidates
            # stale pickles (both phase-1 renames and phase-2 DOCUMENT IDENTITY
            # prefix change the effective prompt / output data).
            hashlib.md5(  # nosec B324
                repr(sorted((self_reference_map or {}).items())).encode()
            ).hexdigest()[:8],
        ]).encode()
    ).hexdigest()[:8]

    # ── Phase 1: entity extraction ────────────────────────────────────────────
    entity_extractor = GraphExtractor(
        model=model,
        prompt=prompt,
        gleaning_prompt=gleaning_prompt,
        relationship_prompt=relationship_prompt,
        max_gleanings=max_gleanings,
        on_error=lambda e, s, d: logger.error(
            "Entity Extraction Error", exc_info=e, extra={"stack": s, "details": d}
        ),
        relationship_max_tokens=relationship_max_tokens,
    )

    # ── Phase 1 checkpointing ────────────────────────────────────────────────
    p1_ckpt_dir = _phase2_checkpoint_dir(text_units, id_column, _cfg_fp)
    p1_ckpt_saved = _load_pass_checkpoint(p1_ckpt_dir, -1, f"p1_{_cfg_fp}")
    p1_pending_ids = {str(i) for i in text_units[id_column]} - set(p1_ckpt_saved)
    p1_pending_units = text_units[text_units[id_column].astype(str).isin(p1_pending_ids)]

    if p1_ckpt_saved:
        logger.info(
            "Phase 1: checkpoint loaded — %d chunks already done, %d still pending.",
            len(p1_ckpt_saved), len(p1_pending_units),
        )
    else:
        logger.info(
            "Phase 1 chunk-result checkpoints: %s", p1_ckpt_dir,
        )

    async def run_phase1(row):
        return await entity_extractor.extract_entities(
            row[text_column].strip(), entity_types, row[id_column]
        )

    async def run_phase1_with_checkpoint(
        row,
        _inner=run_phase1,
        _saved=p1_ckpt_saved,
        _dir=p1_ckpt_dir,
        _id_col=id_column,
    ):
        result = await _inner(row)
        if result is not None:
            chunk_id = str(row[_id_col])
            _rename_self_references(result, ["title"], self_reference_map, chunk_id)
            _saved[chunk_id] = result
            _save_chunk_checkpoint(_dir, -1, f"p1_{_cfg_fp}", chunk_id, result)
            logger.debug("Phase 1 chunk %s: %d entities.", chunk_id, len(result))
        return result

    if p1_pending_units.empty:
        logger.info(
            "Phase 1: all %d chunks loaded from checkpoint — skipping API calls.",
            len(p1_ckpt_saved),
        )
    else:
        await derive_from_rows(
            p1_pending_units,
            run_phase1_with_checkpoint,
            callbacks,
            num_threads=num_threads,
            async_type=async_type,
            progress_msg="extract graph phase 1 (entities): ",
        )

    # Reconstruct results in original text_units order.
    p1_results = [
        p1_ckpt_saved.get(str(row[id_column]))
        for _, row in text_units.iterrows()
    ]

    raw_entity_dfs: list[pd.DataFrame] = [
        df for df in p1_results if df is not None and len(df) > 0
    ]

    if not raw_entity_dfs:
        logger.warning("Phase 1 returned zero entities — returning empty graph.")
        return _empty_entities_df(), _empty_relationships_df()

    entities = _merge_entities(raw_entity_dfs)
    logger.info(
        "Phase 1 complete: %d unique entities from %d/%d chunks.",
        len(entities), len(raw_entity_dfs), len(text_units),
    )

    # ── Phase 2: relationship passes with salience injection ──────────────────
    pass_error_count = 0

    def _on_relationship_error(e, s, d):
        nonlocal pass_error_count
        pass_error_count += 1
        logger.error(
            "Relationship Extraction Error",
            exc_info=e,
            extra={"stack": s, "details": d},
        )

    relationship_extractor = GraphExtractor(
        model=model,
        prompt=prompt,
        gleaning_prompt=gleaning_prompt,
        relationship_prompt=relationship_prompt,
        max_gleanings=relationship_gleanings,
        on_error=_on_relationship_error,
        relationship_max_tokens=relationship_max_tokens,
    )

    all_raw_rel_dfs: list[pd.DataFrame] = []

    # Setup for top-k entity selection if enabled.
    use_topk = (
        embedding_model is not None
        and relationship_entity_budget is not None
        and relationship_entity_budget > 0
    )
    chunk_vec_by_id: dict[str, np.ndarray] = {}
    entity_vec_cache: dict[str, np.ndarray] = {}
    if use_topk:
        # Embed every chunk ONCE — chunk vectors never change across passes.
        chunk_ids = [str(i) for i in text_units[id_column]]
        logger.info(
            "Top-k entity selection enabled: budget=%d, chunks=%d, model=%s"
            " — embedding chunks now (CPU, may take several minutes)...",
            relationship_entity_budget,
            len(chunk_ids),
            embedding_model.__class__.__name__,
        )
        chunk_matrix = await _embed_texts(
            embedding_model, [str(t) for t in text_units[text_column]]
        )
        chunk_vec_by_id = dict(zip(chunk_ids, chunk_matrix, strict=True))
        logger.info("Chunk embeddings ready: %d vectors cached.", len(chunk_ids))

    ckpt_dir = _phase2_checkpoint_dir(text_units, id_column, _cfg_fp)
    logger.info(
        "Phase 2 chunk-result checkpoints: %s  (delete to force a clean restart)",
        ckpt_dir,
    )

    for pass_idx in range(max(1, relationship_passes)):
        full_reference = _format_entity_reference(entities) if pass_entity_list else ""
        prev_titles = set(entities["title"].astype(str))
        prev_rel_count = sum(len(df) for df in all_raw_rel_dfs)
        pass_error_count = 0
        pass_entity_rows_minted = 0
        pass_relationship_rows_minted = 0

        # Only build the embedding matrix when top-k will actually fire this pass.
        titles = [str(t) for t in entities["title"]]
        active_topk = use_topk and len(titles) > relationship_entity_budget
        entity_matrix = None
        if active_topk:
            new = [t for t in titles if t not in entity_vec_cache]
            cached_count = len(titles) - len(new)
            if new:
                logger.info(
                    "Pass %d: embedding %d entity titles (%d new, %d cached)...",
                    pass_idx + 1, len(titles), len(new), cached_count,
                )
                new_vecs = await _embed_texts(
                    embedding_model, [_entity_embed_text(entities, t) for t in new]
                )
                entity_vec_cache.update(zip(new, new_vecs, strict=True))
                logger.info(
                    "Pass %d: entity matrix ready — %d titles (%d new, %d cached).",
                    pass_idx + 1, len(titles), len(new), cached_count,
                )
            else:
                logger.info(
                    "Pass %d: entity matrix ready — %d titles (all %d cached).",
                    pass_idx + 1, len(titles), cached_count,
                )
            entity_matrix = np.vstack([entity_vec_cache[t] for t in titles])

        _known = frozenset(prev_titles)

        async def run_phase2(
            row,
            ref=full_reference,
            _titles=titles,
            _matrix=entity_matrix,
            _ents=entities,
            _active=active_topk,
            _k=relationship_entity_budget,
            _known_titles=_known,
        ):
            text = row[text_column].strip()
            chunk_id_str = str(row[id_column])
            entity_ref = ref
            selected_titles = None
            if _active:
                chunk_vec = chunk_vec_by_id[chunk_id_str]
                selected = _select_relevant_entities(text, chunk_vec, _titles, _matrix, _k)
                selected_titles = selected
                entity_ref = _format_entity_reference(_ents[_ents["title"].isin(selected)])

            # D4a: inject DOCUMENT IDENTITY prefix so the model attributes
            # self-referential statements to the canonical filer entity.
            filer = (self_reference_map or {}).get(chunk_id_str)
            if filer:
                entity_ref = (
                    f"DOCUMENT IDENTITY: This passage is from {filer}'s filing. "
                    f'In this passage, "we", "our", "us", "the Company", and '
                    f'"the Registrant" refer to {filer}. Attribute every '
                    f"self-referential statement to {filer} — never to another "
                    f"company, even one on this list.\n\n{entity_ref}"
                )

            extraction_result = await relationship_extractor(
                text,
                entity_types,
                row[id_column],
                entity_reference=entity_ref,
            )
            if extraction_result is None:
                # LLM call failed (e.g. auth error, network error) — do not cache
                return None
            ent_df, rel_df = extraction_result
            # D4b: rename self-references BEFORE inline classification so renamed
            # endpoints collapse into the declared filer rather than being minted as orphans.
            _rename_self_references(ent_df, ["title"], self_reference_map, chunk_id_str)
            _rename_self_references(rel_df, ["source", "target"], self_reference_map, chunk_id_str)
            if inline_endpoint_classification and rel_df is not None and len(rel_df) > 0:
                # Include any entities the model emitted this chunk so we don't
                # re-classify them — SECOND_PHASE_PROMPT forbids entity records
                # but weak models occasionally emit them anyway.
                declared = _known_titles | set(
                    ent_df["title"].astype(str) if ent_df is not None and len(ent_df) > 0 else []
                )
                minted = await _classify_chunk_endpoints(
                    relationships=rel_df,
                    known_titles=declared,
                    entity_types=entity_types,
                    model=model,
                    chunk_text=text,
                )
                if minted is not None and len(minted) > 0:
                    ent_df = pd.concat([ent_df, minted], ignore_index=True) if ent_df is not None else minted
            return ent_df, rel_df, selected_titles

        # ── Checkpoint: skip already-done chunks, save each result as it completes ──
        ref_hash = hashlib.md5(full_reference.encode()).hexdigest()[:8]  # nosec B324
        ckpt_saved = _load_pass_checkpoint(ckpt_dir, pass_idx, ref_hash)
        pending_ids = {str(i) for i in text_units[id_column]} - set(ckpt_saved)
        pending_units = text_units[text_units[id_column].astype(str).isin(pending_ids)]

        if ckpt_saved:
            logger.info(
                "Pass %d: checkpoint loaded — %d chunks already done, %d still pending.",
                pass_idx + 1, len(ckpt_saved), len(pending_units),
            )

        async def run_phase2_with_checkpoint(
            row,
            _inner=run_phase2,
            _saved=ckpt_saved,
            _dir=ckpt_dir,
            _pass=pass_idx,
            _rhash=ref_hash,
            _id_col=id_column,
        ):
            result = await _inner(row)
            if result is not None:
                chunk_id = str(row[_id_col])
                # Extract entities and relationships (handle both old and new tuple formats)
                if len(result) >= 3:
                    _e, _r, _selected = result  # New format with selected_titles
                    _saved[chunk_id] = result
                else:
                    _e, _r = result  # Old format, add empty selected_titles
                    result = (_e, _r, set())  # Wrap for checkpoint
                    _saved[chunk_id] = result
                
                _save_chunk_checkpoint(_dir, _pass, _rhash, chunk_id, result)
                logger.debug(
                    "Phase 2 pass %d chunk %s: %d entities, %d relationships.",
                    _pass + 1, chunk_id,
                    len(_e) if _e is not None else 0,
                    len(_r) if _r is not None else 0,
                )
            return result

        if pending_units.empty:
            logger.info(
                "Pass %d: all %d chunks loaded from checkpoint — no API calls needed.",
                pass_idx + 1, len(ckpt_saved),
            )
        else:
            await derive_from_rows(
                pending_units,
                run_phase2_with_checkpoint,
                callbacks,
                num_threads=num_threads,
                async_type=async_type,
                progress_msg=f"extract graph phase 2 pass {pass_idx + 1}: ",
            )

        # Reconstruct full-pass results in original text_units order.
        p2_results = [
            ckpt_saved.get(str(row[id_column]))
            for _, row in text_units.iterrows()
        ]

        for result in p2_results:
            if result is None:
                continue
            # Handle both old format (2-tuple) and new format (3-tuple with selected_titles)
            if len(result) >= 3:
                ent_df, rel_df, _ = result
            else:
                ent_df, rel_df = result
            if ent_df is not None and len(ent_df) > 0:
                pass_entity_rows_minted += len(ent_df)
                raw_entity_dfs.append(ent_df)
            if rel_df is not None and len(rel_df) > 0:
                pass_relationship_rows_minted += len(rel_df)
                all_raw_rel_dfs.append(rel_df)

        entities = _merge_entities(raw_entity_dfs)
        new_rel_count = sum(len(df) for df in all_raw_rel_dfs)
        entity_unique_delta = len(entities) - len(prev_titles)
        relationship_raw_delta = new_rel_count - prev_rel_count
        logger.info(
            "Phase 2 pass %d complete: entities %d->%d (+%d new), "
            "raw relationships this pass: +%d (cumulative raw: %d).",
            pass_idx + 1,
            len(prev_titles), len(entities), entity_unique_delta,
            relationship_raw_delta, new_rel_count,
        )
        logger.info(
            "METRIC phase2_pass=%d minted_entity_rows=%d minted_relationship_rows=%d unique_entity_delta=%d raw_relationship_delta=%d cumulative_entities=%d cumulative_raw_relationships=%d",
            pass_idx + 1,
            pass_entity_rows_minted,
            pass_relationship_rows_minted,
            entity_unique_delta,
            relationship_raw_delta,
            len(entities),
            new_rel_count,
        )

        n_incomplete = len(text_units) - len(ckpt_saved)
        if n_incomplete > 0:
            logger.warning(
                "Pass %d: %d/%d chunks have no result — checkpoint preserved;"
                " re-run to retry the missing chunks.",
                pass_idx + 1, n_incomplete, len(text_units),
            )

        new_titles = set(entities["title"].astype(str))
        if new_titles == prev_titles:
            if pass_error_count == 0 and n_incomplete == 0:
                logger.info(
                    "Phase 2 converged after %d pass(es) — no new entity titles.",
                    pass_idx + 1,
                )
                break
            logger.warning(
                "Phase 2 pass %d surfaced no new entities but logged %d error(s) "
                "and %d incomplete chunk(s) — NOT treating as converged; continuing.",
                pass_idx + 1,
                pass_error_count,
                n_incomplete,
            )

    # ── Orphan-endpoint handling ───────────────────────────────────────────────
    raw_rels_df = (
        pd.concat(all_raw_rel_dfs, ignore_index=True)
        if all_raw_rel_dfs
        else _empty_relationships_df()
    )

    if variant in ("v4_middle_path", "v4_clean_inferred"):
        fallback_type = (
            _ORPHAN_FALLBACK_TYPE if variant == "v4_clean_inferred" else None
        )
        orphan_rows = await _classify_orphan_endpoints(
            raw_relationships=raw_rels_df,
            entity_titles=set(entities["title"].astype(str)),
            model=model,
            entity_types=entity_types,
            fallback_type=fallback_type,
        )
        if orphan_rows:
            raw_entity_dfs.append(pd.DataFrame(orphan_rows))
            entities = _merge_entities(raw_entity_dfs)

    relationships = (
        _merge_relationships(all_raw_rel_dfs)
        if all_raw_rel_dfs
        else _empty_relationships_df()
    )
    relationships = filter_orphan_relationships(relationships, entities)
    logger.info(
        "Two-phase extraction complete: %d entities, %d relationships (after orphan filter).",
        len(entities), len(relationships),
    )
    return entities, relationships


async def _classify_chunk_endpoints(
    relationships: pd.DataFrame,
    known_titles: frozenset[str],
    entity_types: list[str],
    model: "LLMCompletion",
    chunk_text: str,
) -> pd.DataFrame | None:
    """Inline per-chunk classification of relationship endpoints not yet in the entity set.

    Called only when ``inline_endpoint_classification=True``.  Returns a small
    DataFrame of minted entity rows (title, type, description, source_id) or
    None if nothing was minted.
    """
    sources = set(relationships["source"].astype(str))
    targets = set(relationships["target"].astype(str))
    new_endpoints = sorted((sources | targets) - known_titles)
    if not new_endpoints:
        return None

    source_id = str(relationships["source_id"].iloc[0]) if "source_id" in relationships.columns else ""
    entity_types_csv = ", ".join(entity_types)
    endpoint_list = "\n".join(f"- {ep}" for ep in new_endpoints)

    messages = (
        CompletionMessagesBuilder()
        .add_user_message(
            CLASSIFY_CHUNK_ENDPOINTS_PROMPT.format(
                entity_types=entity_types_csv,
                endpoint_list=endpoint_list,
                input_text=chunk_text[:2000],  # trim to avoid token overflow
            )
        )
        .build()
    )
    try:
        response = await model.completion_async(messages=messages)
        result_text = response.content
    except PipelineHalted:
        raise
    except Exception:
        logger.exception("Error during inline endpoint classification")
        return None

    rows: list[dict] = []
    new_endpoints_set = set(new_endpoints)
    for raw_record in result_text.split(RECORD_DELIMITER):
        record = re.sub(r"^\(|\)$", "", raw_record.strip())
        if not record or record == COMPLETION_DELIMITER:
            continue
        record = re.sub(r"<\|(?!>)(?!COMPLETE\|>)", TUPLE_DELIMITER, record)
        parts = record.split(TUPLE_DELIMITER)
        record_type = parts[0].strip().strip('"').strip("'").lower() if parts else ""
        if record_type == "entity" and len(parts) >= 4:
            name = clean_str(_strip_placeholder_brackets(parts[1]).upper())
            etype = clean_str(_strip_placeholder_brackets(parts[2]).upper())
            desc = clean_str(_strip_placeholder_brackets(parts[3]))
            if name and etype and "|" not in name and "|" not in etype and name in new_endpoints_set:
                rows.append({"title": name, "type": etype, "description": desc, "source_id": source_id})

    return pd.DataFrame(rows) if rows else None


async def _classify_orphan_endpoints(
    raw_relationships: pd.DataFrame,
    entity_titles: set[str],
    model: "LLMCompletion",
    entity_types: list[str],
    fallback_type: str | None,
) -> list[dict]:
    """Classify relationship endpoints that were never declared as entities.

    Returns raw entity rows (title, type, description, source_id) ready for
    _merge_entities.  When *fallback_type* is set (v4_clean_inferred), any
    orphan the model did not classify is minted with that type instead of being
    dropped.
    """
    if raw_relationships.empty:
        return []

    sources = set(raw_relationships["source"].astype(str))
    targets = set(raw_relationships["target"].astype(str))
    orphan_names = sorted((sources | targets) - entity_titles)

    if not orphan_names:
        return []

    logger.info("Classifying %d orphan endpoints (variant=%s).", len(orphan_names), fallback_type or "drop")

    # Build per-orphan relationship snippets for context.
    snippets_by_name: dict[str, list[str]] = {name: [] for name in orphan_names}
    for _, row in raw_relationships.iterrows():
        src = str(row["source"])
        tgt = str(row["target"])
        desc = str(row["description"])
        if src in snippets_by_name and len(snippets_by_name[src]) < _MAX_SNIPPETS_PER_ORPHAN:
            snippets_by_name[src].append(desc)
        if tgt in snippets_by_name and len(snippets_by_name[tgt]) < _MAX_SNIPPETS_PER_ORPHAN:
            snippets_by_name[tgt].append(desc)

    entity_types_csv = ", ".join(entity_types)
    batches = [
        orphan_names[i: i + _ORPHAN_CLASSIFY_BATCH_SIZE]
        for i in range(0, len(orphan_names), _ORPHAN_CLASSIFY_BATCH_SIZE)
    ]

    async def classify_batch(batch: list[str]) -> dict[str, tuple[str, str]]:
        lines: list[str] = []
        for name in batch:
            lines.append(f"- {name}")
            lines.extend(f"  • {snip}" for snip in snippets_by_name[name])
        entity_list_str = "\n".join(lines)

        messages = (
            CompletionMessagesBuilder()
            .add_user_message(
                CLASSIFY_ORPHAN_ENDPOINTS_PROMPT.format(
                    entity_types=entity_types_csv,
                    entity_list=entity_list_str,
                )
            )
            .build()
        )
        try:
            response = await model.completion_async(messages=messages)
            result_text = response.content
        except PipelineHalted:
            raise
        except Exception:
            logger.exception("Error during orphan classification batch")
            return {}

        classified: dict[str, tuple[str, str]] = {}
        for raw_record in result_text.split(RECORD_DELIMITER):
            record = re.sub(r"^\(|\)$", "", raw_record.strip())
            if not record or record == COMPLETION_DELIMITER:
                continue
            record = re.sub(r"<\|(?!>)(?!COMPLETE\|>)", TUPLE_DELIMITER, record)
            parts = record.split(TUPLE_DELIMITER)
            if parts[0] == '"entity"' and len(parts) >= 4:
                name = clean_str(_strip_placeholder_brackets(parts[1]).upper())
                etype = clean_str(_strip_placeholder_brackets(parts[2]).upper())
                desc = clean_str(_strip_placeholder_brackets(parts[3]))
                if name and etype and "|" not in name and "|" not in etype:
                    classified[name] = (etype, desc)
        return classified

    batch_results = await asyncio.gather(*[classify_batch(b) for b in batches])

    classified_all: dict[str, tuple[str, str]] = {}
    for br in batch_results:
        classified_all.update(br)

    if fallback_type:
        for name in orphan_names:
            if name not in classified_all:
                classified_all[name] = (
                    fallback_type,
                    _fallback_orphan_description(snippets_by_name[name]),
                )

    # Build raw entity rows using source_ids from the relationships.
    entity_rows: list[dict] = []
    for name, (etype, desc) in classified_all.items():
        mask = (raw_relationships["source"] == name) | (raw_relationships["target"] == name)
        source_ids = raw_relationships.loc[mask, "source_id"].unique()
        if len(source_ids) > 0:
            entity_rows.extend(
                {"title": name, "type": etype, "description": desc, "source_id": sid}
                for sid in source_ids
            )
        else:
            entity_rows.append(
                {"title": name, "type": etype, "description": desc, "source_id": ""}
            )

    return entity_rows


def _fallback_orphan_description(snippets: list[str]) -> str:
    unique = list(dict.fromkeys(snippets))[:2]
    return "; ".join(unique) if unique else "Endpoint referenced in relationships."


def _format_entity_reference(entities: pd.DataFrame) -> str:
    """Format a sorted bare-title list for injection as the phase-2 salience context.

    Bare titles (no [TYPE]) prevent the model from copying type annotations into
    endpoint names, which would fragment the graph.
    """
    return "\n".join(f"- {t}" for t in sorted(entities["title"].astype(str)))


def _merge_entities(entity_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    all_entities = pd.concat(entity_dfs, ignore_index=True)
    # Drop blank source_id rows — orphan-minted entities with no chunk provenance
    # (phase-1 rows for the same title carry valid source_ids and are kept).
    all_entities = all_entities[all_entities["source_id"].astype(str).str.strip() != ""]
    return (
        all_entities
        .groupby(["title", "type"], sort=False)
        .agg(
            description=("description", list),
            text_unit_ids=("source_id", list),
            frequency=("source_id", "count"),
        )
        .reset_index()
    )


def _merge_relationships(relationship_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    all_relationships = pd.concat(relationship_dfs, ignore_index=True)
    return (
        all_relationships
        .groupby(["source", "target"], sort=False)
        .agg(
            description=("description", list),
            text_unit_ids=("source_id", list),
            weight=("weight", "sum"),
        )
        .reset_index()
    )


def _entity_embed_text(entities: pd.DataFrame, title: str) -> str:
    """`"TITLE: first description sentence"` for one merged entity row."""
    row = entities.loc[entities["title"] == title].iloc[0]
    desc = row["description"]
    if isinstance(desc, list):
        desc = next((str(d) for d in desc if d), "")
    return f"{title}: {desc}".strip(": ").strip()


def _phase2_checkpoint_dir(
    text_units: pd.DataFrame, id_column: str, config_fingerprint: str = ""
) -> Path:
    """Stable per-corpus temp dir for phase-2 per-chunk result checkpoints.

    Keyed by the sorted set of chunk IDs **plus** a config fingerprint so that
    changing entity types, variant, prompts, or extraction flags produces a new
    directory rather than silently reusing checkpoints from a different config.
    Delete the returned directory manually for a fully clean restart.
    """
    ids_key = "\n".join(sorted(str(i) for i in text_units[id_column]))
    corpus_hash = hashlib.md5(
        (ids_key + config_fingerprint).encode()
    ).hexdigest()[:12]  # nosec B324
    d = Path(tempfile.gettempdir()) / "graphrag_p2" / corpus_hash
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_pass_checkpoint(ckpt_dir: Path, pass_idx: int, ref_hash: str) -> dict:
    """Load all per-chunk checkpoint files for a phase-2 pass."""
    saved = {}
    for f in ckpt_dir.glob(f"pass_{pass_idx}_{ref_hash}_*.pkl"):
        try:
            with f.open("rb") as fh:
                chunk_id, result = pickle.load(fh)  # nosec B301
                saved[chunk_id] = result
        except Exception as exc:
            logger.warning(
                "Could not read phase-2 checkpoint %s (%s) — skipping chunk.",
                f, exc,
            )
    return saved


def _load_pass_checkpoint_all(ckpt_dir: Path, pass_idx: int) -> dict:
    """Load all per-chunk checkpoint files for a phase-2 pass regardless of hash."""
    saved = {}
    for f in ckpt_dir.glob(f"pass_{pass_idx}_*_*.pkl"):
        try:
            with f.open("rb") as fh:
                chunk_id, result = pickle.load(fh)  # nosec B301
                saved[chunk_id] = result
        except Exception as exc:
            logger.warning(
                "Could not read phase-2 checkpoint %s (%s) — skipping chunk.",
                f, exc,
            )
    return saved


def _save_chunk_checkpoint(
    ckpt_dir: Path, pass_idx: int, ref_hash: str, chunk_id: str, result: object
) -> None:
    """Write a single chunk result to its own file.

    Each chunk gets a unique filename, so concurrent threads never race on
    the same file — no locking required even under Windows.
    """
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", chunk_id)
    f = ckpt_dir / f"pass_{pass_idx}_{ref_hash}_{safe_id}.pkl"
    tmp = f.with_suffix(".pkl.tmp")
    with tmp.open("wb") as fh:
        pickle.dump((chunk_id, result), fh)
    tmp.replace(f)


async def _embed_texts(model: "LLMEmbedding", texts: list[str]) -> np.ndarray:
    """Embed strings -> L2-normalized float32 matrix (len(texts) x dim).

    Batches to respect provider request limits; normalizes so dot == cosine.
    For very large inputs, consider reusing run_embed_text (handles token-aware
    batching + long-text splitting).
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    batch, out = 256, []
    num_batches = (len(texts) + batch - 1) // batch
    for i in range(0, len(texts), batch):
        if num_batches > 1:
            batch_num = i // batch + 1
            logger.debug(
                "_embed_texts: batch %d/%d (%d texts)...",
                batch_num, num_batches, min(batch, len(texts) - i),
            )
        resp = await model.embedding_async(input=texts[i : i + batch])
        out.extend(resp.embeddings)
    vecs = np.asarray(out, dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def _select_relevant_entities(
    chunk_text: str,
    chunk_vec: np.ndarray,
    titles: list[str],
    entity_matrix: np.ndarray,  # rows aligned to `titles`, L2-normalized
    k: int,
) -> set[str]:
    """Hybrid: lexical-present titles, then nearest by cosine, capped at k."""
    lower = chunk_text.lower()
    lexical = [t for t in titles if t.lower() in lower]
    selected = list(lexical)
    lexical_set = set(lexical)
    if len(selected) < k:
        scores = entity_matrix @ chunk_vec
        # deterministic: descending score, then title (stable resume caching)
        order = sorted(
            range(len(titles)), key=lambda i: (-float(scores[i]), titles[i])
        )
        for i in order:
            if len(selected) >= k:
                break
            t = titles[i]
            if t not in lexical_set:
                selected.append(t)
    return set(selected)
