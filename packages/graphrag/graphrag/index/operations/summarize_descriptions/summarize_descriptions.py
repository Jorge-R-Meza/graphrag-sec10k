# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing the summarize_descriptions verb."""

import asyncio
import hashlib
import logging
import os
import pickle
import tempfile
from typing import TYPE_CHECKING

import pandas as pd

from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
from graphrag.index.operations.summarize_descriptions.description_summary_extractor import (
    SummarizeExtractor,
)
from graphrag.index.operations.summarize_descriptions.scoped_summary_extractor import (
    CLAIMS_COLUMNS,
    ScopedSummaryExtractor,
    _attach_document_ids,
    build_fragments,
    claims_to_covariate_rows,
    render_fragment_list,
)
from graphrag.index.operations.summarize_descriptions.typing import (
    SummarizedDescriptionResult,
)
from graphrag.logger.progress import ProgressTicker, progress_ticker

if TYPE_CHECKING:
    from graphrag_llm.completion import LLMCompletion

logger = logging.getLogger(__name__)


def _pair_descriptions_to_text_units(
    descriptions: list[str],
    text_unit_ids: list[str] | None,
    kind: str,
    row_id: str,
) -> tuple[list[str], list[list[str]] | None]:
    """Keep positional description->text unit alignment for scoped fragments."""
    if text_unit_ids is None:
        return descriptions, None

    if len(descriptions) != len(text_unit_ids):
        msg = (
            "ALARM scoped_facts alignment mismatch on "
            f"{kind} {row_id}: descriptions={len(descriptions)} "
            f"text_unit_ids={len(text_unit_ids)}. "
            "Stopping indexing to prevent provenance corruption."
        )
        logger.error(msg)
        raise ValueError(msg)

    if len(descriptions) == 0:
        msg = (
            f"ALARM scoped_facts empty descriptions for {kind} {row_id}. "
            "Stopping indexing because scoped summarization requires aligned inputs."
        )
        logger.error(msg)
        raise ValueError(msg)

    if len(text_unit_ids) == 0:
        msg = (
            f"ALARM scoped_facts empty text_unit_ids for {kind} {row_id}. "
            "Stopping indexing because chunk provenance is missing."
        )
        logger.error(msg)
        raise ValueError(msg)

    if any(not str(tid).strip() for tid in text_unit_ids):
        msg = (
            f"ALARM scoped_facts blank text_unit_id detected for {kind} {row_id}. "
            "Stopping indexing because chunk provenance is invalid."
        )
        logger.error(msg)
        raise ValueError(msg)

    n = len(descriptions)
    return descriptions[:n], [[str(text_unit_ids[i])] for i in range(n)]


async def summarize_descriptions(
    entities_df: pd.DataFrame,
    relationships_df: pd.DataFrame,
    callbacks: WorkflowCallbacks,
    model: "LLMCompletion",
    max_summary_length: int,
    max_input_tokens: int,
    prompt: str,
    num_threads: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize entity and relationship descriptions from an entity graph, using a language model."""

    async def get_summarized(
        nodes: pd.DataFrame, edges: pd.DataFrame, semaphore: asyncio.Semaphore
    ):
        ticker_length = len(nodes) + len(edges)

        ticker = progress_ticker(
            callbacks.progress,
            ticker_length,
            description="Summarize entity/relationship description progress: ",
        )

        node_futures = [
            do_summarize_descriptions(
                str(row.title),  # type: ignore
                sorted(set(row.description)),  # type: ignore
                ticker,
                semaphore,
            )
            for row in nodes.itertuples(index=False)
        ]

        node_results = await asyncio.gather(*node_futures)

        node_descriptions = [
            {
                "title": result.id,
                "description": result.description,
            }
            for result in node_results
        ]

        edge_futures = [
            do_summarize_descriptions(
                (str(row.source), str(row.target)),  # type: ignore
                sorted(set(row.description)),  # type: ignore
                ticker,
                semaphore,
            )
            for row in edges.itertuples(index=False)
        ]

        edge_results = await asyncio.gather(*edge_futures)

        edge_descriptions = [
            {
                "source": result.id[0],
                "target": result.id[1],
                "description": result.description,
            }
            for result in edge_results
        ]

        entity_descriptions = pd.DataFrame(node_descriptions)
        relationship_descriptions = pd.DataFrame(edge_descriptions)
        return entity_descriptions, relationship_descriptions

    async def do_summarize_descriptions(
        id: str | tuple[str, str],
        descriptions: list[str],
        ticker: ProgressTicker,
        semaphore: asyncio.Semaphore,
    ):
        async with semaphore:
            results = await run_summarize_descriptions(
                id,
                descriptions,
                model,
                max_summary_length,
                max_input_tokens,
                prompt,
            )
            ticker(1)
        return results

    semaphore = asyncio.Semaphore(num_threads)

    return await get_summarized(entities_df, relationships_df, semaphore)


async def run_summarize_descriptions(
    id: str | tuple[str, str],
    descriptions: list[str],
    model: "LLMCompletion",
    max_summary_length: int,
    max_input_tokens: int,
    prompt: str,
) -> SummarizedDescriptionResult:
    """Run the graph intelligence entity extraction strategy."""
    extractor = SummarizeExtractor(
        model=model,
        summarization_prompt=prompt,
        on_error=lambda e, stack, details: logger.error(
            "Entity Extraction Error",
            exc_info=e,
            extra={"stack": stack, "details": details},
        ),
        max_summary_length=max_summary_length,
        max_input_tokens=max_input_tokens,
    )

    result = await extractor(id=id, descriptions=descriptions)
    return SummarizedDescriptionResult(id=result.id, description=result.description)


async def summarize_descriptions_scoped(
    entities_df: pd.DataFrame,
    relationships_df: pd.DataFrame,
    callbacks: WorkflowCallbacks,
    model: "LLMCompletion",
    max_summary_length: int,
    max_input_tokens: int,
    prompt: str,
    num_threads: int,
    text_unit_to_document: dict[str, str],
    max_tokens: int | None = None,
    emit_claims: bool = True,
    derive_scope: bool = False,
    context_relationships: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Scope-aware description summarization with per-document claims extraction.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (summarized_entities, summarized_relationships, covariates)
    """
    extractor = ScopedSummaryExtractor(
        model=model,
        summarization_prompt=prompt,
        max_summary_length=max_summary_length,
        max_input_tokens=max_input_tokens,
        max_tokens=max_tokens,
        context_relationships=context_relationships,
        derive_scope=derive_scope,
        on_error=lambda e, stack, details: logger.error(
            "Scoped Summarization Error",
            exc_info=e,
            extra={"stack": stack, "details": details},
        ),
    )

    # --- Checkpoint setup ---
    # Fingerprint is a stable MD5 over entity titles, relationship pairs, and
    # the prompt text. Uses only stdlib MD5 (no Python hash(), which is
    # non-deterministic across process restarts due to hash-seed randomization).
    _fp_parts = (
        "|".join(sorted(str(t) for t in entities_df["title"])) + "\n"
        + "|".join(
            sorted(f"{r.source}->{r.target}" for r in relationships_df.itertuples(index=False))
        ) + "\n"
        + prompt
    )
    _ckpt_key = hashlib.md5(_fp_parts.encode()).hexdigest()[:12]
    _ckpt_dir = os.path.join(tempfile.gettempdir(), "graphrag_summ", _ckpt_key)
    os.makedirs(_ckpt_dir, exist_ok=True)
    logger.info(
        "Scoped summarization checkpoint dir: %s  (delete to force clean restart)",
        _ckpt_dir,
    )

    def _ckpt_path(item_id: str | tuple) -> str:
        key = str(item_id).encode()
        return os.path.join(_ckpt_dir, hashlib.md5(key).hexdigest() + ".pkl")

    def _load_ckpt(item_id: str | tuple):
        p = _ckpt_path(item_id)
        if os.path.exists(p):
            try:
                with open(p, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
        return None

    def _save_ckpt(item_id: str | tuple, result) -> None:
        try:
            with open(_ckpt_path(item_id), "wb") as f:
                pickle.dump(result, f)
        except Exception as exc:
            logger.warning("Failed to save summarization checkpoint for %s: %s", item_id, exc)

    semaphore = asyncio.Semaphore(num_threads)
    all_covariate_rows: list[dict] = []

    async def do_summarize_scoped(
        id: str | tuple[str, str],
        descriptions: list[str],
        text_unit_ids_per_desc: list[list[str]] | None,
        relationship_list: str,
        ticker: ProgressTicker,
    ):
        cached = _load_ckpt(id)
        if cached is not None:
            ticker(1)
            return cached
        async with semaphore:
            result = await extractor(
                id=id,
                descriptions=descriptions,
                text_unit_ids=text_unit_ids_per_desc,
                text_unit_to_document=text_unit_to_document,
                relationship_list=relationship_list,
            )
            _save_ckpt(id, result)
            ticker(1)
        return result

    ticker_length = len(entities_df) + len(relationships_df)
    ticker = progress_ticker(
        callbacks.progress,
        ticker_length,
        description="Scoped summarize entity/relationship description progress: ",
    )

    # Build a provenance-aware edge-context index: entity_title -> adjacent
    # relationship fragments, each labeled by source scope.
    edge_context: dict[str, list[str]] = {}
    if not relationships_df.empty:
        for _, row in relationships_df.iterrows():
            src = str(row.get("source", ""))
            tgt = str(row.get("target", ""))
            raw_desc = row.get("description", [])
            descs = [str(d) for d in raw_desc] if isinstance(raw_desc, (list, tuple)) else [str(raw_desc)]
            raw_tus = row.get("text_unit_ids", [])
            tus = (
                [str(t) for t in raw_tus]
                if isinstance(raw_tus, (list, tuple))
                else [str(raw_tus)]
            )
            edge_id = f"{src}->{tgt}"
            descs, tu_per_desc = _pair_descriptions_to_text_units(
                descs,
                tus,
                "relationship_context",
                edge_id,
            )
            edge_fragments = build_fragments(
                descriptions=descs,
                text_unit_ids=tu_per_desc,
                text_unit_to_document=text_unit_to_document,
            )
            context_lines = render_fragment_list(edge_fragments).splitlines()
            for node in (src, tgt):
                if node:
                    edge_context.setdefault(node, []).extend(context_lines)

    # --- Entity summarization ---
    node_futures = []
    for row in entities_df.itertuples(index=False):
        title = str(row.title)
        descs = [str(d) for d in row.description] if isinstance(row.description, (list, tuple)) else [str(row.description)]
        tus = [str(t) for t in row.text_unit_ids] if hasattr(row, "text_unit_ids") and isinstance(row.text_unit_ids, (list, tuple)) else None
        descs, tu_per_desc = _pair_descriptions_to_text_units(
            descs,
            tus,
            "entity",
            title,
        )
        rel_ctx = "\n".join(edge_context.get(title, [])[:context_relationships])
        node_futures.append(
            do_summarize_scoped(title, descs, tu_per_desc, rel_ctx, ticker)
        )

    node_results = await asyncio.gather(*node_futures)

    node_descriptions = []
    for result in node_results:
        node_descriptions.append({"title": result.id, "description": result.description})
        covariate_rows = claims_to_covariate_rows(result.claims, str(result.id), emit_claims)
        all_covariate_rows.extend(covariate_rows)

    # --- Relationship summarization ---
    edge_futures = []
    for row in relationships_df.itertuples(index=False):
        src = str(row.source)
        tgt = str(row.target)
        id_pair = (src, tgt)
        descs = [str(d) for d in row.description] if isinstance(row.description, (list, tuple)) else [str(row.description)]
        tus = [str(t) for t in row.text_unit_ids] if hasattr(row, "text_unit_ids") and isinstance(row.text_unit_ids, (list, tuple)) else None
        descs, tu_per_desc = _pair_descriptions_to_text_units(
            descs,
            tus,
            "relationship",
            f"{src}->{tgt}",
        )
        edge_futures.append(
            do_summarize_scoped(id_pair, descs, tu_per_desc, "", ticker)
        )

    edge_results = await asyncio.gather(*edge_futures)

    edge_descriptions = []
    for result in edge_results:
        edge_descriptions.append({
            "source": result.id[0],
            "target": result.id[1],
            "description": result.description,
        })
        covariate_rows = claims_to_covariate_rows(
            result.claims, f"{result.id[0]}->{result.id[1]}", emit_claims
        )
        all_covariate_rows.extend(covariate_rows)

    entity_descriptions = pd.DataFrame(node_descriptions)
    relationship_descriptions = pd.DataFrame(edge_descriptions)
    covariates = pd.DataFrame(all_covariate_rows, columns=CLAIMS_COLUMNS) if all_covariate_rows else pd.DataFrame(columns=CLAIMS_COLUMNS)

    return entity_descriptions, relationship_descriptions, covariates
