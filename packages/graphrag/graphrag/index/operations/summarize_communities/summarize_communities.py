# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing summarize_communities method definition."""

import hashlib
import logging
import pickle
import re
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from graphrag_llm.circuit_breaker import PipelineHalted
from graphrag_llm.tokenizer import Tokenizer

import graphrag.data_model.schemas as schemas
from graphrag.callbacks.noop_workflow_callbacks import NoopWorkflowCallbacks
from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
from graphrag.config.enums import AsyncType
from graphrag.index.operations.summarize_communities.community_reports_extractor import (
    CommunityReportsExtractor,
)
from graphrag.index.operations.summarize_communities.typing import (
    CommunityReport,
    CommunityReportsStrategy,
    Finding,
)
from graphrag.index.operations.summarize_communities.utils import (
    get_levels,
)
from graphrag.index.utils.derive_from_rows import derive_from_rows
from graphrag.logger.progress import progress_ticker

if TYPE_CHECKING:
    from graphrag_llm.completion import LLMCompletion

logger = logging.getLogger(__name__)


def _community_ckpt_dir(communities: pd.DataFrame) -> Path:
    """Stable temp dir keyed by sorted community IDs."""
    ids_key = "\n".join(sorted(str(i) for i in communities[schemas.COMMUNITY_ID]))
    corpus_hash = hashlib.md5(ids_key.encode()).hexdigest()[:12]  # nosec B324
    d = Path(tempfile.gettempdir()) / "graphrag_community_reports" / corpus_hash
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_community_checkpoints(ckpt_dir: Path) -> dict:
    """Load all previously-saved community report files into a {(community_id, level): report} dict."""
    saved = {}
    for f in ckpt_dir.glob("report_*.pkl"):
        try:
            with f.open("rb") as fh:
                key, report = pickle.load(fh)  # nosec B301
                saved[key] = report
        except Exception as exc:
            logger.warning("Could not read community checkpoint %s (%s) — skipping.", f, exc)
    return saved


def _save_community_checkpoint(
    ckpt_dir: Path, community_id: int | str, level: int, report: object
) -> None:
    """Write a single community report to its own file."""
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", str(community_id))
    f = ckpt_dir / f"report_l{level}_c{safe_id}.pkl"
    tmp = f.with_suffix(".pkl.tmp")
    with tmp.open("wb") as fh:
        pickle.dump(((community_id, level), report), fh)
    tmp.replace(f)


async def summarize_communities(
    nodes: pd.DataFrame,
    communities: pd.DataFrame,
    local_contexts,
    level_context_builder: Callable,
    callbacks: WorkflowCallbacks,
    model: "LLMCompletion",
    prompt: str,
    tokenizer: Tokenizer,
    max_input_length: int,
    max_report_length: int,
    num_threads: int,
    async_type: AsyncType,
):
    """Generate community summaries."""
    ckpt_dir = _community_ckpt_dir(communities)
    ckpt_saved = _load_community_checkpoints(ckpt_dir)

    if ckpt_saved:
        logger.info(
            "Community reports: checkpoint loaded — %d reports already done.",
            len(ckpt_saved),
        )
    else:
        logger.info(
            "Community reports: checkpoints will be saved to %s", ckpt_dir,
        )

    reports: list[CommunityReport | None] = list(ckpt_saved.values())
    tick = progress_ticker(callbacks.progress, len(local_contexts))
    community_hierarchy = (
        communities
        .explode("children")
        .rename({"children": "sub_community"}, axis=1)
        .loc[:, ["community", "level", "sub_community"]]
    ).dropna()

    levels = get_levels(nodes)

    level_contexts = []
    for level in levels:
        level_context = level_context_builder(
            pd.DataFrame(reports),
            community_hierarchy_df=community_hierarchy,
            local_context_df=local_contexts,
            level=level,
            tokenizer=tokenizer,
            max_context_tokens=max_input_length,
        )
        level_contexts.append(level_context)

    for i, level_context in enumerate(level_contexts):
        level = levels[i]

        # Filter to communities not yet checkpointed at this level.
        done_ids = {
            cid for (cid, lvl) in ckpt_saved if lvl == level
        }
        pending_context = level_context[
            ~level_context[schemas.COMMUNITY_ID].isin(done_ids)
        ]

        if len(pending_context) < len(level_context):
            logger.info(
                "Community reports level %s: %d/%d already done, %d pending.",
                level, len(done_ids), len(level_context), len(pending_context),
            )

        async def run_generate(
            record,
            _level=level,
            _ckpt_dir=ckpt_dir,
            _ckpt_saved=ckpt_saved,
        ):
            result = await _generate_report(
                run_extractor,
                community_id=record[schemas.COMMUNITY_ID],
                community_level=record[schemas.COMMUNITY_LEVEL],
                community_context=record[schemas.CONTEXT_STRING],
                model=model,
                extraction_prompt=prompt,
                max_report_length=max_report_length,
            )
            tick()
            if result is not None:
                key = (record[schemas.COMMUNITY_ID], _level)
                _ckpt_saved[key] = result
                _save_community_checkpoint(
                    _ckpt_dir, record[schemas.COMMUNITY_ID], _level, result
                )
            return result

        if pending_context.empty:
            logger.info(
                "Community reports level %s: all %d reports loaded from checkpoint.",
                level, len(done_ids),
            )
        else:
            local_reports = await derive_from_rows(
                pending_context,
                run_generate,
                callbacks=NoopWorkflowCallbacks(),
                num_threads=num_threads,
                async_type=async_type,
                progress_msg=f"level {level} summarize communities progress: ",
            )
            reports.extend([lr for lr in local_reports if lr is not None])

    return pd.DataFrame(reports)


async def _generate_report(
    runner: CommunityReportsStrategy,
    model: "LLMCompletion",
    extraction_prompt: str,
    community_id: int,
    community_level: int,
    community_context: str,
    max_report_length: int,
) -> CommunityReport | None:
    """Generate a report for a single community."""
    return await runner(
        community_id,
        community_context,
        community_level,
        model,
        extraction_prompt,
        max_report_length,
    )


async def run_extractor(
    community: str | int,
    input: str,
    level: int,
    model: "LLMCompletion",
    extraction_prompt: str,
    max_report_length: int,
) -> CommunityReport | None:
    """Run the graph intelligence entity extraction strategy."""
    extractor = CommunityReportsExtractor(
        model,
        extraction_prompt=extraction_prompt,
        max_report_length=max_report_length,
        on_error=lambda e, stack, _data: logger.error(
            "Community Report Extraction Error", exc_info=e, extra={"stack": stack}
        ),
    )

    try:
        results = await extractor(input)
        report = results.structured_output
        if report is None:
            logger.warning("No report found for community: %s", community)
            return None

        return CommunityReport(
            community=community,
            full_content=results.output,
            level=level,
            rank=report.rating,
            title=report.title,
            rating_explanation=report.rating_explanation,
            summary=report.summary,
            findings=[
                Finding(explanation=f.explanation, summary=f.summary)
                for f in report.findings
            ],
            full_content_json=report.model_dump_json(indent=4),
        )
    except PipelineHalted:
        raise
    except Exception:
        logger.exception("Error processing community: %s", community)
        return None
