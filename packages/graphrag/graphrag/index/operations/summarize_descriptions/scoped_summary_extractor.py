# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Scope-aware description summarizer with per-document provenance and claims extraction."""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd
from graphrag_llm.utils import CompletionMessagesBuilder

from graphrag.data_model.schemas import DOCUMENT_IDS

if TYPE_CHECKING:
    from graphrag_llm.completion import LLMCompletion

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schemda constants for claim rows written to the covariates table
# ---------------------------------------------------------------------------
CLAIMS_COLUMNS = [
    "covariate_type",
    "type",
    "description",
    "subject_id",
    "object_id",
    "status",
    "start_date",
    "end_date",
    "source_text",
    "text_unit_id",
]

# ---------------------------------------------------------------------------
# Claim-line parsing patterns
# ---------------------------------------------------------------------------
_CLAIM_LINE = re.compile(r"^\s*(?:[-*]\s*)?CLAIM\s*\|", re.IGNORECASE)
_FRAGMENT_REF = re.compile(r"F(\d+)")
_COMPLETION_DELIMITER = "<|COMPLETE|>"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class ScopedFragment:
    """A single description fragment tagged with its source document."""

    text: str
    label: str  # sanitized document title
    text_unit_ids: list[str] = field(default_factory=list)


@dataclass
class ScopedClaim:
    """A scope-specific claim extracted from a summarization response."""

    scope: str
    statement: str
    source_text: str
    text_unit_ids: list[str] = field(default_factory=list)


@dataclass
class ScopedSummaryResult:
    """Result returned by the scoped extractor for one entity/relationship."""

    id: str | tuple[str, str]
    description: str
    claims: list[ScopedClaim] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Scope label sanitization
# ---------------------------------------------------------------------------
def sanitize_scope_label(title: str) -> str:
    """Make a document title safe for use as a fragment tag.

    Pipes in titles become slashes (pipes are the CLAIM-line delimiter).
    Runs of whitespace are collapsed to a single space.
    """
    return re.sub(r"\s+", " ", str(title).replace("|", "/")).strip()


# ---------------------------------------------------------------------------
# Document-id attachment
# ---------------------------------------------------------------------------
def _attach_document_ids(
    df: pd.DataFrame,
    text_unit_to_document: dict[str, str],
    kind: str,
) -> pd.DataFrame:
    """Add a document_ids column derived from each row's text_unit_ids.

    Parameters
    ----------
    df : pd.DataFrame
        Entity or relationship DataFrame that has a ``text_unit_ids`` column
        containing lists of text-unit id strings.
    text_unit_to_document : dict[str, str]
        Mapping from text-unit id → document title/id.
    kind : str
        Human-readable label for warning messages ("entity" or "relationship").
    """
    unresolved: set[str] = set()

    def to_documents(text_unit_ids: list[str] | None) -> list[str]:
        documents: set[str] = set()
        for tu in text_unit_ids or []:
            doc = text_unit_to_document.get(tu)
            if doc is None:
                unresolved.add(tu)
            else:
                documents.add(doc)
        return sorted(documents)

    df = df.copy()
    df[DOCUMENT_IDS] = df["text_unit_ids"].apply(to_documents)

    if unresolved:
        logger.warning(
            "scoped_facts: %d text unit id(s) on %s did not resolve to a document. "
            "Check that the text_units table has a document_id column.",
            len(unresolved),
            kind,
        )
    return df


# ---------------------------------------------------------------------------
# Fragment construction
# ---------------------------------------------------------------------------
def build_fragments(
    descriptions: list[str] | list[list[str]],
    text_unit_ids: list[list[str]] | None,
    text_unit_to_document: dict[str, str],
) -> list[ScopedFragment]:
    """Pair description strings with their source document label.

    *descriptions* may be a flat list of strings (one per occurrence) or a
    list of lists (after groupby aggregation).  When *text_unit_ids* is
    supplied, each description[i] is matched with text_unit_ids[i].
    Duplicate (text, label) pairs are de-duplicated.
    """
    fragments: list[ScopedFragment] = []
    seen: dict[tuple[str, str], int] = {}

    flat_descs: list[str] = []
    for d in descriptions:
        if isinstance(d, list):
            flat_descs.extend(d)
        else:
            flat_descs.append(d)

    flat_tus: list[list[str]] = []
    if text_unit_ids is not None:
        for t in text_unit_ids:
            if isinstance(t, list):
                flat_tus.append(t)
            else:
                flat_tus.append([t] if t else [])
    else:
        flat_tus = [[] for _ in flat_descs]

    if len(flat_descs) != len(flat_tus):
        msg = (
            "ALARM scoped_facts build_fragments alignment mismatch: "
            f"descriptions={len(flat_descs)} text_unit_groups={len(flat_tus)}. "
            "Stopping indexing to prevent provenance corruption."
        )
        logger.error(msg)
        raise ValueError(msg)

    for text, tu_ids in zip(flat_descs, flat_tus, strict=False):
        text = str(text).strip()
        if not text:
            continue
        normalized_tus = [str(tu).strip() for tu in tu_ids if str(tu).strip()]
        if not normalized_tus:
            msg = (
                "ALARM scoped_facts build_fragments found fragment with empty "
                "text_unit_ids after normalization. Stopping indexing because "
                "scope provenance cannot be trusted."
            )
            logger.error(msg)
            raise ValueError(msg)

        labels = [
            sanitize_scope_label(doc)
            for tu in normalized_tus
            if (doc := text_unit_to_document.get(tu))
        ]
        if not labels:
            sample = ", ".join(normalized_tus[:3])
            msg = (
                "ALARM scoped_facts unresolved text_unit_ids in build_fragments: "
                f"sample=[{sample}] (count={len(normalized_tus)}). "
                "Stopping indexing; check text_unit_id->document mapping."
            )
            logger.error(msg)
            raise ValueError(msg)
        label = labels[0]
        key = (text, label)
        if key not in seen:
            seen[key] = len(fragments)
            fragments.append(
                ScopedFragment(text=text, label=label, text_unit_ids=normalized_tus)
            )
        else:
            idx = seen[key]
            existing = fragments[idx].text_unit_ids
            for tu in normalized_tus:
                if tu not in existing:
                    existing.append(tu)

    # Keep deterministic ordering by scope label while preserving first-seen
    # order within each label bucket.
    indexed = list(enumerate(fragments))
    indexed.sort(key=lambda item: (item[1].label, item[0]))
    fragments = [fragment for _, fragment in indexed]

    return fragments


def render_fragment_list(fragments: list[ScopedFragment]) -> str:
    """Format fragments for insertion into the scoped summarization prompt."""
    return "\n".join(
        f"F{i + 1} [{fragment.label}] {fragment.text}"
        for i, fragment in enumerate(fragments)
    )


# ---------------------------------------------------------------------------
# Supporting-fragment resolver (used by _build_claims_derived)
# ---------------------------------------------------------------------------
def _supporting_fragments(
    fragments: list[ScopedFragment],
    refs: list[int],
) -> list[ScopedFragment]:
    """Resolve 1-based fragment refs; fall back to all fragments when none resolve."""
    resolved = [fragments[ref - 1] for ref in refs if 1 <= ref <= len(fragments)]
    return resolved if resolved else list(fragments)


# ---------------------------------------------------------------------------
# Claim-line parser
# ---------------------------------------------------------------------------
def _parse_claim_lines(
    response_text: str,
    fragments: list[ScopedFragment],
    entity_id: str | tuple[str, str],
    derive_scope: bool = False,
) -> list[ScopedClaim]:
    """Extract CLAIM lines from the LLM response and map them to ScopedClaim objects."""
    claims: list[ScopedClaim] = []
    entity_str = str(entity_id) if isinstance(entity_id, str) else f"{entity_id[0]}-{entity_id[1]}"

    for line in response_text.splitlines():
        if not _CLAIM_LINE.match(line):
            continue
        # Remove the leading "CLAIM |" marker
        body = re.sub(r"^\s*(?:[-*]\s*)?CLAIM\s*\|", "", line, flags=re.IGNORECASE).strip()
        parts = [p.strip() for p in body.split("|")]
        if len(parts) < 2:
            continue

        # Determine if last part is fragment-refs or statement
        last = parts[-1]
        has_refs = bool(_FRAGMENT_REF.search(last)) and len(parts) >= 3
        fragment_indices = [int(m) - 1 for m in _FRAGMENT_REF.findall(last)] if has_refs else []
        statement_parts = parts[1:-1] if has_refs else parts[1:]
        statement = " | ".join(p for p in statement_parts if p).strip()
        raw_scope = parts[0]

        if not statement:
            continue

        # Resolve scope: for derived mode, look up documents from fragment refs
        if derive_scope and raw_scope.upper() == "DERIVED" and fragment_indices:
            scope_docs: set[str] = set()
            for idx in fragment_indices:
                if 0 <= idx < len(fragments):
                    frag = fragments[idx]
                    for tu in frag.text_unit_ids:
                        scope_docs.add(frag.label)
            scope = sorted(scope_docs)[0] if scope_docs else raw_scope
        else:
            scope = raw_scope

        # Collect text_unit_ids from cited fragments
        tu_ids: list[str] = []
        for idx in fragment_indices:
            if 0 <= idx < len(fragments):
                tu_ids.extend(fragments[idx].text_unit_ids)

        source_text = f"{entity_str}: {statement}"
        claims.append(ScopedClaim(
            scope=scope,
            statement=statement,
            source_text=source_text,
            text_unit_ids=tu_ids,
        ))

    return claims


# ---------------------------------------------------------------------------
# Validation guards
# ---------------------------------------------------------------------------
def scope_matches_source(claim: ScopedClaim, fragments: list[ScopedFragment]) -> bool:
    """Check that a claim's scope label matches at least one supporting fragment's label."""
    if any(f.label == claim.scope for f in fragments):
        return True

    # Allow condition-scoped labels like:
    #   "Document A / speculative-grade"
    # by validating the document-prefix portion.
    if " / " in claim.scope:
        doc_scope = claim.scope.split(" / ", 1)[0].strip()
        return any(f.label == doc_scope for f in fragments)

    return False


def has_unsourced_acronym_expansion(claim: ScopedClaim, fragments: list[ScopedFragment]) -> bool:
    """Detect acronym expansions that never appear in any source fragment.

    Heuristic: if the statement looks like 'X stands for Y' or 'X (Y)' and
    the expansion Y does not appear in any fragment text, the claim is
    potentially hallucinated.
    """
    expansion_pattern = re.compile(
        r"\b(?:stands? for|is short for|means?|i\.?e\.?|a\.?k\.?a\.?)\b",
        re.IGNORECASE,
    )
    if not expansion_pattern.search(claim.statement):
        return False
    all_text = " ".join(f.text for f in fragments).lower()
    # Check if any significant word from the statement appears in sources
    words = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", claim.statement)
    for word in words:
        if word.lower() not in all_text:
            return True
    return False


# ---------------------------------------------------------------------------
# Main scoped extractor
# ---------------------------------------------------------------------------
class ScopedSummaryExtractor:
    """Scope-aware description summarizer.

    For each entity/relationship, groups description fragments by source
    document, produces a general portrait, and extracts scope-specific
    claims where descriptions differ across documents.
    """

    def __init__(
        self,
        model: "LLMCompletion",
        summarization_prompt: str,
        max_summary_length: int,
        max_input_tokens: int,
        max_tokens: int | None = None,
        context_relationships: int = 5,
        derive_scope: bool = False,
        on_error=None,
    ):
        self._model = model
        self._prompt = summarization_prompt
        self._max_summary_length = max_summary_length
        self._max_input_tokens = max_input_tokens
        self._max_tokens = max_tokens
        self._context_relationships = context_relationships
        self._derive_scope = derive_scope
        self._on_error = on_error or (lambda e, s, d: None)

    async def __call__(
        self,
        id: str | tuple[str, str],
        descriptions: list[str],
        text_unit_ids: list[list[str]] | None = None,
        text_unit_to_document: dict[str, str] | None = None,
        relationship_list: str = "",
    ) -> ScopedSummaryResult:
        """Summarize descriptions with scope awareness."""
        tu_to_doc = text_unit_to_document or {}

        # Build fragments
        fragments = build_fragments(
            descriptions=descriptions,
            text_unit_ids=text_unit_ids,
            text_unit_to_document=tu_to_doc,
        )

        # If single-document or no fragments: fall back to plain concatenation
        if not fragments:
            return ScopedSummaryResult(id=id, description=" ".join(str(d) for d in descriptions))

        fragment_list_str = render_fragment_list(fragments)
        entity_name = str(id) if isinstance(id, str) else f"{id[0]} -> {id[1]}"

        description, claims = await self._summarize_call(
            id=id,
            fragments=fragments,
            fragment_list_str=fragment_list_str,
            relationship_list=relationship_list,
            entity_name=entity_name,
        )

        return ScopedSummaryResult(id=id, description=description, claims=claims)

    def _build_claims_derived(
        self,
        claim_lines_text: str,
        fragments: list[ScopedFragment],
        entity_name: str,
    ) -> tuple[list[ScopedClaim], int]:
        """Parse claim lines in derived mode and assign document scope from fragment refs.

        The model emits only a condition (or nothing) — the document half of the scope
        is derived deterministically from the cited fragment labels. A claim that cites
        fragments spanning N documents is split into N ScopedClaim objects, one per
        document. scope_matches_source is NOT called here because the document was never
        authored by the model.
        """
        claims: list[ScopedClaim] = []
        rejected = 0
        for line in claim_lines_text.splitlines():
            if not _CLAIM_LINE.match(line):
                continue
            body = re.sub(r"^\s*(?:[-*]\s*)?CLAIM\s*\|", "", line, flags=re.IGNORECASE).strip()
            parts = [p.strip() for p in body.split("|")]
            if len(parts) < 2:
                rejected += 1
                continue
            last = parts[-1]
            has_refs = bool(_FRAGMENT_REF.search(last)) and len(parts) >= 3
            ref_nums = [int(m) for m in _FRAGMENT_REF.findall(last)] if has_refs else []
            statement_parts = parts[1:-1] if has_refs else parts[1:]
            statement = " | ".join(p for p in statement_parts if p).strip()
            condition = parts[0]  # empty string when model follows the empty-field contract

            if not statement:
                rejected += 1
                continue

            supporting = _supporting_fragments(fragments, ref_nums)
            temp_claim = ScopedClaim(
                scope="", statement=statement,
                source_text="".join(f.text for f in supporting),
                text_unit_ids=[],
            )
            if has_unsourced_acronym_expansion(temp_claim, supporting):
                rejected += 1
                continue

            # Split by document: one ScopedClaim per document that cited fragments come from.
            by_document: dict[str, list[ScopedFragment]] = {}
            for fragment in supporting:
                by_document.setdefault(fragment.label, []).append(fragment)

            for document in sorted(by_document):
                doc_fragments = by_document[document]
                scope = f"{document} / {condition}" if condition else document
                text_units = sorted({
                    tu for frag in doc_fragments for tu in frag.text_unit_ids
                })
                claims.append(ScopedClaim(
                    scope=scope,
                    statement=statement,
                    source_text="\n".join(f.text for f in doc_fragments),
                    text_unit_ids=text_units,
                ))

        return claims, rejected

    async def _summarize_call(
        self,
        id: str | tuple[str, str],
        fragments: list[ScopedFragment],
        fragment_list_str: str,
        relationship_list: str,
        entity_name: str,
    ) -> tuple[str, list[ScopedClaim]]:
        """Make one LLM call and parse the result."""
        messages = (
            CompletionMessagesBuilder()
            .add_user_message(
                self._prompt.format(
                    entity_name=entity_name,
                    fragment_list=fragment_list_str,
                    relationship_list=relationship_list or "(none)",
                    max_length=self._max_summary_length,
                )
            )
            .build()
        )
        try:
            call_kwargs: dict = {"messages": messages}
            if self._max_tokens is not None:
                call_kwargs["max_tokens"] = self._max_tokens
            _t0 = time.perf_counter()
            response = await self._model.completion_async(**call_kwargs)
            _elapsed_ms = int((time.perf_counter() - _t0) * 1000)
            result_text = response.content or ""
        except Exception as e:
            # Re-raise authentication errors — swallowing them produces garbage
            # summaries for every remaining entity. Crash loudly so the user can
            # fix the API key and re-run.
            if "AuthenticationError" in {t.__name__ for t in type(e).__mro__}:
                logger.error(
                    "Authentication failed — scoped summarization halted at id=%s. "
                    "Check the API key and re-run 'graphrag index'.",
                    id,
                )
                raise
            logger.exception("Error during scoped summarization for %s", id)
            self._on_error(e, "", {"id": id})
            return " ".join(f.text for f in fragments[:3]), []

        # Split portrait from CLAIM lines
        portrait_lines = []
        claim_lines = []
        for line in result_text.splitlines():
            if _CLAIM_LINE.match(line):
                claim_lines.append(line)
            elif line.strip() == _COMPLETION_DELIMITER:
                break
            else:
                portrait_lines.append(line)

        description = "\n".join(portrait_lines).strip()
        claim_text = "\n".join(claim_lines)

        if self._derive_scope:
            # Derived mode: document scope is assigned deterministically from
            # fragment refs; scope_matches_source is NOT called (would always fail
            # since the model never authors a document name).
            valid_claims, rejected = self._build_claims_derived(claim_text, fragments, entity_name)
            logger.info(
                "METRIC scoped_claim_validation id=%s kept=%d rejected=%d elapsed_ms=%d (derived)",
                id,
                len(valid_claims),
                rejected,
                _elapsed_ms,
            )
        else:
            claims = _parse_claim_lines(
                claim_text,
                fragments=fragments,
                entity_id=id,
                derive_scope=False,
            )
            invalid_scope = 0
            invalid_acronym = 0
            valid_claims = []
            for claim in claims:
                if not scope_matches_source(claim, fragments):
                    invalid_scope += 1
                    continue
                if has_unsourced_acronym_expansion(claim, fragments):
                    invalid_acronym += 1
                    continue
                valid_claims.append(claim)
            logger.info(
                "METRIC scoped_claim_validation id=%s parsed=%d kept=%d rejected_scope=%d rejected_acronym=%d elapsed_ms=%d",
                id,
                len(claims),
                len(valid_claims),
                invalid_scope,
                invalid_acronym,
                _elapsed_ms,
            )

        return description, valid_claims


# ---------------------------------------------------------------------------
# Claims → covariates row builder
# ---------------------------------------------------------------------------
def claims_to_covariate_rows(
    claims: list[ScopedClaim],
    subject_id: str,
    emit_claims: bool,
) -> list[dict]:
    """Convert ScopedClaim objects to covariate table rows."""
    if not emit_claims:
        return []
    rows = []
    for claim in claims:
        rows.append({
            "covariate_type": "scoped_fact",
            "type": claim.scope,
            "description": claim.statement,
            "subject_id": subject_id,
            "object_id": None,
            "status": "TRUE",
            "start_date": None,
            "end_date": None,
            "source_text": claim.source_text,
            "text_unit_id": ",".join(claim.text_unit_ids),
        })
    return rows
