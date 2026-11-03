# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Parameterization settings for the scoped facts configuration."""

from pydantic import BaseModel, Field

from graphrag.config.defaults import graphrag_config_defaults


class ScopedFactsConfig(BaseModel):
    """Configuration for document-provenance scoped facts extraction.

    When enabled, entity and relationship descriptions are merged with
    per-document scope awareness. Conflicting statements across documents
    are captured as claims (covariate_type='scoped_fact') in the
    covariates table instead of being silently averaged.
    """

    enabled: bool = Field(
        description="Enable scoped facts extraction.",
        default=False,
    )
    emit_claims: bool = Field(
        description=(
            "Write scope-conflict claims to the covariates table as "
            "covariate_type='scoped_fact'. Mutually exclusive with "
            "extract_claims.enabled (both write the covariates table)."
        ),
        default=True,
    )
    derive_scope: bool = Field(
        description=(
            "Use the derived-scope prompt variant: the model emits a condition "
            "plus fragment citations and the system assigns the document scope. "
            "Measured claims 233→428 vs base scoped mode."
        ),
        default=False,
    )
    context_relationships: int = Field(
        description="Number of adjacent relationships to include as context when summarizing entity descriptions.",
        default=5,
    )
    label_pattern: str | None = Field(
        description=(
            "Regex applied to each document title (via re.sub with label_replacement) "
            "to derive the scope label, e.g. '^([A-Z.-]+)_10k_(\\d{4}).*$' + '\\1 FY\\2' "
            "turns 'JPM_10k_2023.pdf' into 'JPM FY2023'. Non-matching titles pass through."
        ),
        default=graphrag_config_defaults.scoped_facts.label_pattern,
    )
    label_replacement: str = Field(
        description="Replacement template for label_pattern (re.sub syntax; backreferences allowed).",
        default=graphrag_config_defaults.scoped_facts.label_replacement,
    )
