# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Parameterization settings for entity alias resolution configuration."""

from pydantic import BaseModel, Field


class ResolveEntityAliasesConfig(BaseModel):
    """Configuration for deterministic entity alias resolution.

    Merges alias forms into canonical entity nodes using a hand-curated
    YAML mapping. Runs after extract_graph and before finalize_graph.
    No LLM calls — purely deterministic text substitution.

    Mapping file format:
        - canonical: "International Business Machines"
          aliases: ["IBM", "Big Blue"]
    """

    enabled: bool = Field(
        description="Enable entity alias resolution.",
        default=False,
    )
    mapping_file: str | None = Field(
        description=(
            "Path to the YAML aliases mapping file. Required when enabled=True. "
            "Each entry has a 'canonical' name and a list of 'aliases' to merge into it."
        ),
        default=None,
    )
