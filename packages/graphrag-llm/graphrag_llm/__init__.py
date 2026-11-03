# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""GraphRAG LLM Package."""

import nest_asyncio2

nest_asyncio2.apply()  # noqa: RUF067

from graphrag_llm.circuit_breaker import CircuitBreaker, PipelineHalted, get_breaker

__all__ = ["CircuitBreaker", "PipelineHalted", "get_breaker"]
