# Copyright (c) 2026.
"""Register a local (fastembed) embedding provider with graphrag-llm."""

from graphrag_llm.embedding import register_embedding

from ragtest.extensions.local_embed.local_embedding import LocalFastEmbedEmbedding

register_embedding("local_embeddings", LocalFastEmbedEmbedding)

__all__ = ["LocalFastEmbedEmbedding"]
