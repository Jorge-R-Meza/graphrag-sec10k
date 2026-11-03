# Copyright (c) 2026.
"""Sentence-transformers embedding adapter for GraphRAG.

Registers a `sentence-transformers` embedding provider so settings.yaml can do:

    embedding_models:
      default_embedding_model:
        type: sentence-transformers
        model_provider: huggingface
        model: BAAI/bge-m3
        cell_args:
          device: cpu
          trust_remote_code: true

Runs entirely offline after first model download.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Unpack

from graphrag_llm.embedding.embedding import LLMEmbedding
from graphrag_llm.types import (
    LLMEmbedding as LLMEmbeddingItem,
    LLMEmbeddingResponse,
    LLMEmbeddingUsage,
)

if TYPE_CHECKING:
    from graphrag_cache import Cache, CacheKeyCreator

    from graphrag_llm.config import ModelConfig
    from graphrag_llm.metrics import MetricsProcessor, MetricsStore
    from graphrag_llm.rate_limit import RateLimiter
    from graphrag_llm.retry import Retry
    from graphrag_llm.tokenizer import Tokenizer
    from graphrag_llm.types import LLMEmbeddingArgs

DEFAULT_MODEL = "BAAI/bge-m3"


class SentenceTransformersEmbedding(LLMEmbedding):
    """Embedding implementation backed by sentence-transformers."""

    _model = None
    _current_model_name: str | None = None

    def __init__(
        self,
        *,
        model_id: str,
        model_config: "ModelConfig",
        tokenizer: "Tokenizer",
        metrics_store: "MetricsStore",
        metrics_processor: "MetricsProcessor | None" = None,
        rate_limiter: "RateLimiter | None" = None,
        retrier: "Retry | None" = None,
        cache: "Cache | None" = None,
        cache_key_creator: "CacheKeyCreator | None" = None,
        **kwargs: Any,
    ) -> None:
        from graphrag_llm.middleware import with_middleware_pipeline

        self._model_id = model_id
        self._model_config = model_config
        self._tokenizer = tokenizer
        self._metrics_store = metrics_store
        self._metrics_processor = metrics_processor
        self._st_model_name = (
            model_config.model or DEFAULT_MODEL
            if model_config.model not in (None, "", "local")
            else DEFAULT_MODEL
        )
        extra = model_config.model_extra or {}
        cell_args = extra.get("cell_args", {})
        self._device = cell_args.get("device", "cpu")
        self._trust_remote_code = cell_args.get("trust_remote_code", False)

        def _base_embedding(**kw: Any) -> "LLMEmbeddingResponse":
            inputs = kw.pop("input", [])
            kw.pop("metrics", None)
            if isinstance(inputs, str):
                inputs = [inputs]
            return self._build_response(self._embed(list(inputs)))

        async def _base_embedding_async(**kw: Any) -> "LLMEmbeddingResponse":
            import anyio
            return await anyio.to_thread.run_sync(lambda: _base_embedding(**kw))

        self._embedding, self._embedding_async = with_middleware_pipeline(
            model_config=self._model_config,
            model_fn=_base_embedding,
            async_model_fn=_base_embedding_async,
            request_type="embedding",
            cache=cache,
            cache_key_creator=cache_key_creator,
            tokenizer=self._tokenizer,
            metrics_processor=self._metrics_processor,
            rate_limiter=rate_limiter,
            retrier=retrier,
        )

    def _get_model(self):
        if (
            SentenceTransformersEmbedding._model is None
            or SentenceTransformersEmbedding._current_model_name != self._st_model_name
        ):
            try:
                import truststore
                truststore.inject_into_ssl()
            except ImportError:
                pass

            from sentence_transformers import SentenceTransformer

            SentenceTransformersEmbedding._model = SentenceTransformer(
                self._st_model_name,
                device=self._device,
                trust_remote_code=self._trust_remote_code,
            )
            SentenceTransformersEmbedding._current_model_name = self._st_model_name
        return SentenceTransformersEmbedding._model

    def _embed(self, inputs: list[str]) -> list[list[float]]:
        if not inputs:
            return []
        model = self._get_model()
        embeddings = model.encode(inputs, convert_to_numpy=True)
        return [vec.tolist() for vec in embeddings]

    def _build_response(self, vectors: list[list[float]]) -> LLMEmbeddingResponse:
        data = [
            LLMEmbeddingItem(object="embedding", embedding=vec, index=i)
            for i, vec in enumerate(vectors)
        ]
        return LLMEmbeddingResponse(
            object="list",
            data=data,
            model=self._model_id,
            usage=LLMEmbeddingUsage(prompt_tokens=0, total_tokens=0),
        )

    def embedding(
        self, /, **kwargs: Unpack["LLMEmbeddingArgs"]
    ) -> "LLMEmbeddingResponse":
        request_metrics = kwargs.pop("metrics", None) or {}
        if not self._metrics_processor:
            request_metrics = None
        return self._embedding(metrics=request_metrics, **kwargs)

    async def embedding_async(
        self, /, **kwargs: Unpack["LLMEmbeddingArgs"]
    ) -> "LLMEmbeddingResponse":
        request_metrics = kwargs.pop("metrics", None) or {}
        if not self._metrics_processor:
            request_metrics = None
        return await self._embedding_async(metrics=request_metrics, **kwargs)

    def embedding_iter(
        self, /, **kwargs: Unpack["LLMEmbeddingArgs"]
    ) -> Iterator["LLMEmbeddingResponse"]:
        yield self.embedding(**kwargs)

    @property
    def metrics_store(self) -> "MetricsStore":
        return self._metrics_store

    @property
    def tokenizer(self) -> "Tokenizer":
        return self._tokenizer
