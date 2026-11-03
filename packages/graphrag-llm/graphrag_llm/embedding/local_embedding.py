# Copyright (c) 2026.
"""Local embedding adapter for GraphRAG (uses fastembed / ONNX).

Registers a `local_embeddings` LLM embedding provider so settings.yaml can do:

    embedding_models:
      default_embedding_model:
        type: local_embeddings
        model_provider: local
        model: BAAI/bge-small-en-v1.5
        api_key: not-used

Runs entirely offline after first model download (~80 MB to fastembed's cache).
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

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class LocalFastEmbedEmbedding(LLMEmbedding):
    """Embedding implementation backed by fastembed (ONNX) for offline use."""

    _model = None  # cached fastembed.TextEmbedding instance per process

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
        # Allow either `model` or settings.yaml `model:` to specify model name.
        self._fastembed_model_name = (
            model_config.model or DEFAULT_MODEL
            if model_config.model not in (None, "", "local")
            else DEFAULT_MODEL
        )

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

    # --------------------------------------------------------------- internals

    def _get_model(self):
        if LocalFastEmbedEmbedding._model is None:
            # On corp networks with TLS interception, httpx (used by
            # huggingface_hub for model download) fails cert verification
            # because its default certifi bundle doesn't include the corp CA.
            # `truststore` injects the OS cert store into SSL globally, which
            # makes both httpx and requests pick up corp-signed certs.
            try:
                import ssl

                import truststore  # type: ignore

                truststore.inject_into_ssl()
            except ImportError:
                pass

            # Lazy import: keeps cold import time low and lets tooling load
            # without immediately pulling onnxruntime.
            from fastembed import TextEmbedding  # type: ignore

            LocalFastEmbedEmbedding._model = TextEmbedding(
                model_name=self._fastembed_model_name
            )
        return LocalFastEmbedEmbedding._model

    def _embed(self, inputs: list[str]) -> list[list[float]]:
        if not inputs:
            return []
        model = self._get_model()
        # fastembed returns a generator of numpy arrays
        vectors: list[list[float]] = []
        for vec in model.embed(inputs):
            vectors.append([float(x) for x in vec.tolist()])
        return vectors

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

    # ------------------------------------------------------------ LLMEmbedding

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

    # GraphRAG's threaded thread pool wraps `embedding`; iterator method below
    # is for symmetry with the LiteLLM impl.
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
