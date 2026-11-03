# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Circuit breaker middleware — fail fast once the pipeline is halting."""

from typing import TYPE_CHECKING, Any

from graphrag_llm.circuit_breaker import PipelineHalted, get_breaker

if TYPE_CHECKING:
    from graphrag_llm.types import AsyncLLMFunction, LLMFunction


def with_circuit_breaker(
    *,
    sync_middleware: "LLMFunction",
    async_middleware: "AsyncLLMFunction",
) -> tuple["LLMFunction", "AsyncLLMFunction"]:
    """Fail fast before any network call once the breaker has tripped.

    Placed OUTSIDE the retrier (sees post-retry failures exactly once) and
    INSIDE the cache (cache hits are still served while tripped).
    """
    breaker = get_breaker()

    def _breaker_middleware(**kwargs: Any):
        if breaker.is_tripped:
            raise PipelineHalted(breaker.reason)
        try:
            result = sync_middleware(**kwargs)
        except PipelineHalted:
            raise
        except Exception as e:
            breaker.record_failure(e)
            raise
        else:
            breaker.record_success()
            return result

    async def _breaker_middleware_async(**kwargs: Any):
        if breaker.is_tripped:
            raise PipelineHalted(breaker.reason)
        try:
            result = await async_middleware(**kwargs)
        except PipelineHalted:
            raise
        except Exception as e:
            breaker.record_failure(e)
            raise
        else:
            breaker.record_success()
            return result

    return (_breaker_middleware, _breaker_middleware_async)  # type: ignore
