# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Global circuit breaker for graceful pipeline pause on fatal LLM errors."""

import logging
import threading

logger = logging.getLogger(__name__)

# Exception class NAMES matched against type(exc).__mro__ names.
# Mirrors the philosophy of retry/exceptions_to_skip.py — no litellm import.
_FATAL_IMMEDIATELY = {
    "AuthenticationError",  # expired/invalid key — never transient
    "PermissionDeniedError",
}
_FATAL_AFTER_CONSECUTIVE = {
    "APIConnectionError",
    "ServiceUnavailableError",
    "Timeout",
    "APITimeoutError",
    "InternalServerError",
}
_CONSECUTIVE_LIMIT = 5


class PipelineHalted(Exception):
    """Raised by the LLM layer when the circuit breaker has tripped.

    Must NEVER be swallowed into an empty per-item result — always re-raise.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(f"Pipeline halted: {reason}")
        self.reason = reason


class CircuitBreaker:
    """Process-wide latch. Once tripped, every LLM/embedding call fails fast."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tripped = threading.Event()
        self._reason = ""
        self._consecutive_connection_errors = 0

    @property
    def is_tripped(self) -> bool:
        return self._tripped.is_set()

    @property
    def reason(self) -> str:
        return self._reason

    def trip(self, reason: str) -> None:
        with self._lock:
            if not self._tripped.is_set():
                self._reason = reason
                self._tripped.set()
                logger.warning(
                    "Circuit breaker TRIPPED (%s) — draining in-flight work and pausing.",
                    reason,
                )

    def reset(self) -> None:
        """Called once at pipeline start."""
        with self._lock:
            self._tripped.clear()
            self._reason = ""
            self._consecutive_connection_errors = 0

    def record_success(self) -> None:
        with self._lock:
            self._consecutive_connection_errors = 0

    def record_failure(self, exc: BaseException) -> None:
        """Classify a post-retry failure; trip if fatal."""
        names = {t.__name__ for t in type(exc).__mro__}
        if names & _FATAL_IMMEDIATELY:
            self.trip(f"{type(exc).__name__}: {exc}")
            return
        if names & _FATAL_AFTER_CONSECUTIVE:
            with self._lock:
                self._consecutive_connection_errors += 1
                count = self._consecutive_connection_errors
            if count >= _CONSECUTIVE_LIMIT:
                self.trip(
                    f"{count} consecutive connection errors "
                    f"(last: {type(exc).__name__}: {exc})"
                )


_breaker = CircuitBreaker()


def get_breaker() -> CircuitBreaker:
    """Return the process-wide circuit breaker."""
    return _breaker
