# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""List of exception names to skip for retries."""

_default_exceptions_to_skip = [
    "BadRequestError",
    "UnsupportedParamsError",
    "ContextWindowExceededError",
    "ContentPolicyViolationError",
    "ImageFetchError",
    "InvalidRequestError",
    "AuthenticationError",
    "PermissionDeniedError",
    "NotFoundError",
    "UnprocessableEntityError",
    "APIConnectionError",
    "APIError",
    "ServiceUnavailableError",
    "APIResponseValidationError",
    "BudgetExceededError",
]


def should_skip_retry(exc: BaseException, exceptions_to_skip: list[str]) -> bool:
    """Return True when exception type (or parent type) is marked non-retriable."""
    names = {t.__name__ for t in type(exc).__mro__}
    return bool(names & set(exceptions_to_skip))
