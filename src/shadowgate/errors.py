"""Exception hierarchy. Everything shadowgate raises on purpose derives from ShadowgateError."""

from __future__ import annotations


class ShadowgateError(Exception):
    """Base class for all shadowgate errors."""


class ConfigError(ShadowgateError):
    """Invalid or inconsistent configuration (bad TOML, unknown backend, bad threshold ...)."""


class BackendError(ShadowgateError):
    """A model call failed after retries were exhausted, or failed non-retryably.

    ``retryable`` records whether the final underlying failure was of a transient kind
    (rate limit, overload, timeout, connection) so callers can decide whether to resume later.
    """

    def __init__(self, message: str, *, backend: str = "", status: int | None = None,
                 retryable: bool = False) -> None:
        super().__init__(message)
        self.backend = backend
        self.status = status
        self.retryable = retryable


class BudgetExceeded(ShadowgateError):
    """A run hit its configured spending cap; completed work is already in the ledger."""


class LedgerError(ShadowgateError):
    """The ledger file is missing, corrupt, or belongs to an incompatible schema version."""


class DatasetError(ShadowgateError):
    """A task file could not be parsed or contains invalid/duplicate tasks."""


class InsufficientData(ShadowgateError):
    """A statistic was requested on too few observations to be meaningful."""
