"""LLM access types for the optimizer in the review artifact.

These types define LLM access: which model, the credentials to reach
it, and cumulative usage tracking. Budgets are not part of access. The
attacker's per-task cost cap lives on the Controller
(``task_cost_cap_usd``); any other ``LLMClient`` user sets its own cap.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMConfig:
    """LLM access: which model and the credentials to reach it.

    Pure access, used for both the attacker (``Controller(llm_config=...)``)
    and any other LLM caller such as a judge. Budgets are not part of access:
    the attacker's per-task cost cap is ``Controller.task_cost_cap_usd``, and
    any other ``LLMClient`` user sets its own cap when it builds the client.

    Attributes:
        model: LiteLLM model identifier (e.g. ``"gpt-4o-mini"``).
        api_base: LiteLLM-compatible API base URL.
        api_key: API key for the LLM provider.
    """

    model: str
    api_base: str
    api_key: str

    def __repr__(self) -> str:
        """Mask api_key in repr to avoid leaking secrets."""
        masked = self.api_key[:4] + "..." if len(self.api_key) > 4 else "***"
        return f"LLMConfig(model={self.model!r}, api_base={self.api_base!r}, api_key={masked!r})"


@dataclass(frozen=True)
class LLMUsage:
    """Cumulative LLM usage snapshot.

    Attributes:
        calls: Number of LLM calls made.
        cost: Total cost in USD, computed via ``litellm.completion_cost()``.
    """

    calls: int = 0
    cost: float = 0.0


class BudgetExhaustedError(Exception):
    """Raised when the optimizer exceeds its LLM budget.

    Attributes:
        usage: The cumulative usage at the time the budget was exhausted.
    """

    def __init__(self, message: str, usage: LLMUsage) -> None:
        super().__init__(message)
        self.usage = usage
