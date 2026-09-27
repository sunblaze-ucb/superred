"""LLM client: constrained async LLM access for optimizers.

The controller creates an :class:`LLMClient` from an :class:`LLMConfig`
and passes it to the optimizer. The client locks the model, API base,
and API key — the optimizer can only send messages and receive responses.

Budget enforcement is pre-call: the client checks cumulative cost
against its cost cap (``cost_cap_usd``, passed at construction) before
making each LLM call.

Uses litellm internally, so the response is a standard
``litellm.ModelResponse`` (OpenAI ``ChatCompletion`` format).
"""

from __future__ import annotations

import threading
from typing import Any, cast

from litellm import ModelResponse, acompletion, completion_cost

from anonframework.core.types.llm import BudgetExhaustedError, LLMConfig, LLMUsage

#: Keys a caller must never set on a completion. The point of ``LLMClient`` is
#: that the experiment, not the optimizer, decides which model is called, where,
#: and with whose credentials, so every litellm parameter that can redirect the
#: request or supply alternative credentials is stripped. ``base_url`` is
#: litellm's alias for ``api_base``: left in place, it sends the locked
#: ``api_key`` to a caller-chosen host.
_LOCKED_PARAMS = frozenset(
    {
        "api_base",
        "api_key",
        "api_version",
        "base_url",
        "custom_llm_provider",
        "extra_headers",
        "headers",
        "model",
        "model_list",
    }
)


class LLMClient:
    """Constrained LLM client for optimizer use.

    The model, API base, and API key are fixed at construction by the
    caller. The optimizer cannot change them.

    Thread-safe: usage tracking is protected by a lock.

    Args:
        config: The LLM access configuration (model + credentials).
        cost_cap_usd: Maximum cumulative cost in USD for this client;
            ``None`` (default) means unlimited. The controller passes the
            attacker's ``task_cost_cap_usd`` here (a fresh client per task,
            so it is a per-task cap); other callers pass their own cap.
    """

    def __init__(self, config: LLMConfig, cost_cap_usd: float | None = None) -> None:
        self._model = config.model
        self._api_base = config.api_base
        self._api_key = config.api_key
        self._cost_cap_usd = cost_cap_usd
        self._lock = threading.Lock()
        self._calls = 0
        self._cost = 0.0

    @classmethod
    def _make_noop(cls) -> LLMClient:
        """Create a zero-budget client for non-LLM optimizers.

        The client is a real ``LLMClient`` with ``cost_cap_usd=0`` so any
        ``complete()`` call immediately raises ``BudgetExhaustedError``.
        """
        return cls(
            LLMConfig(
                model="noop",
                api_base="http://noop",
                api_key="noop",
            ),
            cost_cap_usd=0,
        )

    async def complete(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> ModelResponse:
        """Send a chat completion request.

        Args:
            messages: OpenAI-format messages list.
            **kwargs: Additional parameters forwarded to litellm
                (e.g. ``temperature``, ``max_tokens``, ``stop``).
                Parameters that would redirect the request or supply other
                credentials cannot be overridden (see ``_LOCKED_PARAMS``).
                ``drop_params`` defaults to ``True`` (see below) but may be
                overridden by the caller.

        Returns:
            A ``litellm.ModelResponse`` (OpenAI ``ChatCompletion`` format).

        Raises:
            BudgetExhaustedError: If the cost budget has been reached.
        """
        # Strip keys that would escape the locked configuration.
        for locked in _LOCKED_PARAMS:
            kwargs.pop(locked, None)

        # Drop provider-unsupported sampling params instead of raising. An
        # optimizer is general-purpose: it does not know which model it is
        # pointed at, so a paper-faithful attacker that sends OpenAI-style
        # params (e.g. ``top_p``) must not crash when the locked model rejects
        # them. Anthropic Claude on AWS Bedrock, for instance, raises
        # ``UnsupportedParamsError`` on ``top_p``; with ``drop_params`` litellm
        # silently drops it and keeps it where it is supported. Caller may
        # override (pass ``drop_params=False``) to opt into strict behaviour.
        drop_params = kwargs.pop("drop_params", True)
        if "temperature" in kwargs and "top_p" in kwargs and "anthropic" in self._model.lower():
            kwargs.pop("temperature", None)

        if "temperature" in kwargs and "gpt-5" in self._model.lower():
            kwargs.pop("temperature", None)
        self._check_budget_pre_call()

        # acompletion handles every provider, including models litellm routes
        # through the Responses API (e.g. a model routed through Bedrock):
        # litellm (>=1.89.0) bridges chat<->responses internally and returns a
        # normal ModelResponse with usage, so we never branch on the model here.
        response = cast(
            ModelResponse,
            await acompletion(
                model=self._model,
                messages=messages,
                api_base=self._api_base,
                api_key=self._api_key,
                drop_params=drop_params,
                **kwargs,
            ),
        )

        self._record_usage(response)
        return response

    @property
    def usage(self) -> LLMUsage:
        """Current cumulative usage."""
        with self._lock:
            return LLMUsage(
                calls=self._calls,
                cost=self._cost,
            )

    def _check_budget_pre_call(self) -> None:
        """Raise BudgetExhaustedError if the cost cap has been reached."""
        if self._cost_cap_usd is None:
            return
        with self._lock:
            current = LLMUsage(
                calls=self._calls,
                cost=self._cost,
            )
        if current.cost >= self._cost_cap_usd:
            raise BudgetExhaustedError(
                f"Cost cap reached: ${current.cost:.6f}/${self._cost_cap_usd:.6f}",
                usage=current,
            )

    def _record_usage(self, response: ModelResponse) -> None:
        """Update cumulative usage from a response.

        Raises:
            RuntimeError: If the response does not include usage data.
                Usage tracking is essential for budget enforcement.
        """
        # usage is a dynamic extra field on litellm's ModelResponse (Pydantic extra="allow")
        usage = getattr(response, "usage", None)
        if usage is None:
            raise RuntimeError(
                f"LLM response missing usage data (model={self._model}). "
                "Budget tracking requires usage reporting from the provider."
            )
        call_cost = completion_cost(completion_response=response)

        # Lock: optimizers may issue concurrent LLM calls (parallel
        # consumption model), so _calls/_cost must be updated atomically.
        with self._lock:
            self._calls += 1
            self._cost += call_cost
