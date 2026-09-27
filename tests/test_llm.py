"""Tests for LLM types and LLMClient."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from anonframework.core.llm import LLMClient
from anonframework.core.types.llm import BudgetExhaustedError, LLMConfig, LLMUsage

# ---------------------------------------------------------------------------
# LLMConfig tests
# ---------------------------------------------------------------------------


class TestLLMConfig:
    def test_construction(self) -> None:
        config = LLMConfig(
            model="gpt-4o-mini",
            api_base="http://localhost:8000",
            api_key="sk-test-key-123",
        )
        assert config.model == "gpt-4o-mini"
        assert config.api_base == "http://localhost:8000"
        assert config.api_key == "sk-test-key-123"

    def test_frozen(self) -> None:
        config = LLMConfig(model="m", api_base="b", api_key="k")
        with pytest.raises(AttributeError):
            config.model = "other"  # type: ignore[misc]

    def test_repr_masks_api_key(self) -> None:
        config = LLMConfig(model="m", api_base="b", api_key="sk-very-secret-key")
        r = repr(config)
        assert "sk-v..." in r
        assert "sk-very-secret-key" not in r

    def test_repr_masks_short_key(self) -> None:
        config = LLMConfig(model="m", api_base="b", api_key="abc")
        r = repr(config)
        assert "***" in r
        assert "abc" not in r


# ---------------------------------------------------------------------------
# LLMUsage tests
# ---------------------------------------------------------------------------


class TestLLMUsage:
    def test_defaults(self) -> None:
        usage = LLMUsage()
        assert usage.calls == 0
        assert usage.cost == 0.0

    def test_construction(self) -> None:
        usage = LLMUsage(calls=5, cost=0.05)
        assert usage.calls == 5
        assert usage.cost == 0.05

    def test_frozen(self) -> None:
        usage = LLMUsage()
        with pytest.raises(AttributeError):
            usage.calls = 1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# BudgetExhaustedError tests
# ---------------------------------------------------------------------------


class TestBudgetExhaustedError:
    def test_has_usage(self) -> None:
        usage = LLMUsage(calls=10, cost=0.75)
        exc = BudgetExhaustedError("limit reached", usage=usage)
        assert exc.usage == usage
        assert "limit reached" in str(exc)


# ---------------------------------------------------------------------------
# LLMClient tests
# ---------------------------------------------------------------------------


def _make_mock_response() -> MagicMock:
    """Create a mock litellm.ModelResponse with usage data."""
    resp = MagicMock()
    resp.usage = MagicMock()  # non-None usage satisfies the existence check
    return resp


class TestLLMClient:
    def _make_config(self, **overrides: object) -> LLMConfig:
        defaults = dict(model="gpt-4o-mini", api_base="http://localhost", api_key="sk-test")
        defaults.update(overrides)
        return LLMConfig(**defaults)  # type: ignore[arg-type]

    @patch("anonframework.core.llm.completion_cost", return_value=0.001)
    @patch("anonframework.core.llm.acompletion")
    async def test_complete_calls_litellm(
        self,
        mock_acompletion: AsyncMock,
        _mock_cost: MagicMock,
    ) -> None:
        mock_acompletion.return_value = _make_mock_response()
        client = LLMClient(self._make_config())
        messages = [{"role": "user", "content": "hello"}]

        await client.complete(messages, temperature=0.5)

        mock_acompletion.assert_awaited_once_with(
            model="gpt-4o-mini",
            messages=messages,
            api_base="http://localhost",
            api_key="sk-test",
            drop_params=True,
            temperature=0.5,
        )

    @patch("anonframework.core.llm.completion_cost", return_value=0.001)
    @patch("anonframework.core.llm.acompletion")
    async def test_complete_drops_unsupported_params_by_default(
        self,
        mock_acompletion: AsyncMock,
        _mock_cost: MagicMock,
    ) -> None:
        """By default litellm is told to drop provider-unsupported params.

        Regression test for a provider-specific ``top_p`` failure: a paper-
        faithful attacker sends ``top_p``; the locked model may reject it; the
        client must not crash. ``drop_params=True`` makes litellm drop the
        unsupported param instead of raising ``UnsupportedParamsError``.
        """
        mock_acompletion.return_value = _make_mock_response()
        client = LLMClient(self._make_config())

        await client.complete([{"role": "user", "content": "x"}], top_p=0.9)

        assert mock_acompletion.call_args.kwargs["drop_params"] is True
        # The optimizer's param is still forwarded; litellm decides per-model.
        assert mock_acompletion.call_args.kwargs["top_p"] == 0.9

    @patch("anonframework.core.llm.completion_cost", return_value=0.001)
    @patch("anonframework.core.llm.acompletion")
    async def test_complete_drop_params_overridable(
        self,
        mock_acompletion: AsyncMock,
        _mock_cost: MagicMock,
    ) -> None:
        """A caller may opt into strict behaviour with drop_params=False."""
        mock_acompletion.return_value = _make_mock_response()
        client = LLMClient(self._make_config())

        await client.complete([{"role": "user", "content": "x"}], drop_params=False)

        assert mock_acompletion.call_args.kwargs["drop_params"] is False

    @patch("anonframework.core.llm.completion_cost", return_value=0.001)
    @patch("anonframework.core.llm.acompletion")
    async def test_complete_strips_locked_kwargs(
        self,
        mock_acompletion: AsyncMock,
        _mock_cost: MagicMock,
    ) -> None:
        mock_acompletion.return_value = _make_mock_response()
        client = LLMClient(self._make_config())
        messages = [{"role": "user", "content": "hello"}]

        await client.complete(
            messages,
            model="other-model",
            api_base="other-base",
            api_key="other-key",
        )

        # Locked values should be used, not the overrides
        call_kwargs = mock_acompletion.call_args
        assert call_kwargs.kwargs["model"] == "gpt-4o-mini"
        assert call_kwargs.kwargs["api_base"] == "http://localhost"
        assert call_kwargs.kwargs["api_key"] == "sk-test"

    @patch("anonframework.core.llm.completion_cost", return_value=0.001)
    @patch("anonframework.core.llm.acompletion")
    async def test_complete_strips_endpoint_aliases(
        self,
        mock_acompletion: AsyncMock,
        _mock_cost: MagicMock,
    ) -> None:
        """``base_url`` and friends must not redirect the call or swap credentials.

        ``base_url`` is litellm's alias for ``api_base``; before this was
        stripped, a caller could point the request at its own host and the
        locked ``api_key`` was sent there in the Authorization header.
        """
        mock_acompletion.return_value = _make_mock_response()
        client = LLMClient(self._make_config())

        await client.complete(
            [{"role": "user", "content": "hello"}],
            base_url="http://attacker.invalid/v1",
            api_version="2020-01-01",
            custom_llm_provider="openai",
            model_list=[{"model_name": "x"}],
            extra_headers={"authorization": "Bearer sk-attacker"},
            headers={"authorization": "Bearer sk-attacker"},
        )

        call_kwargs = mock_acompletion.call_args.kwargs
        assert call_kwargs["api_base"] == "http://localhost"
        assert call_kwargs["api_key"] == "sk-test"
        for leaked in (
            "base_url",
            "api_version",
            "custom_llm_provider",
            "model_list",
            "extra_headers",
            "headers",
        ):
            assert leaked not in call_kwargs

    @patch("anonframework.core.llm.completion_cost", return_value=0.005)
    @patch("anonframework.core.llm.acompletion")
    async def test_usage_tracking(
        self,
        mock_acompletion: AsyncMock,
        _mock_cost: MagicMock,
    ) -> None:
        mock_acompletion.return_value = _make_mock_response()
        client = LLMClient(self._make_config())

        assert client.usage == LLMUsage()

        await client.complete([{"role": "user", "content": "a"}])
        u1 = client.usage
        assert u1.calls == 1
        assert u1.cost == pytest.approx(0.005)

        await client.complete([{"role": "user", "content": "b"}])
        u2 = client.usage
        assert u2.calls == 2
        assert u2.cost == pytest.approx(0.010)

    @patch("anonframework.core.llm.completion_cost", return_value=0.60)
    @patch("anonframework.core.llm.acompletion")
    async def test_budget_cost_cap(
        self,
        mock_acompletion: AsyncMock,
        _mock_cost: MagicMock,
    ) -> None:
        mock_acompletion.return_value = _make_mock_response()
        client = LLMClient(self._make_config(), cost_cap_usd=1.00)

        await client.complete([{"role": "user", "content": "a"}])
        assert client.usage.cost == pytest.approx(0.60)

        await client.complete([{"role": "user", "content": "b"}])
        assert client.usage.cost == pytest.approx(1.20)

        with pytest.raises(BudgetExhaustedError, match="Cost cap reached"):
            await client.complete([{"role": "user", "content": "c"}])

    @patch("anonframework.core.llm.completion_cost", return_value=1.00)
    @patch("anonframework.core.llm.acompletion")
    async def test_budget_exact_boundary(
        self,
        mock_acompletion: AsyncMock,
        _mock_cost: MagicMock,
    ) -> None:
        """Cost exactly equal to the cap triggers BudgetExhaustedError."""
        mock_acompletion.return_value = _make_mock_response()
        client = LLMClient(self._make_config(), cost_cap_usd=1.00)

        # First call: $0 < $1.00 → allowed, cost becomes $1.00
        await client.complete([{"role": "user", "content": "a"}])
        assert client.usage.cost == pytest.approx(1.00)

        # Second call: $1.00 >= $1.00 → blocked
        with pytest.raises(BudgetExhaustedError, match="Cost cap reached"):
            await client.complete([{"role": "user", "content": "b"}])

    @patch("anonframework.core.llm.completion_cost", return_value=0.001)
    @patch("anonframework.core.llm.acompletion")
    async def test_no_budget_limits(
        self,
        mock_acompletion: AsyncMock,
        _mock_cost: MagicMock,
    ) -> None:
        """When no limits are set, calls never raise BudgetExhaustedError."""
        mock_acompletion.return_value = _make_mock_response()
        client = LLMClient(self._make_config())

        for _ in range(10):
            await client.complete([{"role": "user", "content": "hello"}])

        assert client.usage.calls == 10

    @patch("anonframework.core.llm.completion_cost", return_value=0.001)
    @patch("anonframework.core.llm.acompletion")
    async def test_missing_usage_raises(
        self,
        mock_acompletion: AsyncMock,
        _mock_cost: MagicMock,
    ) -> None:
        """Responses without usage data must raise RuntimeError."""
        resp = MagicMock()
        resp.usage = None
        mock_acompletion.return_value = resp
        client = LLMClient(self._make_config())

        with pytest.raises(RuntimeError, match="LLM response missing usage data.*Budget tracking"):
            await client.complete([{"role": "user", "content": "a"}])


# ---------------------------------------------------------------------------
# Controller integration: LLM client passed to optimizer
# ---------------------------------------------------------------------------


class TestControllerLLMIntegration:
    """Verify the controller creates and passes LLM clients."""

    async def test_optimizer_receives_llm_client(self) -> None:
        """The optimizer can access self.llm after controller sets it."""
        from anonframework.core.controller import Controller, TargetFactory
        from anonframework.core.interfaces.security_claim import SecurityClaim
        from anonframework.core.types.security_domain import Scope

        from .conftest import EXTERNAL_TAG, StubOptimizer, StubTarget, StubTask

        captured_client: object = None

        class CapturingOptimizer(StubOptimizer):
            async def initialize(self, goal, controllables, observables, llm_client) -> None:
                await super().initialize(goal, controllables, observables, llm_client)
                nonlocal captured_client
                captured_client = self.llm

        config = LLMConfig(
            model="test-model",
            api_base="http://test",
            api_key="sk-test",
        )
        scope: Scope = frozenset({EXTERNAL_TAG})
        controller = Controller(
            scope=scope,
            optimizer_factory=lambda: CapturingOptimizer(done=True),
            target_factory=TargetFactory.singleton(StubTarget()),
            security_claim=SecurityClaim.from_tasks([StubTask()]),
            llm_config=config,
            task_cost_cap_usd=5.00,
        )
        await controller.run()

        assert captured_client is not None
        assert isinstance(captured_client, LLMClient)

    async def test_result_always_has_llm_usage(self) -> None:
        """RunResult and TaskResult always have LLMUsage (zero calls when unused)."""
        from anonframework.core.controller import Controller, TargetFactory
        from anonframework.core.interfaces.security_claim import SecurityClaim
        from anonframework.core.types.security_domain import Scope

        from .conftest import EXTERNAL_TAG, STUB_LLM_CONFIG, StubOptimizer, StubTarget, StubTask

        scope: Scope = frozenset({EXTERNAL_TAG})
        controller = Controller(
            scope=scope,
            optimizer_factory=lambda: StubOptimizer(done=True),
            target_factory=TargetFactory.singleton(StubTarget()),
            security_claim=SecurityClaim.from_tasks([StubTask()]),
            llm_config=STUB_LLM_CONFIG,
        )
        result = await controller.run()

        tr = result.task_results[0]
        assert tr.llm_usage.calls == 0
        assert tr.llm_usage.cost == 0.0
        assert tr.runs[0].llm_usage.calls == 0

    @patch("anonframework.core.llm.completion_cost", return_value=0.005)
    @patch("anonframework.core.llm.acompletion")
    async def test_result_tracks_llm_usage(
        self,
        mock_acompletion: AsyncMock,
        _mock_cost: MagicMock,
    ) -> None:
        """RunResult and TaskResult track LLM usage."""
        from anonframework.core.controller import Controller, TargetFactory
        from anonframework.core.interfaces.security_claim import SecurityClaim
        from anonframework.core.types.security_domain import Scope

        from .conftest import EXTERNAL_TAG, StubOptimizer, StubTarget, StubTask

        mock_acompletion.return_value = _make_mock_response()

        class LLMUsingOptimizer(StubOptimizer):
            async def on_event(self, event):
                from anonframework.core.types.events import (
                    ControllableInjection,
                    ControllablePreCallEvent,
                    RunEndEvent,
                    RunEndResponse,
                    RunStartEvent,
                )

                if isinstance(event, RunStartEvent):
                    return EventResponse(event=event)
                if isinstance(event, ControllablePreCallEvent):
                    await self.llm.complete([{"role": "user", "content": "attack"}])
                    return ControllableInjection(
                        event=event,
                        controllable=event.controllable,
                        value="attack",
                    )
                if isinstance(event, RunEndEvent):
                    return RunEndResponse(event=event, done=True)
                return EventResponse(event=event)

        from anonframework.core.types.event import EventResponse

        config = LLMConfig(
            model="test-model",
            api_base="http://test",
            api_key="sk-test",
        )
        scope: Scope = frozenset({EXTERNAL_TAG})
        controller = Controller(
            scope=scope,
            optimizer_factory=lambda: LLMUsingOptimizer(done=True),
            target_factory=TargetFactory.singleton(StubTarget()),
            security_claim=SecurityClaim.from_tasks([StubTask()]),
            llm_config=config,
        )
        result = await controller.run()

        tr = result.task_results[0]
        assert tr.llm_usage.calls == 1
        assert tr.llm_usage.cost == pytest.approx(0.005)

        run = tr.runs[0]
        assert run.llm_usage.calls == 1


# ---------------------------------------------------------------------------
# Budget exhaustion graceful handling
# ---------------------------------------------------------------------------


class TestBudgetExhaustionGraceful:
    """Verify budget exhaustion stops the task gracefully, not the whole run."""

    @patch("anonframework.core.llm.completion_cost", return_value=0.60)
    @patch("anonframework.core.llm.acompletion")
    async def test_budget_exhaustion_stops_task_not_evaluation(
        self,
        mock_acompletion: AsyncMock,
        _mock_cost: MagicMock,
    ) -> None:
        """When budget runs out mid-task, that task stops but results are returned."""
        from anonframework.core.controller import Controller, TargetFactory
        from anonframework.core.interfaces.security_claim import SecurityClaim
        from anonframework.core.types.event import EventResponse
        from anonframework.core.types.security_domain import Scope

        from .conftest import EXTERNAL_TAG, StubOptimizer, StubTarget, StubTask

        mock_acompletion.return_value = _make_mock_response()

        class LLMEveryRunOptimizer(StubOptimizer):
            """Uses the LLM on every controllable event, never signals done."""

            async def on_event(self, event):
                from anonframework.core.types.events import (
                    ControllableInjection,
                    ControllablePreCallEvent,
                    RunEndEvent,
                    RunEndResponse,
                    RunStartEvent,
                )

                if isinstance(event, RunStartEvent):
                    return EventResponse(event=event)
                if isinstance(event, ControllablePreCallEvent):
                    await self.llm.complete([{"role": "user", "content": "attack"}])
                    return ControllableInjection(
                        event=event,
                        controllable=event.controllable,
                        value="attack",
                    )
                if isinstance(event, RunEndEvent):
                    return RunEndResponse(event=event, done=False)
                return EventResponse(event=event)

        # Budget of $1.00, each call costs $0.60 → 1st call OK, 2nd call OK,
        # 3rd call blocked (cost $1.20 >= $1.00)
        config = LLMConfig(
            model="test-model",
            api_base="http://test",
            api_key="sk-test",
        )
        target = StubTarget()
        scope: Scope = frozenset({EXTERNAL_TAG})
        controller = Controller(
            scope=scope,
            optimizer_factory=lambda: LLMEveryRunOptimizer(done=False),
            target_factory=TargetFactory.singleton(target),
            security_claim=SecurityClaim.from_tasks([StubTask(success=False)]),
            llm_config=config,
            task_cost_cap_usd=1.00,
            max_runs_per_task=10,
        )
        result = await controller.run()

        # Should have results, not a crash
        tmr = result
        assert len(tmr.task_results) == 1
        tr = tmr.task_results[0]
        # 2 runs completed (calls 1 and 2), 3rd run hit budget and was aborted
        assert len(tr.runs) == 2
        assert tr.llm_usage.cost == pytest.approx(1.20)
        # Budget exhaustion should stop immediately — not continue looping
        assert target.run_count == 3  # 2 completed + 1 aborted

    @patch("anonframework.core.llm.completion_cost", return_value=1.00)
    @patch("anonframework.core.llm.acompletion")
    async def test_budget_exhaustion_on_first_run(
        self,
        mock_acompletion: AsyncMock,
        _mock_cost: MagicMock,
    ) -> None:
        """Budget exhaustion on the very first run still produces a TaskResult."""
        from anonframework.core.controller import Controller, TargetFactory
        from anonframework.core.interfaces.security_claim import SecurityClaim
        from anonframework.core.types.event import EventResponse
        from anonframework.core.types.security_domain import Scope

        from .conftest import EXTERNAL_TAG, StubOptimizer, StubTarget, StubTask

        mock_acompletion.return_value = _make_mock_response()

        class DoubleCallOptimizer(StubOptimizer):
            async def on_event(self, event):
                from anonframework.core.types.events import (
                    ControllableInjection,
                    ControllablePreCallEvent,
                    RunEndEvent,
                    RunEndResponse,
                    RunStartEvent,
                )

                if isinstance(event, RunStartEvent):
                    return EventResponse(event=event)
                if isinstance(event, ControllablePreCallEvent):
                    await self.llm.complete([{"role": "user", "content": "a"}])
                    # Second call will raise BudgetExhaustedError
                    await self.llm.complete([{"role": "user", "content": "b"}])
                    return ControllableInjection(
                        event=event,
                        controllable=event.controllable,
                        value="x",
                    )
                if isinstance(event, RunEndEvent):
                    return RunEndResponse(event=event, done=True)
                return EventResponse(event=event)

        config = LLMConfig(
            model="test-model",
            api_base="http://test",
            api_key="sk-test",
        )
        scope: Scope = frozenset({EXTERNAL_TAG})
        controller = Controller(
            scope=scope,
            optimizer_factory=lambda: DoubleCallOptimizer(done=True),
            target_factory=TargetFactory.singleton(StubTarget()),
            security_claim=SecurityClaim.from_tasks([StubTask()]),
            llm_config=config,
            task_cost_cap_usd=0.50,
        )
        result = await controller.run()

        tmr = result
        assert len(tmr.task_results) == 1
        tr = tmr.task_results[0]
        # First run aborted — no completed runs
        assert len(tr.runs) == 0
        assert tr.success is False
        assert tr.best_score.value == 0.0
        assert tr.best_evaluation.success is False
        assert "budget exhausted" in tr.best_evaluation.rationale.lower()

    @patch("anonframework.core.llm.completion_cost", return_value=0.60)
    @patch("anonframework.core.llm.acompletion")
    async def test_budget_exhaustion_continues_to_next_task(
        self,
        mock_acompletion: AsyncMock,
        _mock_cost: MagicMock,
    ) -> None:
        """After budget exhaustion on one task, the next task still runs."""
        from anonframework.core.controller import Controller, TargetFactory
        from anonframework.core.interfaces.security_claim import SecurityClaim
        from anonframework.core.types.event import EventResponse
        from anonframework.core.types.security_domain import Scope

        from .conftest import EXTERNAL_TAG, StubOptimizer, StubTarget, StubTask

        mock_acompletion.return_value = _make_mock_response()

        class LLMOnceOptimizer(StubOptimizer):
            async def on_event(self, event):
                from anonframework.core.types.events import (
                    ControllableInjection,
                    ControllablePreCallEvent,
                    RunEndEvent,
                    RunEndResponse,
                    RunStartEvent,
                )

                if isinstance(event, RunStartEvent):
                    return EventResponse(event=event)
                if isinstance(event, ControllablePreCallEvent):
                    await self.llm.complete([{"role": "user", "content": "attack"}])
                    return ControllableInjection(
                        event=event,
                        controllable=event.controllable,
                        value="attack",
                    )
                if isinstance(event, RunEndEvent):
                    return RunEndResponse(event=event, done=False)
                return EventResponse(event=event)

        # Budget $0.50, each call $0.60 → first run OK ($0.60),
        # second run blocked ($0.60 >= $0.50).
        # Budget resets per task (fresh LLMClient), so task 2 also gets one run.
        config = LLMConfig(
            model="test-model",
            api_base="http://test",
            api_key="sk-test",
        )
        task_a = StubTask(goal_text="Task A")
        task_b = StubTask(goal_text="Task B")
        scope: Scope = frozenset({EXTERNAL_TAG})
        controller = Controller(
            scope=scope,
            optimizer_factory=lambda: LLMOnceOptimizer(done=False),
            target_factory=TargetFactory.singleton(StubTarget()),
            security_claim=SecurityClaim.from_tasks([task_a, task_b]),
            llm_config=config,
            task_cost_cap_usd=0.50,
        )
        result = await controller.run()

        # Both tasks should have results
        tmr = result
        assert len(tmr.task_results) == 2
        # Each task got 1 completed run before budget stopped it
        assert len(tmr.task_results[0].runs) == 1
        assert len(tmr.task_results[1].runs) == 1


# ---------------------------------------------------------------------------
# Optimizer.llm property tests
# ---------------------------------------------------------------------------


class TestLLMClientNoop:
    """Tests for LLMClient._make_noop() — the zero-budget client for non-LLM optimizers."""

    def test_noop_is_llm_client(self) -> None:
        """The noop client is a real LLMClient instance."""
        client = LLMClient._make_noop()
        assert isinstance(client, LLMClient)

    def test_noop_has_zero_usage(self) -> None:
        """The noop client starts with zero usage."""
        client = LLMClient._make_noop()
        assert client.usage.calls == 0
        assert client.usage.cost == 0.0

    async def test_noop_raises_budget_exhausted(self) -> None:
        """Any complete() call immediately raises BudgetExhaustedError."""
        client = LLMClient._make_noop()
        with pytest.raises(BudgetExhaustedError):
            await client.complete([{"role": "user", "content": "hello"}])

    async def test_noop_controller_no_llm_configs(self) -> None:
        """Controller without llm_configs passes noop client to optimizer."""
        from anonframework.core.controller import Controller, TargetFactory
        from anonframework.core.interfaces.security_claim import SecurityClaim
        from anonframework.core.types.security_domain import Scope

        from .conftest import EXTERNAL_TAG, StubOptimizer, StubTarget, StubTask

        captured_client: object = None

        class CapturingOptimizer(StubOptimizer):
            async def initialize(self, goal, controllables, observables, llm_client) -> None:
                await super().initialize(goal, controllables, observables, llm_client)
                nonlocal captured_client
                captured_client = llm_client

        scope: Scope = frozenset({EXTERNAL_TAG})
        controller = Controller(
            scope=scope,
            optimizer_factory=lambda: CapturingOptimizer(done=True),
            target_factory=TargetFactory.singleton(StubTarget()),
            security_claim=SecurityClaim.from_tasks([StubTask()]),
            # No llm_configs
        )
        await controller.run()

        assert captured_client is not None
        assert isinstance(captured_client, LLMClient)
        # Calling complete should raise immediately
        with pytest.raises(BudgetExhaustedError):
            await captured_client.complete([{"role": "user", "content": "x"}])


class TestOptimizerLLMProperty:
    def test_llm_available_after_controller_sets_it(self) -> None:
        from .conftest import StubOptimizer

        config = LLMConfig(model="m", api_base="b", api_key="k")
        client = LLMClient(config)

        opt = StubOptimizer()
        opt._llm_client = client
        assert opt.llm is client
