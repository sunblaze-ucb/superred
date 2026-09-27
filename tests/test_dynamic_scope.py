"""Tests for per-task dynamic scoping.

The Controller ``scope`` parameter accepts either a fixed ``Scope`` (the
classic behavior) or a ``ScopeResolver`` (``Callable[[Task], Scope]``) resolved
once per task.  These tests cover the resolver contract (called once per task,
receives the task, result is used), per-task gating of every optimizer-facing
surface, construction validation, the skip/error containment paths, and the new
``TaskResult.scope`` / ``TaskResult.read_only`` and
``ThreatModelResult.scope_label`` fields.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from anonframework.core.controller import Controller, TargetFactory, ThreatModelResult
from anonframework.core.interfaces.security_claim import SecurityClaim
from anonframework.core.interfaces.target import Target
from anonframework.core.interfaces.task import NotApplicable, Task
from anonframework.core.persistence import (
    SCHEMA_VERSION,
    ExperimentMeta,
    iter_task_dirs,
    load_manifest,
    load_result,
    load_task,
    plan_resume,
)
from anonframework.core.types.events import (
    ControllableInjection,
    ControllableNoInjection,
    ControllablePreCallEvent,
    RunEndEvent,
)
from anonframework.core.types.security_domain import Scope

from .conftest import (
    EXTERNAL_TAG,
    INTERNAL_TAG,
    ROOT_TAG,
    STUB_LLM_CONFIG,
    StubOptimizer,
    StubTarget,
    StubTask,
)
from .test_controller import (
    _CapturingOptimizer,
    _MultiControllableTarget,
    _ScopedScoresTask,
    _TwoChannelTarget,
)

EXTERNAL_SCOPE: Scope = frozenset({EXTERNAL_TAG})
INTERNAL_SCOPE: Scope = frozenset({INTERNAL_TAG})
ROOT_SCOPE: Scope = frozenset({ROOT_TAG})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _TaggedTask(StubTask):
    """A StubTask carrying a tag a resolver can dispatch on."""

    def __init__(self, goal_text: str, want: Scope) -> None:
        super().__init__(goal_text=goal_text)
        self.want = want


# ---------------------------------------------------------------------------
# Resolver contract: called once per task, receives the task, result is used
# ---------------------------------------------------------------------------


class TestResolverContract:
    async def test_resolver_called_once_per_task(self) -> None:
        """The resolver runs exactly once per task, not per run, not at init."""
        calls: list[Task[Target]] = []

        def resolver(task: Task[Target]) -> Scope:
            calls.append(task)
            return EXTERNAL_SCOPE

        a, b = StubTask(goal_text="a"), StubTask(goal_text="b")
        controller = Controller(
            scope=resolver,
            scope_label="dyn",
            # Many runs per task: the resolver must NOT be re-invoked per run.
            optimizer_factory=lambda: StubOptimizer(done=False),
            target_factory=TargetFactory.singleton(StubTarget()),
            security_claim=SecurityClaim.from_tasks([a, b]),
            llm_config=STUB_LLM_CONFIG,
            max_runs_per_task=4,
        )
        await controller.run()
        # Two tasks -> exactly two resolver calls (despite 4 runs each).
        assert len(calls) == 2

    async def test_resolver_receives_the_task(self) -> None:
        """The resolver is handed the very Task object it must scope."""
        seen: list[Task[Target]] = []

        def resolver(task: Task[Target]) -> Scope:
            seen.append(task)
            return EXTERNAL_SCOPE

        a, b = StubTask(goal_text="a"), StubTask(goal_text="b")
        controller = Controller(
            scope=resolver,
            scope_label="dyn",
            optimizer_factory=lambda: StubOptimizer(done=True),
            target_factory=TargetFactory.singleton(StubTarget()),
            security_claim=SecurityClaim.from_tasks([a, b]),
            llm_config=STUB_LLM_CONFIG,
        )
        await controller.run()
        # Same identities, in input order (results land in input order).
        assert seen == [a, b]

    async def test_resolver_result_is_used_not_a_default(self) -> None:
        """The resolved scope actually gates: a resolver returning INTERNAL
        lets an INTERNAL controllable through where a default/empty/EXTERNAL
        scope would have declined it."""
        optimizer = StubOptimizer(done=True)
        controller = Controller(
            scope=lambda _t: INTERNAL_SCOPE,
            scope_label="dyn",
            optimizer_factory=lambda: optimizer,
            target_factory=TargetFactory.singleton(StubTarget(tag=INTERNAL_TAG)),
            security_claim=SecurityClaim.from_tasks([StubTask()]),
            llm_config=STUB_LLM_CONFIG,
        )
        await controller.run()
        ctrl_events = [
            e for e in optimizer.events_received if isinstance(e, ControllablePreCallEvent)
        ]
        # INTERNAL controllable offered because the resolver returned INTERNAL,
        # not some EXTERNAL/empty default.
        assert len(ctrl_events) == 1


# ---------------------------------------------------------------------------
# Two tasks resolved to different scopes gate differently across all surfaces
# ---------------------------------------------------------------------------


def _dispatch_resolver(task: Task[Target]) -> Scope:
    """Resolve EXTERNAL or INTERNAL based on a _TaggedTask's declared want."""
    assert isinstance(task, _TaggedTask)
    return task.want


class TestPerTaskGatingDiffers:
    async def test_controllables_gated_per_task(self) -> None:
        """Two tasks, one EXTERNAL one INTERNAL: each optimizer only receives
        the controllable matching its own resolved scope."""
        ext_task = _TaggedTask("ext", EXTERNAL_SCOPE)
        int_task = _TaggedTask("int", INTERNAL_SCOPE)

        # Capture every optimizer the factory hands out.
        opts: list[_CapturingOptimizer] = []

        def factory() -> _CapturingOptimizer:
            o = _CapturingOptimizer()
            opts.append(o)
            return o

        controller = Controller(
            scope=_dispatch_resolver,
            scope_label="dyn",
            optimizer_factory=factory,
            target_factory=TargetFactory(create=_MultiControllableTarget),
            security_claim=SecurityClaim.from_tasks([ext_task, int_task]),
            llm_config=STUB_LLM_CONFIG,
        )
        await controller.run()
        assert len(opts) == 2
        # Tasks run concurrently; sort the captured optimizers by which tag
        # they saw rather than relying on order.
        names_by_opt = [sorted(c.name for c in o.received_controllables) for o in opts]
        assert ["external_input"] in names_by_opt
        assert ["internal_input"] in names_by_opt

    async def test_observables_gated_per_task(self) -> None:
        opts: list[_CapturingOptimizer] = []

        def factory() -> _CapturingOptimizer:
            o = _CapturingOptimizer()
            opts.append(o)
            return o

        controller = Controller(
            scope=_dispatch_resolver,
            scope_label="dyn",
            optimizer_factory=factory,
            target_factory=TargetFactory(create=_MultiControllableTarget),
            security_claim=SecurityClaim.from_tasks(
                [_TaggedTask("ext", EXTERNAL_SCOPE), _TaggedTask("int", INTERNAL_SCOPE)]
            ),
            llm_config=STUB_LLM_CONFIG,
        )
        await controller.run()
        obs_by_opt = [sorted(o.observable.name for o in c.received_observables) for c in opts]
        assert ["ext_obs"] in obs_by_opt
        assert ["int_obs"] in obs_by_opt

    async def test_trajectory_injection_vs_decline_gated_per_task(self) -> None:
        """The _TwoChannelTarget fires both an EXTERNAL and an INTERNAL event.
        The EXTERNAL task injects only into external_input (declines internal);
        the INTERNAL task injects only into internal_input (declines external)."""
        results_by_scope: dict[str, tuple[list[str], list[str]]] = {}

        controller = Controller(
            scope=_dispatch_resolver,
            scope_label="dyn",
            optimizer_factory=lambda: StubOptimizer(done=True),
            target_factory=TargetFactory(create=_TwoChannelTarget),
            security_claim=SecurityClaim.from_tasks(
                [_TaggedTask("ext", EXTERNAL_SCOPE), _TaggedTask("int", INTERNAL_SCOPE)]
            ),
            llm_config=STUB_LLM_CONFIG,
        )
        result = await controller.run()
        for tr in result.task_results:
            snap = tr.runs[0].trajectory.snapshot()
            injected = sorted(
                r.controllable.name for r in snap if isinstance(r, ControllableInjection)
            )
            declined = sorted(
                r.controllable.name for r in snap if isinstance(r, ControllableNoInjection)
            )
            # Key by the resolved write scope recorded on the result.
            key = sorted(t.name for t in tr.scope)[0]
            results_by_scope[key] = (injected, declined)

        assert results_by_scope["external"] == (["external_input"], ["internal_input"])
        assert results_by_scope["internal"] == (["internal_input"], ["external_input"])

    async def test_feedback_sub_scores_gated_per_task(self) -> None:
        """_ScopedScoresTask emits an EXTERNAL and an INTERNAL sub_score; each
        task's RunEndEvent feedback keeps only the in-scope one."""

        # _ScopedScoresTask supplies the scoped sub_scores; carry a `want` tag
        # so the resolver can dispatch on it.
        class _ScopedTagged(_ScopedScoresTask):
            def __init__(self, goal_text: str, want: Scope) -> None:
                super().__init__(goal_text=goal_text)
                self.want = want

        ext_task = _ScopedTagged("ext", EXTERNAL_SCOPE)
        int_task = _ScopedTagged("int", INTERNAL_SCOPE)

        def resolver(task: Task[Target]) -> Scope:
            return task.want  # type: ignore[attr-defined]

        controller = Controller(
            scope=resolver,
            scope_label="dyn",
            optimizer_factory=lambda: StubOptimizer(done=True),
            target_factory=TargetFactory(create=StubTarget),
            security_claim=SecurityClaim.from_tasks([ext_task, int_task]),
            llm_config=STUB_LLM_CONFIG,
        )
        result = await controller.run()
        keep_by_scope: dict[str, set[str]] = {}
        for tr in result.task_results:
            entries = tr.runs[0].trajectory.snapshot()
            fb = next(e for e in entries if isinstance(e, RunEndEvent) and e.evaluation is not None)
            key = sorted(t.name for t in tr.scope)[0]
            keep_by_scope[key] = set(fb.evaluation.sub_scores.keys())

        # The EXTERNAL task keeps external_asr, drops internal_leak; vice versa.
        assert keep_by_scope["external"] == {"external_asr"}
        assert keep_by_scope["internal"] == {"internal_leak"}


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


class TestDynamicScopeValidation:
    def test_callable_scope_without_label_raises(self) -> None:
        with pytest.raises(ValueError, match="scope_label is required"):
            Controller(
                scope=lambda _t: EXTERNAL_SCOPE,
                optimizer_factory=lambda: StubOptimizer(),
                target_factory=TargetFactory.singleton(StubTarget()),
                security_claim=SecurityClaim.from_tasks([StubTask()]),
                llm_config=STUB_LLM_CONFIG,
            )

    def test_callable_scope_with_blank_label_raises(self) -> None:
        """A whitespace-only label is treated as empty."""
        with pytest.raises(ValueError, match="scope_label is required"):
            Controller(
                scope=lambda _t: EXTERNAL_SCOPE,
                scope_label="   ",
                optimizer_factory=lambda: StubOptimizer(),
                target_factory=TargetFactory.singleton(StubTarget()),
                security_claim=SecurityClaim.from_tasks([StubTask()]),
                llm_config=STUB_LLM_CONFIG,
            )

    def test_static_scope_with_label_raises(self) -> None:
        with pytest.raises(ValueError, match="only valid when scope or read_only is a callable"):
            Controller(
                scope=EXTERNAL_SCOPE,
                scope_label="oops",
                optimizer_factory=lambda: StubOptimizer(),
                target_factory=TargetFactory.singleton(StubTarget()),
                security_claim=SecurityClaim.from_tasks([StubTask()]),
                llm_config=STUB_LLM_CONFIG,
            )

    def test_callable_scope_with_label_constructs(self) -> None:
        """The valid dynamic-mode construction does not raise."""
        Controller(
            scope=lambda _t: EXTERNAL_SCOPE,
            scope_label="dyn",
            optimizer_factory=lambda: StubOptimizer(),
            target_factory=TargetFactory.singleton(StubTarget()),
            security_claim=SecurityClaim.from_tasks([StubTask()]),
            llm_config=STUB_LLM_CONFIG,
        )


# ---------------------------------------------------------------------------
# Per-task error / skip containment
# ---------------------------------------------------------------------------


class TestResolverContainment:
    async def test_resolver_empty_scope_skips_task(self) -> None:
        """A resolver returning an empty scope (no visibility, default empty
        read_only) SKIPS that task; a sibling with a real scope still runs."""

        def resolver(task: Task[Target]) -> Scope:
            if task.goal.description == "skip":
                return frozenset()
            return EXTERNAL_SCOPE

        skip_me = StubTask(goal_text="skip")
        good = StubTask(goal_text="good")
        controller = Controller(
            scope=resolver,
            scope_label="dyn",
            optimizer_factory=lambda: StubOptimizer(done=True),
            target_factory=TargetFactory(create=StubTarget),
            security_claim=SecurityClaim.from_tasks([skip_me, good]),
            llm_config=STUB_LLM_CONFIG,
        )
        result = await controller.run()
        # Empty visibility -> skipped, not a contained error.
        assert result.skipped_tasks == [skip_me]
        # Sibling unaffected, runs normally.
        assert [tr.task for tr in result.task_results] == [good]
        assert result.task_results[0].stop_reason == "success"
        assert result.task_results[0].success is True

    async def test_resolver_generic_exception_is_contained_error(self) -> None:
        """A resolver raising a non-NotApplicable exception becomes a per-task
        error, not a NotApplicable skip and not a crash of the run."""

        def resolver(task: Task[Target]) -> Scope:
            if task.goal.description == "boom":
                raise RuntimeError("resolver exploded")
            return EXTERNAL_SCOPE

        controller = Controller(
            scope=resolver,
            scope_label="dyn",
            optimizer_factory=lambda: StubOptimizer(done=True),
            target_factory=TargetFactory(create=StubTarget),
            security_claim=SecurityClaim.from_tasks(
                [StubTask(goal_text="boom"), StubTask(goal_text="fine")]
            ),
            llm_config=STUB_LLM_CONFIG,
        )
        result = await controller.run()
        by_goal = {tr.task.goal.description: tr for tr in result.task_results}
        assert by_goal["boom"].stop_reason == "error"
        assert by_goal["boom"].error is not None
        assert "resolver exploded" in by_goal["boom"].error
        assert "RuntimeError" in by_goal["boom"].error
        assert by_goal["boom"].runs == []
        # Resolution failed before any scope was known, so the synthesized error
        # result records an empty scope (not a resolved one).
        assert by_goal["boom"].scope == frozenset()
        assert by_goal["boom"].read_only == frozenset()
        assert by_goal["fine"].stop_reason == "success"
        assert result.skipped_tasks == []

    async def test_resolved_scope_recorded_on_post_resolution_error(self) -> None:
        """When an error occurs AFTER the scope is resolved (here the target
        factory raises), the synthesized error TaskResult records the task's
        resolved scope and read_only, not empty. In dynamic mode the per-task
        detail file is the only on-disk record of a failed task's enforced
        scope, so this must not silently regress to empty."""

        def boom_create() -> Target:
            raise RuntimeError("target factory boom")

        controller = Controller(
            scope=lambda _t: EXTERNAL_SCOPE,
            read_only=lambda _t: INTERNAL_SCOPE,
            scope_label="dyn",
            optimizer_factory=lambda: StubOptimizer(done=True),
            target_factory=TargetFactory(create=boom_create),
            security_claim=SecurityClaim.from_tasks([StubTask()]),
            llm_config=STUB_LLM_CONFIG,
        )
        result = await controller.run()
        tr = result.task_results[0]
        assert tr.stop_reason == "error"
        assert tr.error is not None and "target factory boom" in tr.error
        # The resolved per-task scope is preserved on the error result.
        assert tr.scope == EXTERNAL_SCOPE
        assert tr.read_only == INTERNAL_SCOPE

    async def test_lone_scope_resolver_not_applicable_skips_task(self) -> None:
        """A lone scope resolver raising NotApplicable (with the default empty
        read_only) leaves the total visibility empty, so the task is SKIPPED.
        NotApplicable is equivalent to returning an empty set: either way no tag
        in any dimension means skip.  A sibling with a real scope still runs."""

        opt_out = StubTask(goal_text="opt_out")
        keep_me = StubTask(goal_text="keep")

        def resolver(task: Task[Target]) -> Scope:
            if task is opt_out:
                raise NotApplicable("scope resolver opts out")
            return EXTERNAL_SCOPE

        controller = Controller(
            scope=resolver,
            scope_label="dyn",
            optimizer_factory=lambda: StubOptimizer(done=True),
            target_factory=TargetFactory(create=StubTarget),
            security_claim=SecurityClaim.from_tasks([opt_out, keep_me]),
            llm_config=STUB_LLM_CONFIG,
        )
        result = await controller.run()
        assert result.skipped_tasks == [opt_out]
        assert [tr.task for tr in result.task_results] == [keep_me]
        assert result.task_results[0].stop_reason == "success"


# ---------------------------------------------------------------------------
# Result fields: TaskResult.scope/read_only and ThreatModelResult.scope_label
# ---------------------------------------------------------------------------


class TestResultScopeFields:
    async def test_dynamic_task_results_record_resolved_scope(self) -> None:
        """Each TaskResult.scope is the per-task resolved write scope, and
        TaskResult.read_only is the run-wide read_only set."""
        ro: Scope = frozenset({ROOT_TAG})

        controller = Controller(
            scope=_dispatch_resolver,
            scope_label="per-tool",
            read_only=ro,
            optimizer_factory=lambda: StubOptimizer(done=True),
            target_factory=TargetFactory(create=StubTarget),
            security_claim=SecurityClaim.from_tasks(
                [_TaggedTask("ext", EXTERNAL_SCOPE), _TaggedTask("int", INTERNAL_SCOPE)]
            ),
            llm_config=STUB_LLM_CONFIG,
        )
        result = await controller.run()
        scopes = {sorted(t.name for t in tr.scope)[0]: tr for tr in result.task_results}
        assert set(scopes) == {"external", "internal"}
        assert scopes["external"].scope == EXTERNAL_SCOPE
        assert scopes["internal"].scope == INTERNAL_SCOPE
        # read_only is the fixed run-wide value on every task.
        assert all(tr.read_only == ro for tr in result.task_results)

    async def test_dynamic_threat_model_result_scope_label_and_empty_scope(self) -> None:
        """In dynamic mode the run-level scope/read_only are empty and the
        scope_label carries the run identity."""
        controller = Controller(
            scope=lambda _t: EXTERNAL_SCOPE,
            scope_label="my-run",
            optimizer_factory=lambda: StubOptimizer(done=True),
            target_factory=TargetFactory.singleton(StubTarget()),
            security_claim=SecurityClaim.from_tasks([StubTask()]),
            llm_config=STUB_LLM_CONFIG,
        )
        result = await controller.run()
        assert result.scope_label == "my-run"
        assert result.scope == frozenset()
        assert result.read_only == frozenset()
        # The per-task truth still lives on the TaskResult.
        assert result.task_results[0].scope == EXTERNAL_SCOPE

    async def test_static_mode_task_scope_equals_controller_scope(self) -> None:
        """Static mode is unchanged: every TaskResult.scope equals the
        controller scope and scope_label is None."""
        controller = Controller(
            scope=EXTERNAL_SCOPE,
            optimizer_factory=lambda: StubOptimizer(done=True),
            target_factory=TargetFactory(create=StubTarget),
            security_claim=SecurityClaim.from_tasks(
                [StubTask(goal_text="a"), StubTask(goal_text="b")]
            ),
            llm_config=STUB_LLM_CONFIG,
        )
        result = await controller.run()
        assert result.scope_label is None
        assert result.scope == EXTERNAL_SCOPE
        assert all(tr.scope == EXTERNAL_SCOPE for tr in result.task_results)

    def test_task_result_scope_defaults_empty(self) -> None:
        """The two new fields default to empty frozensets."""
        from anonframework.core.controller import TaskResult
        from anonframework.core.types.evaluation import EvaluationResult, Score
        from anonframework.core.types.llm import LLMUsage

        tr = TaskResult(
            task=StubTask(),
            runs=[],
            best_score=Score(0.0),
            best_evaluation=EvaluationResult(success=False, primary_score=Score(0.0)),
            success=False,
            llm_usage=LLMUsage(),
            stop_reason="done",
        )
        assert tr.scope == frozenset()
        assert tr.read_only == frozenset()

    def test_threat_model_result_scope_label_defaults_none(self) -> None:
        tmr = ThreatModelResult(
            scope=EXTERNAL_SCOPE,
            read_only=frozenset(),
            llm_config=None,
            task_results=[],
        )
        assert tmr.scope_label is None


# ---------------------------------------------------------------------------
# Persistence in dynamic mode
# ---------------------------------------------------------------------------


class TestDynamicScopePersistence:
    """Persistence in dynamic-scope mode under the schema v4 layout.

    In v4 the on-disk identity of an experiment is ``{slug}-{hash8}``: the slug
    is a human label built from ``attacker__target__claim__model`` (no scope tag
    names, no ``scope_label``), and the ``scope_label`` is folded into the
    identity hash instead of the folder-name stem.  The run-level scope /
    read_only are empty in dynamic mode; each task's own resolved scope lives in
    that task's ``task.json``.  Re-running an identical Controller RESUMES the
    same experiment dir rather than raising ``FileExistsError``.
    """

    @staticmethod
    def _experiment_dir(root: Path) -> Path:
        """Return the single experiment dir written under ``root``."""
        dirs = [p for p in root.iterdir() if p.is_dir()]
        assert len(dirs) == 1, f"expected one experiment dir, got {[p.name for p in dirs]}"
        return dirs[0]

    async def test_experiment_dir_uses_slug_hash_stem_not_label(self, tmp_path: Path) -> None:
        """The experiment dir is ``{slug}-{hash8}``; the ``scope_label`` is NOT
        the folder-name stem (it feeds the identity hash instead), and the slug
        carries the attacker/target/claim/model, not the label or scope tags."""
        controller = Controller(
            scope=lambda _t: EXTERNAL_SCOPE,
            scope_label="my-label",
            optimizer_factory=lambda: StubOptimizer(done=True),
            target_factory=TargetFactory.singleton(StubTarget()),
            security_claim=SecurityClaim.from_tasks([StubTask()]),
            llm_config=STUB_LLM_CONFIG,
            results_dir=tmp_path,
            persist=True,
            report=False,
        )
        await controller.run()

        exp_dir = self._experiment_dir(tmp_path)
        # {slug}-{hash8}: slug of [a-z0-9._-], then a dash and 8 hex chars.
        assert re.fullmatch(r"[a-z0-9._-]+-[0-9a-f]{8}", exp_dir.name), exp_dir.name
        # The label is folded into the hash, never spelled out in the folder name.
        assert "my-label" not in exp_dir.name
        # The slug is attacker__target__claim__model; the model is present, the
        # scope tag name ("external") is not part of the slug.
        stem, _, hash8 = exp_dir.name.rpartition("-")
        assert stem == "optimizer__target__securityclaim__test-model"
        assert "external" not in stem
        # result.json is the completion marker; manifest.json is the index.
        assert (exp_dir / "result.json").exists()
        assert (exp_dir / "manifest.json").exists()

        # The scope_label is part of the measurement identity: changing it lands
        # in a different {hash8} (a different experiment dir).
        other = ExperimentMeta(
            attacker="optimizer",
            target="target",
            claim="SecurityClaim",
            model="test-model",
            scope_label="a-different-label",
        )
        assert other.identity_hash() != hash8

    async def test_claim_summary_records_label_and_empty_scope(self, tmp_path: Path) -> None:
        """The experiment block (in manifest.json and result.json) records the
        scope_label and, in dynamic mode, empty run-level scope / read_only."""
        controller = Controller(
            scope=lambda _t: EXTERNAL_SCOPE,
            scope_label="my-label",
            optimizer_factory=lambda: StubOptimizer(done=True),
            target_factory=TargetFactory.singleton(StubTarget()),
            security_claim=SecurityClaim.from_tasks([StubTask()]),
            llm_config=STUB_LLM_CONFIG,
            results_dir=tmp_path,
            persist=True,
            report=False,
        )
        await controller.run()

        exp_dir = self._experiment_dir(tmp_path)
        for payload in (load_manifest(exp_dir), load_result(exp_dir)):
            assert payload["schema_version"] == SCHEMA_VERSION
            exp = payload["experiment"]
            assert exp["scope_label"] == "my-label"
            assert exp["scope"] == []
            assert exp["read_only"] == []

    async def test_per_task_detail_records_own_resolved_scope(self, tmp_path: Path) -> None:
        """Each task's ``task.json`` records that task's OWN resolved scope, so
        two tasks at different scopes produce task files with different scope
        arrays (the run-level scope stays empty)."""
        controller = Controller(
            scope=_dispatch_resolver,
            scope_label="per-tool",
            optimizer_factory=lambda: StubOptimizer(done=True),
            target_factory=TargetFactory(create=StubTarget),
            security_claim=SecurityClaim.from_tasks(
                [_TaggedTask("ext", EXTERNAL_SCOPE), _TaggedTask("int", INTERNAL_SCOPE)]
            ),
            llm_config=STUB_LLM_CONFIG,
            results_dir=tmp_path,
            persist=True,
            report=False,
        )
        await controller.run()

        exp_dir = self._experiment_dir(tmp_path)
        task_dirs = iter_task_dirs(exp_dir)
        assert len(task_dirs) == 2
        scopes_recorded = sorted(load_task(td)["scope"] for td in task_dirs)
        # One task scoped to external, the other to internal.
        assert scopes_recorded == [["external"], ["internal"]]

    async def test_reused_identity_resumes_without_error(self, tmp_path: Path) -> None:
        """Re-running an identical Controller (same results_dir + same identity)
        RESUMES the same experiment dir instead of raising ``FileExistsError``:
        the already-succeeded task is KEPT (not rerun) and no second experiment
        dir is created."""
        target = StubTarget()

        def make() -> Controller:
            return Controller(
                scope=lambda _t: EXTERNAL_SCOPE,
                scope_label="dup",
                optimizer_factory=lambda: StubOptimizer(done=True),
                target_factory=TargetFactory.singleton(target),
                security_claim=SecurityClaim.from_tasks([StubTask()]),
                llm_config=STUB_LLM_CONFIG,
                results_dir=tmp_path,
                persist=True,
                report=False,
            )

        await make().run()
        assert target.run_count == 1
        exp_dir = self._experiment_dir(tmp_path)

        # The resume plan keeps the prior success; nothing is scheduled to rerun.
        plan = plan_resume(exp_dir, ["Test goal"], overwrite=False)
        assert plan.keep == frozenset({1})
        assert plan.rerun == frozenset()
        assert plan.is_fresh is False

        # Second run of the identical Controller must not raise and must resume:
        # the kept task is not re-executed (run_count stays 1) and there is still
        # exactly one experiment dir.
        await make().run()
        assert target.run_count == 1
        assert self._experiment_dir(tmp_path) == exp_dir


# ===========================================================================
# read_only RESOLVER
#
# Symmetric to the scope (write) resolver: ``read_only`` also accepts a
# ``Scope | ScopeResolver``.  A read_only resolver is called once per task and
# produces that task's visible-but-not-injectable tags.  Its ``NotApplicable``
# contributes an EMPTY read_only set; the task is skipped only when the write
# resolver ALSO raises ``NotApplicable``.  These tests mirror the write-resolver
# tests above and reuse the same fixtures.
# ===========================================================================


class _RWTask(StubTask):
    """A StubTask carrying separate write and read_only wants for resolvers."""

    def __init__(self, goal_text: str, write: Scope, read_only: Scope) -> None:
        super().__init__(goal_text=goal_text)
        self.write_want = write
        self.read_only_want = read_only


def _read_only_dispatch(task: Task[Target]) -> Scope:
    """Resolve read_only from a _TaggedTask's declared want (mirrors the write
    dispatcher ``_dispatch_resolver``)."""
    assert isinstance(task, _TaggedTask)
    return task.want


# ---------------------------------------------------------------------------
# read_only resolver gates a different read-only surface per task
# ---------------------------------------------------------------------------


class TestReadOnlyResolverGatingDiffers:
    async def test_read_only_surface_differs_per_task(self) -> None:
        """A read_only resolver re-presents a DIFFERENT visible-but-not-
        injectable controllable per task.  Both tasks keep EXTERNAL injectable
        (fixed write scope); the read_only resolver makes INTERNAL visible for
        one task and nothing extra for the other.  The distinction shows in the
        optimizer-facing FilteredTrajectory: the INTERNAL decline is visible
        only for the task whose read_only resolved to INTERNAL; for the other
        task INTERNAL is fully out-of-scope and hidden from the filtered view.
        In both cases EXTERNAL is the only injectable surface, and
        ``TaskResult.read_only`` equals the per-task resolved value."""
        # write scope is fixed EXTERNAL for both; read_only varies per task.
        ro_task = _TaggedTask("ro_internal", INTERNAL_SCOPE)
        no_ro_task = _TaggedTask("no_ro", frozenset())

        controller = Controller(
            scope=EXTERNAL_SCOPE,
            read_only=_read_only_dispatch,
            scope_label="dyn-ro",
            optimizer_factory=lambda: StubOptimizer(done=True),
            target_factory=TargetFactory(create=_TwoChannelTarget),
            security_claim=SecurityClaim.from_tasks([ro_task, no_ro_task]),
            llm_config=STUB_LLM_CONFIG,
        )
        result = await controller.run()
        by_goal = {tr.task.goal.description: tr for tr in result.task_results}

        # Per-task resolved read_only is recorded on the TaskResult.
        assert by_goal["ro_internal"].read_only == INTERNAL_SCOPE
        assert by_goal["no_ro"].read_only == frozenset()
        # Write scope is the same fixed EXTERNAL for both.
        assert by_goal["ro_internal"].scope == EXTERNAL_SCOPE
        assert by_goal["no_ro"].scope == EXTERNAL_SCOPE

        def injected_full(tr: object) -> list[str]:
            snap = tr.runs[0].trajectory.snapshot()  # type: ignore[attr-defined]
            return sorted(r.controllable.name for r in snap if isinstance(r, ControllableInjection))

        def filtered_declined(tr: object) -> list[str]:
            # The optimizer-facing view: read_only declines are visible here,
            # out-of-scope declines are not.
            snap = tr.runs[0].trajectory.filtered.snapshot()  # type: ignore[attr-defined]
            return sorted(
                r.controllable.name for r in snap if isinstance(r, ControllableNoInjection)
            )

        # Only EXTERNAL is injectable for both tasks (read_only is never
        # injectable).
        assert injected_full(by_goal["ro_internal"]) == ["external_input"]
        assert injected_full(by_goal["no_ro"]) == ["external_input"]
        # read_only=INTERNAL: the internal_input decline is re-presented in the
        # filtered (optimizer-visible) trajectory.
        assert filtered_declined(by_goal["ro_internal"]) == ["internal_input"]
        # read_only=empty: INTERNAL is out-of-scope, so its decline is hidden
        # from the optimizer's filtered view entirely.
        assert filtered_declined(by_goal["no_ro"]) == []


# ---------------------------------------------------------------------------
# Construction validation (read_only resolver mirrors scope resolver)
# ---------------------------------------------------------------------------


class TestReadOnlyResolverValidation:
    def test_callable_read_only_without_label_raises(self) -> None:
        """A callable read_only (with a fixed scope) still requires scope_label."""
        with pytest.raises(ValueError, match="scope_label is required"):
            Controller(
                scope=EXTERNAL_SCOPE,
                read_only=lambda _t: INTERNAL_SCOPE,
                optimizer_factory=lambda: StubOptimizer(),
                target_factory=TargetFactory.singleton(StubTarget()),
                security_claim=SecurityClaim.from_tasks([StubTask()]),
                llm_config=STUB_LLM_CONFIG,
            )

    def test_callable_read_only_with_label_constructs(self) -> None:
        """Fixed scope + callable read_only + scope_label constructs fine."""
        Controller(
            scope=EXTERNAL_SCOPE,
            read_only=lambda _t: INTERNAL_SCOPE,
            scope_label="dyn-ro",
            optimizer_factory=lambda: StubOptimizer(),
            target_factory=TargetFactory.singleton(StubTarget()),
            security_claim=SecurityClaim.from_tasks([StubTask()]),
            llm_config=STUB_LLM_CONFIG,
        )

    def test_static_scope_static_read_only_with_label_raises(self) -> None:
        """Two fixed scopes plus a scope_label is rejected: the label is only
        valid when at least one of scope / read_only is a callable resolver."""
        with pytest.raises(ValueError, match="only valid when scope or read_only is a callable"):
            Controller(
                scope=EXTERNAL_SCOPE,
                read_only=INTERNAL_SCOPE,
                scope_label="oops",
                optimizer_factory=lambda: StubOptimizer(),
                target_factory=TargetFactory.singleton(StubTarget()),
                security_claim=SecurityClaim.from_tasks([StubTask()]),
                llm_config=STUB_LLM_CONFIG,
            )


# ---------------------------------------------------------------------------
# Truth table rows driven through the read_only resolver
# ---------------------------------------------------------------------------


class TestReadOnlyResolverTruthTable:
    async def test_row1_both_resolvers_not_applicable_skips_task(self) -> None:
        """Row 1 (W=NA, R=NA): when BOTH the write and read_only resolvers raise
        NotApplicable for a task, it is skipped (lands in skipped_tasks); a
        sibling for which both resolve still runs."""
        skip_me = StubTask(goal_text="skip")
        keep_me = StubTask(goal_text="keep")

        def write_resolver(task: Task[Target]) -> Scope:
            if task is skip_me:
                raise NotApplicable("write opts out")
            return EXTERNAL_SCOPE

        def read_only_resolver(task: Task[Target]) -> Scope:
            if task is skip_me:
                raise NotApplicable("read_only opts out")
            return frozenset()

        controller = Controller(
            scope=write_resolver,
            read_only=read_only_resolver,
            scope_label="dyn",
            optimizer_factory=lambda: StubOptimizer(done=True),
            target_factory=TargetFactory(create=StubTarget),
            security_claim=SecurityClaim.from_tasks([skip_me, keep_me]),
            llm_config=STUB_LLM_CONFIG,
        )
        result = await controller.run()
        assert result.skipped_tasks == [skip_me]
        assert [tr.task for tr in result.task_results] == [keep_me]

    async def test_row2_write_na_read_only_nonempty_runs_with_empty_write(self) -> None:
        """Row 2 (W=NA, R=nonempty): the task RUNS with an empty write scope (no
        injectable controllable) but the read_only surface stays visible.  The
        EXTERNAL controllable event is therefore auto-declined, never injected,
        and TaskResult.scope is empty while read_only carries the resolved set."""
        optimizer = StubOptimizer(done=True)
        controller = Controller(
            scope=lambda _t: (_ for _ in ()).throw(NotApplicable("no write")),
            read_only=lambda _t: EXTERNAL_SCOPE,
            scope_label="dyn",
            optimizer_factory=lambda: optimizer,
            target_factory=TargetFactory.singleton(StubTarget(tag=EXTERNAL_TAG)),
            security_claim=SecurityClaim.from_tasks([StubTask()]),
            llm_config=STUB_LLM_CONFIG,
        )
        result = await controller.run()
        tr = result.task_results[0]
        assert tr.stop_reason == "success"
        assert tr.scope == frozenset()
        assert tr.read_only == EXTERNAL_SCOPE
        # Nothing is injectable, so the optimizer is never offered the event.
        offered = [e for e in optimizer.events_received if isinstance(e, ControllablePreCallEvent)]
        assert offered == []
        # The visible read_only event is recorded and auto-declined.
        snap = tr.runs[0].trajectory.snapshot()
        assert [r.controllable.name for r in snap if isinstance(r, ControllableInjection)] == []
        assert [r.controllable.name for r in snap if isinstance(r, ControllableNoInjection)] == [
            "user_input"
        ]

    async def test_row4_write_nonempty_read_only_na_runs_with_empty_read_only(self) -> None:
        """Row 4 (W=nonempty, R=NA): the task RUNS with the resolved write scope
        and an EMPTY read_only; TaskResult.read_only == frozenset() and the
        in-scope controllable is injected normally."""
        optimizer = StubOptimizer(done=True)
        controller = Controller(
            scope=lambda _t: EXTERNAL_SCOPE,
            read_only=lambda _t: (_ for _ in ()).throw(NotApplicable("no read_only")),
            scope_label="dyn",
            optimizer_factory=lambda: optimizer,
            target_factory=TargetFactory.singleton(StubTarget(tag=EXTERNAL_TAG)),
            security_claim=SecurityClaim.from_tasks([StubTask()]),
            llm_config=STUB_LLM_CONFIG,
        )
        result = await controller.run()
        tr = result.task_results[0]
        assert tr.stop_reason == "success"
        assert tr.scope == EXTERNAL_SCOPE
        assert tr.read_only == frozenset()
        # The write-scoped controllable was offered and injected.
        snap = tr.runs[0].trajectory.snapshot()
        assert [r.controllable.name for r in snap if isinstance(r, ControllableInjection)] == [
            "user_input"
        ]

    async def test_row3_write_na_read_only_empty_skips_task(self) -> None:
        """Row 3 (W=NA, R=empty -> visibility empty): no tag in either dimension,
        so the task is SKIPPED (lands in skipped_tasks); a SIBLING with a real
        write scope still completes."""
        skip_me = StubTask(goal_text="skip")
        good = StubTask(goal_text="good")

        def write_resolver(task: Task[Target]) -> Scope:
            if task is skip_me:
                raise NotApplicable("no write for skip_me")
            return EXTERNAL_SCOPE

        controller = Controller(
            scope=write_resolver,
            read_only=lambda _t: frozenset(),  # always empty read_only
            scope_label="dyn",
            optimizer_factory=lambda: StubOptimizer(done=True),
            target_factory=TargetFactory(create=StubTarget),
            security_claim=SecurityClaim.from_tasks([skip_me, good]),
            llm_config=STUB_LLM_CONFIG,
        )
        result = await controller.run()
        # skip_me: W=NA, R=empty -> visibility empty -> skipped.
        assert result.skipped_tasks == [skip_me]
        # good: W=EXTERNAL, R=empty -> runs.
        assert [tr.task for tr in result.task_results] == [good]
        assert result.task_results[0].stop_reason == "success"
        assert result.task_results[0].success is True

    async def test_row7_both_resolvers_empty_skips_task(self) -> None:
        """Row 7 (W=empty, R=empty -> visibility empty): no tag in either
        dimension, so the task is SKIPPED; a sibling whose write resolves
        non-empty still succeeds.  Returning an empty set is equivalent to
        raising NotApplicable."""
        skip_me = StubTask(goal_text="skip")
        good = StubTask(goal_text="good")

        def write_resolver(task: Task[Target]) -> Scope:
            if task is skip_me:
                return frozenset()
            return EXTERNAL_SCOPE

        controller = Controller(
            scope=write_resolver,
            read_only=lambda _t: frozenset(),  # always empty read_only
            scope_label="dyn",
            optimizer_factory=lambda: StubOptimizer(done=True),
            target_factory=TargetFactory(create=StubTarget),
            security_claim=SecurityClaim.from_tasks([skip_me, good]),
            llm_config=STUB_LLM_CONFIG,
        )
        result = await controller.run()
        # skip_me: visibility empty -> skipped.
        assert result.skipped_tasks == [skip_me]
        assert [tr.task for tr in result.task_results] == [good]
        assert result.task_results[0].stop_reason == "success"
        assert result.task_results[0].success is True


# ---------------------------------------------------------------------------
# Both scope AND read_only callable: both vary per task and both are recorded
# ---------------------------------------------------------------------------


class TestBothResolversCallable:
    async def test_both_resolvers_vary_and_are_recorded(self) -> None:
        """With BOTH scope and read_only as resolvers (and a scope_label), each
        task's write scope and read_only set are resolved independently and both
        land on the TaskResult.  Task A: write=EXTERNAL, read_only=INTERNAL;
        Task B: write=INTERNAL, read_only=EXTERNAL (mirror image)."""
        a = _RWTask("a", write=EXTERNAL_SCOPE, read_only=INTERNAL_SCOPE)
        b = _RWTask("b", write=INTERNAL_SCOPE, read_only=EXTERNAL_SCOPE)

        def write_resolver(task: Task[Target]) -> Scope:
            assert isinstance(task, _RWTask)
            return task.write_want

        def read_only_resolver(task: Task[Target]) -> Scope:
            assert isinstance(task, _RWTask)
            return task.read_only_want

        controller = Controller(
            scope=write_resolver,
            read_only=read_only_resolver,
            scope_label="dyn-both",
            optimizer_factory=lambda: StubOptimizer(done=True),
            target_factory=TargetFactory(create=_TwoChannelTarget),
            security_claim=SecurityClaim.from_tasks([a, b]),
            llm_config=STUB_LLM_CONFIG,
        )
        result = await controller.run()
        by_goal = {tr.task.goal.description: tr for tr in result.task_results}

        assert by_goal["a"].scope == EXTERNAL_SCOPE
        assert by_goal["a"].read_only == INTERNAL_SCOPE
        assert by_goal["b"].scope == INTERNAL_SCOPE
        assert by_goal["b"].read_only == EXTERNAL_SCOPE

        def surfaces(tr: object) -> tuple[list[str], list[str]]:
            snap = tr.runs[0].trajectory.snapshot()  # type: ignore[attr-defined]
            injected = sorted(
                r.controllable.name for r in snap if isinstance(r, ControllableInjection)
            )
            declined = sorted(
                r.controllable.name for r in snap if isinstance(r, ControllableNoInjection)
            )
            return injected, declined

        # Task A injects its write (external), declines its read_only (internal).
        assert surfaces(by_goal["a"]) == (["external_input"], ["internal_input"])
        # Task B is the mirror image.
        assert surfaces(by_goal["b"]) == (["internal_input"], ["external_input"])

    async def test_both_resolvers_run_level_scope_empty_label_recorded(self) -> None:
        """In dynamic mode (both resolvers) the run-level scope/read_only stay
        empty and the scope_label carries the run identity."""
        controller = Controller(
            scope=lambda _t: EXTERNAL_SCOPE,
            read_only=lambda _t: INTERNAL_SCOPE,
            scope_label="both-run",
            optimizer_factory=lambda: StubOptimizer(done=True),
            target_factory=TargetFactory.singleton(StubTarget()),
            security_claim=SecurityClaim.from_tasks([StubTask()]),
            llm_config=STUB_LLM_CONFIG,
        )
        result = await controller.run()
        assert result.scope_label == "both-run"
        assert result.scope == frozenset()
        assert result.read_only == frozenset()
        assert result.task_results[0].scope == EXTERNAL_SCOPE
        assert result.task_results[0].read_only == INTERNAL_SCOPE


# ---------------------------------------------------------------------------
# read_only resolver raising a non-NotApplicable exception -> contained error
# ---------------------------------------------------------------------------


class TestReadOnlyResolverContainment:
    async def test_read_only_generic_exception_is_contained_error(self) -> None:
        """A read_only resolver raising a non-NotApplicable exception becomes a
        per-task error (not a NotApplicable skip, not a run crash); a sibling
        whose read_only resolves cleanly still completes."""

        def read_only_resolver(task: Task[Target]) -> Scope:
            if task.goal.description == "boom":
                raise RuntimeError("read_only resolver exploded")
            return frozenset()

        controller = Controller(
            scope=lambda _t: EXTERNAL_SCOPE,
            read_only=read_only_resolver,
            scope_label="dyn",
            optimizer_factory=lambda: StubOptimizer(done=True),
            target_factory=TargetFactory(create=StubTarget),
            security_claim=SecurityClaim.from_tasks(
                [StubTask(goal_text="boom"), StubTask(goal_text="fine")]
            ),
            llm_config=STUB_LLM_CONFIG,
        )
        result = await controller.run()
        by_goal = {tr.task.goal.description: tr for tr in result.task_results}
        assert by_goal["boom"].stop_reason == "error"
        assert by_goal["boom"].error is not None
        assert "read_only resolver exploded" in by_goal["boom"].error
        assert "RuntimeError" in by_goal["boom"].error
        assert by_goal["boom"].runs == []
        assert by_goal["fine"].stop_reason == "success"
        assert result.skipped_tasks == []
