"""Extra unit tests for ``anonframework.core.persistence`` (schema v4).

Complements ``tests/test_persistence.py`` by exercising the paths that the
end-to-end tree tests do not reach: the individual serializer edges, the
atomic filesystem primitives (link/copy fallback, publish-over-existing),
the advisory experiment lock and its contention path, staging garbage
collection, the ``ExperimentSession`` per-task/finalize/abort surface, and
kept-task reconstruction.  These are driven directly (no Controller, no LLM)
with hand-built value objects.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from anonframework.core import persistence
from anonframework.core.controller import RunResult, TaskResult
from anonframework.core.persistence import (
    ExperimentMeta,
    ExperimentSession,
    _atomic_write_json,
    _duration_s,
    _ExperimentLock,
    _gc_staging,
    _json_fallback,
    _link_or_copy,
    _model_llm_config,
    _next_previous_index,
    _parse_iso,
    _publish_dir,
    _serialize_event,
    _serialize_llm_config,
    _serialize_response,
    _status_of,
    _task_dirname,
    goal_hash,
    iter_task_dirs,
    plan_resume,
    reconstruct_kept_task_result,
    snapshot_current,
)
from anonframework.core.types.controllable import Controllable
from anonframework.core.types.evaluation import EvaluationResult, Score
from anonframework.core.types.event import Event, EventResponse
from anonframework.core.types.events import (
    ControllableNoInjection,
    ControllablePostCallEvent,
    RunEndEvent,
    RunEndResponse,
)
from anonframework.core.types.llm import LLMUsage
from anonframework.core.types.trajectory import Trajectory

from .conftest import EXTERNAL_TAG, StubTask

BASE_META = ExperimentMeta(
    attacker="atk",
    target="tgt",
    claim="clm",
    model="test-model",
    scope=("external",),
)


# ---------------------------------------------------------------------------
# Builders (kept local so test_persistence.py stays untouched)
# ---------------------------------------------------------------------------


def _make_task_result(
    goal: str = "Test goal",
    *,
    score: float = 1.0,
    success: bool = True,
    stop_reason: str = "done",
    error: str | None = None,
) -> TaskResult:
    evaluation = EvaluationResult(success=success, primary_score=Score(value=score))
    started = datetime(2026, 1, 1, tzinfo=UTC)
    ended = datetime(2026, 1, 1, 0, 1, tzinfo=UTC)
    run = RunResult(
        trajectory=Trajectory(),
        evaluation=evaluation,
        llm_usage=LLMUsage(calls=1, cost=0.01),
        run_usage_delta=LLMUsage(calls=1, cost=0.01),
        started_at=started,
        ended_at=ended,
        evaluated=error is None,
        errored=error is not None,
        done=stop_reason == "done",
    )
    return TaskResult(
        task=StubTask(score=score, success=success, goal_text=goal),
        runs=[run],
        best_score=Score(value=score),
        best_evaluation=evaluation,
        success=success,
        llm_usage=LLMUsage(calls=1, cost=0.01),
        stop_reason=stop_reason,  # type: ignore[arg-type]
        scope=frozenset({EXTERNAL_TAG}),
        error=error,
        started_at=started,
        ended_at=ended,
    )


def _write_tree(root: Path, meta: ExperimentMeta, task_results: list[TaskResult]) -> Path:
    goals = [tr.task.goal.description for tr in task_results]
    session = ExperimentSession.open(root, meta, goals)
    for i, tr in enumerate(task_results, start=1):
        session.begin_task(i, tr.task.goal.description)
        session.publish_task(i, tr)
    session.finalize(datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, 0, 5, tzinfo=UTC))
    return root / meta.dirname()


# ---------------------------------------------------------------------------
# Serializer edges
# ---------------------------------------------------------------------------


def test_duration_s_none_when_either_endpoint_missing() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    assert _duration_s(None, now) is None
    assert _duration_s(now, None) is None
    assert _duration_s(None, None) is None
    later = datetime(2026, 1, 1, 0, 0, 5, tzinfo=UTC)
    assert _duration_s(now, later) == pytest.approx(5.0)


def test_serialize_llm_config_none() -> None:
    assert _serialize_llm_config(None) is None


def test_serialize_event_post_call_carries_answer() -> None:
    ctrl = Controllable(name="inp", security_domain=EXTERNAL_TAG)
    ev = ControllablePostCallEvent(controllable=ctrl, request="req", answer="ans")
    payload = _serialize_event(ev)
    assert payload["type"] == "ControllablePostCallEvent"
    assert payload["controllable"]["name"] == "inp"
    assert payload["request"] == "req"
    assert payload["answer"] == "ans"


def test_serialize_event_run_end_none_evaluation() -> None:
    ev = RunEndEvent(evaluation=None, security_domain=EXTERNAL_TAG)
    payload = _serialize_event(ev)
    assert payload["type"] == "RunEndEvent"
    assert payload["evaluation"] is None


def test_serialize_event_unhandled_type_is_base_payload_only() -> None:
    # A bare Event is none of the handled subclasses: the elif chain falls
    # straight through to the base payload (no type-specific keys).
    ev = Event(security_domain=EXTERNAL_TAG)
    payload = _serialize_event(ev)
    assert payload["kind"] == "event"
    assert payload["type"] == "Event"
    assert "evaluation" not in payload
    assert "controllable" not in payload


def test_serialize_response_no_injection() -> None:
    ctrl = Controllable(name="inp", security_domain=EXTERNAL_TAG)
    ev = ControllablePostCallEvent(controllable=ctrl, request="r", answer="a")
    resp = ControllableNoInjection(event=ev, controllable=ctrl)
    payload = _serialize_response(resp)
    assert payload["type"] == "ControllableNoInjection"
    assert payload["controllable"] == "inp"
    assert "value" not in payload


def test_serialize_response_run_end_carries_done() -> None:
    ev = RunEndEvent(evaluation=None, security_domain=EXTERNAL_TAG)
    payload = _serialize_response(RunEndResponse(event=ev, done=True))
    assert payload["type"] == "RunEndResponse"
    assert payload["done"] is True


def test_serialize_response_unhandled_type_is_base_only() -> None:
    ev = Event(security_domain=EXTERNAL_TAG)
    payload = _serialize_response(EventResponse(event=ev))
    assert payload == {"kind": "response", "type": "EventResponse"}


def test_status_of_error_and_budget_and_failed() -> None:
    assert _status_of(False, "error", 0) == "error"
    assert _status_of(False, "budget_exhausted", 2) == "budget_exhausted"
    assert _status_of(True, "done", 1) == "success"
    assert _status_of(False, "max_runs", 5) == "failed"


def test_model_llm_config_none_when_no_model() -> None:
    meta = ExperimentMeta(attacker="a", target="t", claim="c", model=None)
    assert _model_llm_config(meta) is None
    with_model = ExperimentMeta(attacker="a", target="t", claim="c", model="m")
    cfg = _model_llm_config(with_model)
    assert cfg is not None
    assert cfg.model == "m"
    assert cfg.api_key == ""


def test_json_fallback_repr_for_non_datetime() -> None:
    dt = datetime(2026, 1, 1, tzinfo=UTC)
    assert _json_fallback(dt) == dt.isoformat()
    sentinel = object()
    assert _json_fallback(sentinel) == repr(sentinel)


def test_atomic_write_json_uses_repr_fallback_for_unserializable(tmp_path: Path) -> None:
    path = tmp_path / "weird.json"
    _atomic_write_json(path, {"obj": {1, 2, 3}})  # a set is not JSON-serializable
    text = path.read_text()
    assert "1" in text  # the repr of the set landed in the file


# ---------------------------------------------------------------------------
# Filesystem primitives
# ---------------------------------------------------------------------------


def test_link_or_copy_falls_back_to_copy_when_link_unsupported(tmp_path: Path) -> None:
    src = tmp_path / "src.txt"
    src.write_text("payload")
    dst = tmp_path / "dst.txt"
    dst.write_text("stale")  # dst already exists -> os.link raises -> copy2 fallback
    _link_or_copy(src, dst)
    assert dst.read_text() == "payload"


def test_publish_dir_overwrites_existing_and_clears_stale_old(tmp_path: Path) -> None:
    target = tmp_path / "cur"
    target.mkdir()
    (target / "a.txt").write_text("v1")
    # A stale ``.old`` from a prior interrupted swap must be cleared first.
    old = tmp_path / "cur.old"
    old.mkdir()
    (old / "junk.txt").write_text("junk")
    staging = tmp_path / "stage"
    staging.mkdir()
    (staging / "a.txt").write_text("v2")

    _publish_dir(staging, target)

    assert (target / "a.txt").read_text() == "v2"
    assert not old.exists()
    assert not staging.exists()


def test_publish_dir_fresh_target_moves_staging(tmp_path: Path) -> None:
    target = tmp_path / "cur"
    staging = tmp_path / "stage"
    staging.mkdir()
    (staging / "a.txt").write_text("v1")
    _publish_dir(staging, target)
    assert (target / "a.txt").read_text() == "v1"
    assert not staging.exists()


# ---------------------------------------------------------------------------
# Scan / index helpers
# ---------------------------------------------------------------------------


def test_iter_task_dirs_missing_tasks_dir_is_empty(tmp_path: Path) -> None:
    assert iter_task_dirs(tmp_path / "does-not-exist") == []
    empty = tmp_path / "exp"
    empty.mkdir()
    assert iter_task_dirs(empty) == []


def test_plan_resume_skips_corrupt_prior_task(tmp_path: Path) -> None:
    exp = tmp_path / "exp"
    good = exp / "tasks" / _task_dirname(1, "g1")
    good.mkdir(parents=True)
    _atomic_write_json(
        good / "task.json",
        {"index": 1, "goal": "g1", "goal_hash": goal_hash("g1"), "status": "success"},
    )
    bad = exp / "tasks" / _task_dirname(2, "g2")
    bad.mkdir(parents=True)
    _atomic_write_json(bad / "task.json", {"no_index_here": True})  # raises on int(data["index"])

    plan = plan_resume(exp, ["g1", "g2"], overwrite=False)
    assert 1 in plan.keep
    assert 2 in plan.rerun  # corrupt prior is treated as missing -> rerun


def test_next_previous_index_ignores_non_matching_names(tmp_path: Path) -> None:
    (tmp_path / "previous_01").mkdir()
    (tmp_path / "previous_02").mkdir()
    (tmp_path / "previous_bad").mkdir()  # matches glob, not the \d+ regex
    (tmp_path / "previous_03.wip").mkdir()  # matches glob, not the regex
    assert _next_previous_index(tmp_path) == 3


def test_snapshot_current_removes_stale_wip_staging(tmp_path: Path) -> None:
    exp = tmp_path / "exp"
    exp.mkdir()
    _atomic_write_json(exp / "result.json", {"marker": "v1"})
    # A leftover staging dir from a prior crash must be removed before writing.
    stale = exp / "previous_01.wip"
    stale.mkdir()
    (stale / "garbage.txt").write_text("stale")

    snap = snapshot_current(exp)
    assert snap is not None
    assert snap.name == "previous_01"
    assert not stale.exists()
    assert (snap / "result.json").exists()


def test_gc_staging_removes_stray_wip_and_old(tmp_path: Path) -> None:
    exp = tmp_path / "exp"
    (exp / "tasks").mkdir(parents=True)
    (exp / "a.wip").mkdir()
    (exp / "tasks" / "b.wip").mkdir()
    (exp / "tasks" / "c.old").mkdir()
    (exp / "keep").mkdir()
    # A stray *file* (not a dir) matching the glob is left alone (the is_dir gate).
    stray_file = exp / "stray.wip"
    stray_file.write_text("not a dir")

    _gc_staging(exp)

    assert not (exp / "a.wip").exists()
    assert not (exp / "tasks" / "b.wip").exists()
    assert not (exp / "tasks" / "c.old").exists()
    assert (exp / "keep").exists()  # non-staging dirs untouched
    assert stray_file.exists()  # a matching non-dir is skipped, not removed


# ---------------------------------------------------------------------------
# Experiment lock
# ---------------------------------------------------------------------------


def test_experiment_lock_contention_raises_and_release_is_safe(tmp_path: Path) -> None:
    exp = tmp_path / "exp"
    exp.mkdir()
    lock1 = _ExperimentLock(exp)
    lock1.acquire()
    try:
        lock2 = _ExperimentLock(exp)
        with pytest.raises(RuntimeError, match="locked"):
            lock2.acquire()
        # A failed acquire leaves no fh; release is a safe no-op.
        lock2.release()
    finally:
        lock1.release()
    # After release the dir is lockable again.
    lock3 = _ExperimentLock(exp)
    lock3.acquire()
    lock3.release()


# ---------------------------------------------------------------------------
# ExperimentSession: per-task surface, resume snapshot, abort
# ---------------------------------------------------------------------------


def test_session_log_paths(tmp_path: Path) -> None:
    root = tmp_path / "results"
    session = ExperimentSession.open(root, BASE_META, ["g1", "g2"])
    try:
        assert session.experiment_log_path() == session.experiment_dir / "logs" / "diagnostics.log"
        # No staging yet -> both index and None fall back to the experiment log.
        assert session.task_log_path(1) == session.experiment_log_path()
        assert session.task_log_path(None) == session.experiment_log_path()
        # With a staging dir the task's own log path is used.
        staging = session.begin_task(1, "g1")
        assert session.task_log_path(1) == staging / "logs" / "diagnostics.log"
    finally:
        session.abort()


def test_session_begin_task_twice_replaces_stale_staging(tmp_path: Path) -> None:
    root = tmp_path / "results"
    session = ExperimentSession.open(root, BASE_META, ["g1"])
    try:
        first = session.begin_task(1, "g1")
        (first / "marker.txt").write_text("stale")
        second = session.begin_task(1, "g1")
        assert first == second
        assert not (second / "marker.txt").exists()  # stale staging was wiped
        assert (second / "logs").is_dir()
        assert (second / "trajectories").is_dir()
    finally:
        session.abort()


def test_session_publish_without_begin_creates_staging(tmp_path: Path) -> None:
    root = tmp_path / "results"
    session = ExperimentSession.open(root, BASE_META, ["only goal"])
    session.publish_task(1, _make_task_result("only goal"))  # no begin_task first
    session.finalize(None, None)
    dirs = iter_task_dirs(session.experiment_dir)
    assert len(dirs) == 1


def test_session_open_resume_snapshots_prior_then_republishes(tmp_path: Path) -> None:
    root = tmp_path / "results"
    # First run leaves an errored task -> it reruns on resume.
    _write_tree(
        root,
        BASE_META,
        [_make_task_result("g1", score=0.0, success=False, stop_reason="error", error="boom")],
    )

    session = ExperimentSession.open(root, BASE_META, ["g1"])
    assert session.plan.is_fresh is False
    assert session.plan.rerun == frozenset({1})
    # A previous_NN snapshot was taken before the rerun touched current.
    assert (session.experiment_dir / "previous_01").is_dir()

    # Republishing over the existing current task dir exercises the swap path.
    session.begin_task(1, "g1")
    session.publish_task(1, _make_task_result("g1", score=1.0, success=True, stop_reason="done"))
    session.finalize(datetime(2026, 1, 2, tzinfo=UTC), datetime(2026, 1, 2, 0, 1, tzinfo=UTC))

    from anonframework.core.persistence import load_task

    current = iter_task_dirs(session.experiment_dir)[0]
    assert load_task(current)["success"] is True  # current now reflects the rerun
    # ...and the snapshot RETAINED the pre-rerun errored result after the
    # republish over current (snapshot-before-rerun immutability, the guarantee).
    snap_task = iter_task_dirs(session.experiment_dir / "previous_01")[0]
    snap = load_task(snap_task)
    assert snap["success"] is False
    assert snap["stop_reason"] == "error"


def test_session_open_releases_lock_on_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "results"

    def boom(*_a: object, **_k: object) -> None:
        raise RuntimeError("plan failed")

    monkeypatch.setattr(persistence, "plan_resume", boom)
    with pytest.raises(RuntimeError, match="plan failed"):
        ExperimentSession.open(root, BASE_META, ["g1"])

    # The lock was released on the failure path, so a clean re-open succeeds.
    monkeypatch.undo()
    session = ExperimentSession.open(root, BASE_META, ["g1"])
    session.abort()


# ---------------------------------------------------------------------------
# Kept-task reconstruction
# ---------------------------------------------------------------------------


def test_reconstruct_kept_task_result_scalar_faithful(tmp_path: Path) -> None:
    root = tmp_path / "results"
    exp_dir = _write_tree(
        root,
        BASE_META,
        [
            _make_task_result("g1", score=0.3, success=False, stop_reason="max_runs"),
            _make_task_result("g2", score=0.9, success=True, stop_reason="done"),
        ],
    )
    # Reconstructing index 2 must skip the index-1 dir first (the continue path).
    tr = reconstruct_kept_task_result(exp_dir, 2, StubTask(goal_text="g2"))
    assert tr.stop_reason == "done"
    assert tr.success is True
    assert tr.best_score.value == pytest.approx(0.9)
    assert tr.runs == []  # trajectories stay on disk, not reloaded
    assert tr.started_at is not None


def test_reconstruct_kept_task_result_missing_index_raises(tmp_path: Path) -> None:
    root = tmp_path / "results"
    exp_dir = _write_tree(root, BASE_META, [_make_task_result("g1")])
    with pytest.raises(FileNotFoundError, match="index 99"):
        reconstruct_kept_task_result(exp_dir, 99, StubTask(goal_text="g1"))


# ---------------------------------------------------------------------------
# _parse_iso
# ---------------------------------------------------------------------------


def test_parse_iso_none_empty_and_invalid() -> None:
    assert _parse_iso(None) is None
    assert _parse_iso("") is None
    assert _parse_iso("not-a-timestamp") is None
    parsed = _parse_iso("2026-01-01T00:00:00+00:00")
    assert parsed == datetime(2026, 1, 1, tzinfo=UTC)
