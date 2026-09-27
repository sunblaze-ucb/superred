"""Unit tests for ``anonframework.core.persistence`` (schema v4).

Covers the pure / unit layer: naming + identity, goal hashing, resume
planning, snapshot immutability, the public reader API over a written tree,
summary computation, atomic writes, and results-root resolution.  A tree is
written by driving an :class:`ExperimentSession` directly (no LLM, no
Controller) with hand-built :class:`TaskResult` objects.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from anonframework.core.controller import RunResult, TaskResult
from anonframework.core.persistence import (
    DEFAULT_RESULTS_ROOT,
    SCHEMA_VERSION,
    ExperimentMeta,
    ExperimentSession,
    TaskView,
    _atomic_write_json,
    _compute_summary,
    _goal_slug,
    _slug_segment,
    _task_dirname,
    goal_hash,
    iter_task_dirs,
    iter_tasks,
    load_experiments_index,
    load_iterations,
    load_manifest,
    load_result,
    load_task,
    load_trajectory,
    plan_resume,
    resolve_results_root,
    snapshot_current,
)
from anonframework.core.types.evaluation import EvaluationResult, Score
from anonframework.core.types.llm import LLMUsage
from anonframework.core.types.trajectory import Trajectory

from .conftest import EXTERNAL_TAG, INTERNAL_TAG, ROOT_TAG, StubTask

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _make_task_result(
    goal: str = "Test goal",
    *,
    score: float = 1.0,
    success: bool = True,
    stop_reason: str = "done",
    calls: int = 2,
    cost: float = 0.05,
    error: str | None = None,
    n_runs: int = 1,
) -> TaskResult:
    """Hand-build a ``TaskResult`` with ``n_runs`` runs and empty trajectories."""
    evaluation = EvaluationResult(success=success, primary_score=Score(value=score))
    started = datetime(2026, 1, 1, tzinfo=UTC)
    ended = datetime(2026, 1, 1, 0, 1, tzinfo=UTC)
    runs = [
        RunResult(
            trajectory=Trajectory(),
            evaluation=evaluation,
            llm_usage=LLMUsage(calls=calls, cost=cost),
            run_usage_delta=LLMUsage(calls=calls, cost=cost),
            started_at=started,
            ended_at=ended,
            evaluated=error is None,
            errored=error is not None,
            done=stop_reason == "done",
        )
        for _ in range(n_runs)
    ]
    return TaskResult(
        task=StubTask(score=score, success=success, goal_text=goal),
        runs=runs,
        best_score=Score(value=score),
        best_evaluation=evaluation,
        success=success,
        llm_usage=LLMUsage(calls=calls, cost=cost),
        stop_reason=stop_reason,  # type: ignore[arg-type]
        scope=frozenset({EXTERNAL_TAG}),
        error=error,
        started_at=started,
        ended_at=ended,
    )


def _write_tree(
    root: Path,
    meta: ExperimentMeta,
    task_results: list[TaskResult],
    *,
    skipped: list[tuple[int, str]] | None = None,
) -> Path:
    """Drive an ``ExperimentSession`` to write a full experiment tree.

    Returns the experiment directory.
    """
    goals = [tr.task.goal.description for tr in task_results]
    session = ExperimentSession.open(root, meta, goals)
    for i, tr in enumerate(task_results, start=1):
        session.begin_task(i, tr.task.goal.description)
        session.publish_task(i, tr)
    for index, goal in skipped or []:
        session.mark_skipped(index, goal)
    session.finalize(datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, 0, 5, tzinfo=UTC))
    return root / meta.dirname()


def _make_task_view(
    index: int,
    *,
    success: bool,
    stop_reason: str,
    best_score: float,
    calls: int = 1,
    cost: float = 0.01,
    error: str | None = None,
    status: str = "success",
) -> TaskView:
    return TaskView(
        index=index,
        goal=f"goal {index}",
        goal_hash=goal_hash(f"goal {index}"),
        dir=f"tasks/{index:05d}__goal",
        status=status,
        success=success,
        stop_reason=stop_reason,
        best_score=best_score,
        n_runs=1,
        calls=calls,
        cost=cost,
        error=error,
        started_at=None,
        ended_at=None,
    )


BASE_META = ExperimentMeta(
    attacker="atk",
    target="tgt",
    claim="clm",
    model="test-model",
    scope=("external",),
)


# ---------------------------------------------------------------------------
# (1) ExperimentMeta: slug / identity_hash / dirname
# ---------------------------------------------------------------------------


def test_slug_shape_and_lowercasing() -> None:
    meta = ExperimentMeta(attacker="MyAtk", target="Tgt", claim="Claim", model="GPT-4o")
    assert meta.slug() == "myatk__tgt__claim__gpt-4o"


def test_slug_model_none_becomes_no_llm() -> None:
    meta = ExperimentMeta(attacker="a", target="t", claim="c", model=None)
    assert meta.slug() == "a__t__c__no-llm"


def test_slug_excludes_scope_tag_names() -> None:
    meta = ExperimentMeta(
        attacker="a", target="t", claim="c", model="m", scope=("external", "internal")
    )
    slug = meta.slug()
    assert "external" not in slug
    assert "internal" not in slug
    assert slug == "a__t__c__m"


def test_slug_sanitizes_unsafe_chars() -> None:
    meta = ExperimentMeta(attacker="openai/gpt 4o", target="t", claim="c", model="m")
    # spaces and slashes collapse to a single dash.
    assert meta.slug().split("__")[0] == "openai-gpt-4o"


def test_slug_segment_reserved_and_empty_become_x() -> None:
    assert _slug_segment("con") == "x"
    assert _slug_segment("COM1") == "x"
    assert _slug_segment("") == "x"
    assert _slug_segment("///") == "x"
    assert _slug_segment("...") == "x"


def test_slug_segment_truncates_to_24() -> None:
    seg = _slug_segment("a" * 100)
    assert seg == "a" * 24


def test_identity_hash_is_8_lowercase_hex() -> None:
    h = BASE_META.identity_hash()
    assert len(h) == 8
    assert h == h.lower()
    assert all(c in "0123456789abcdef" for c in h)


def test_identity_hash_deterministic() -> None:
    assert BASE_META.identity_hash() == BASE_META.identity_hash()


def test_dirname_is_slug_dash_hash() -> None:
    assert BASE_META.dirname() == f"{BASE_META.slug()}-{BASE_META.identity_hash()}"


def test_scope_changes_identity_hash() -> None:
    a = ExperimentMeta(attacker="a", target="t", claim="c", model="m", scope=("external",))
    b = ExperimentMeta(attacker="a", target="t", claim="c", model="m", scope=("internal",))
    assert a.identity_hash() != b.identity_hash()


def test_scope_order_does_not_change_identity_hash() -> None:
    a = ExperimentMeta(
        attacker="a", target="t", claim="c", model="m", scope=("external", "internal")
    )
    b = ExperimentMeta(
        attacker="a", target="t", claim="c", model="m", scope=("internal", "external")
    )
    assert a.identity_hash() == b.identity_hash()


def test_task_cost_cap_changes_identity_hash() -> None:
    a = ExperimentMeta(attacker="a", target="t", claim="c", model="m", task_cost_cap_usd=1.0)
    b = ExperimentMeta(attacker="a", target="t", claim="c", model="m", task_cost_cap_usd=2.0)
    assert a.identity_hash() != b.identity_hash()


def test_concurrency_does_not_change_identity_hash() -> None:
    a = ExperimentMeta(attacker="a", target="t", claim="c", model="m", concurrency=1)
    b = ExperimentMeta(attacker="a", target="t", claim="c", model="m", concurrency=8)
    assert a.identity_hash() == b.identity_hash()
    assert a.dirname() == b.dirname()


def test_n_tasks_does_not_change_identity_hash() -> None:
    a = ExperimentMeta(attacker="a", target="t", claim="c", model="m", n_tasks=None)
    b = ExperimentMeta(attacker="a", target="t", claim="c", model="m", n_tasks=42)
    assert a.identity_hash() == b.identity_hash()


def test_read_only_and_labels_change_identity_hash() -> None:
    base = ExperimentMeta(attacker="a", target="t", claim="c", model="m", scope=("external",))
    ro = ExperimentMeta(
        attacker="a",
        target="t",
        claim="c",
        model="m",
        scope=("external",),
        read_only=("internal",),
    )
    labeled = ExperimentMeta(
        attacker="a",
        target="t",
        claim="c",
        model="m",
        scope=("external",),
        scope_label="lbl",
    )
    feedback = ExperimentMeta(
        attacker="a",
        target="t",
        claim="c",
        model="m",
        scope=("external",),
        include_feedback=False,
    )
    hashes = {
        base.identity_hash(),
        ro.identity_hash(),
        labeled.identity_hash(),
        feedback.identity_hash(),
    }
    assert len(hashes) == 4


def test_experiment_block_sorts_scope() -> None:
    meta = ExperimentMeta(
        attacker="a", target="t", claim="c", model="m", scope=("internal", "external")
    )
    block = meta.experiment_block()
    assert block["scope"] == ["external", "internal"]
    assert block["slug"] == meta.slug()
    assert block["hash"] == meta.identity_hash()


# ---------------------------------------------------------------------------
# (2) goal_hash
# ---------------------------------------------------------------------------


def test_goal_hash_is_16_lowercase_hex() -> None:
    h = goal_hash("some goal")
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)


def test_goal_hash_stable() -> None:
    assert goal_hash("same") == goal_hash("same")


def test_goal_hash_distinct() -> None:
    assert goal_hash("goal a") != goal_hash("goal b")


# ---------------------------------------------------------------------------
# (3) _goal_slug / _task_dirname
# ---------------------------------------------------------------------------


def test_goal_slug_sanitizes_and_lowercases() -> None:
    assert _goal_slug("Steal the API key!") == "steal_the_api_key"


def test_goal_slug_truncates_to_50() -> None:
    assert _goal_slug("z" * 200) == "z" * 50


def test_goal_slug_empty_becomes_task() -> None:
    assert _goal_slug("") == "task"
    assert _goal_slug("///") == "task"


def test_task_dirname_zero_pads_index() -> None:
    assert _task_dirname(1, "Do a thing") == "00001__do_a_thing"
    assert _task_dirname(42, "x") == "00042__x"


# ---------------------------------------------------------------------------
# (4) plan_resume
# ---------------------------------------------------------------------------


def _write_prior_task(exp_dir: Path, index: int, goal: str, status: str) -> None:
    """Hand-write a minimal prior current task dir with a task.json."""
    task_dir = exp_dir / "tasks" / _task_dirname(index, goal)
    task_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(
        task_dir / "task.json",
        {
            "schema_version": SCHEMA_VERSION,
            "index": index,
            "goal": goal,
            "goal_hash": goal_hash(goal),
            "status": status,
        },
    )


def test_plan_resume_fresh_when_no_dir(tmp_path: Path) -> None:
    plan = plan_resume(tmp_path / "nope", ["g1", "g2"], overwrite=False)
    assert plan.is_fresh is True
    assert plan.keep == frozenset()
    assert plan.rerun == frozenset({1, 2})


def test_plan_resume_keeps_matching_kept_status(tmp_path: Path) -> None:
    exp = tmp_path / "exp"
    _write_prior_task(exp, 1, "g1", "success")
    _write_prior_task(exp, 2, "g2", "failed")
    _write_prior_task(exp, 3, "g3", "budget_exhausted")
    plan = plan_resume(exp, ["g1", "g2", "g3"], overwrite=False)
    assert plan.is_fresh is False
    assert plan.keep == frozenset({1, 2, 3})
    assert plan.rerun == frozenset()


def test_plan_resume_reruns_error_status(tmp_path: Path) -> None:
    exp = tmp_path / "exp"
    _write_prior_task(exp, 1, "g1", "error")
    plan = plan_resume(exp, ["g1"], overwrite=False)
    assert plan.keep == frozenset()
    assert plan.rerun == frozenset({1})


def test_plan_resume_reruns_on_goal_hash_mismatch(tmp_path: Path) -> None:
    exp = tmp_path / "exp"
    _write_prior_task(exp, 1, "old goal", "success")
    plan = plan_resume(exp, ["new goal"], overwrite=False)
    assert plan.keep == frozenset()
    assert plan.rerun == frozenset({1})


def test_plan_resume_reruns_appended_task(tmp_path: Path) -> None:
    exp = tmp_path / "exp"
    _write_prior_task(exp, 1, "g1", "success")
    # A second goal appended; no prior task at index 2 -> rerun only it.
    plan = plan_resume(exp, ["g1", "g2"], overwrite=False)
    assert plan.keep == frozenset({1})
    assert plan.rerun == frozenset({2})


def test_plan_resume_overwrite_reruns_all(tmp_path: Path) -> None:
    exp = tmp_path / "exp"
    _write_prior_task(exp, 1, "g1", "success")
    _write_prior_task(exp, 2, "g2", "success")
    plan = plan_resume(exp, ["g1", "g2"], overwrite=True)
    assert plan.is_fresh is False
    assert plan.keep == frozenset()
    assert plan.rerun == frozenset({1, 2})


# ---------------------------------------------------------------------------
# (5) snapshot_current immutability
# ---------------------------------------------------------------------------


def test_snapshot_current_immutability(tmp_path: Path) -> None:
    exp = tmp_path / "exp"
    (exp / "tasks").mkdir(parents=True)
    _atomic_write_json(exp / "result.json", {"marker": "v1"})
    _atomic_write_json(exp / "manifest.json", {"status": "complete"})
    _write_prior_task(exp, 1, "g1", "success")

    first = snapshot_current(exp)
    assert first is not None
    assert first.name == "previous_01"
    assert json.loads((first / "result.json").read_text())["marker"] == "v1"

    # Overwrite a current file via the atomic writer; the hardlinked snapshot
    # copy must be unaffected (os.replace swaps to a fresh inode).
    _atomic_write_json(exp / "result.json", {"marker": "v2"})
    assert json.loads((exp / "result.json").read_text())["marker"] == "v2"
    assert json.loads((first / "result.json").read_text())["marker"] == "v1"

    second = snapshot_current(exp)
    assert second is not None
    assert second.name == "previous_02"
    assert json.loads((second / "result.json").read_text())["marker"] == "v2"


def test_snapshot_current_returns_none_when_nothing(tmp_path: Path) -> None:
    exp = tmp_path / "empty"
    exp.mkdir()
    assert snapshot_current(exp) is None


# ---------------------------------------------------------------------------
# (5b) Resume reconciles current with the live claim (stale-dir pruning)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("first", "second"),
    [
        # Edit one goal's text: its prior dir is orphaned under a stale slug.
        (["goal alpha original", "goal beta"], ["goal alpha edited now", "goal beta"]),
        # Reorder: every index's goal_hash changes, so all prior dirs go stale.
        (["goal alpha", "goal beta"], ["goal beta", "goal alpha"]),
        # Shrink: the dropped task's dir has no live index to overwrite it.
        (["goal a", "goal b", "goal c"], ["goal a", "goal b"]),
    ],
    ids=["edited-goal", "reordered", "shrunk-claim"],
)
def test_resume_prunes_stale_task_dirs(tmp_path: Path, first: list[str], second: list[str]) -> None:
    # A resume against an edited / reordered / shrunk claim used to leave the
    # prior run's now-mismatched task dirs on disk, so iter_tasks + result.json
    # + the experiments.json row double-counted them (inflating n_tasks / ASR).
    # open() now reconciles current with the live claim.
    _write_tree(tmp_path, BASE_META, [_make_task_result(g) for g in first])
    exp = _write_tree(tmp_path, BASE_META, [_make_task_result(g) for g in second])

    assert len(iter_tasks(exp)) == len(second)
    assert load_result(exp)["summary"]["n_tasks"] == len(second)
    row = load_experiments_index(tmp_path)["experiments"][0]["summary"]
    assert row["n_completed"] == len(second)


def test_resume_prune_preserves_pruned_task_in_snapshot(tmp_path: Path) -> None:
    # Pruning must be loss-free: a task dropped from the claim on resume leaves
    # current but survives in the immutable previous_NN snapshot.
    _write_tree(tmp_path, BASE_META, [_make_task_result(g) for g in ("goal a", "goal b", "goal c")])
    exp = _write_tree(tmp_path, BASE_META, [_make_task_result(g) for g in ("goal a", "goal b")])

    assert not any("goal_c" in d.name for d in (exp / "tasks").iterdir())  # gone from current
    hist = [d.name for snap in exp.glob("previous_*/tasks") for d in snap.iterdir()]
    assert any("goal_c" in name for name in hist)  # preserved in history


def test_experiments_index_dedups_on_rerun_and_sorts_rows(tmp_path: Path) -> None:
    # Re-running the SAME experiment updates its single row in place (no
    # duplicate), and multiple experiments in one root are listed sorted by dir.
    _write_tree(tmp_path, BASE_META, [_make_task_result("goal one")])
    _write_tree(tmp_path, BASE_META, [_make_task_result("goal one")])  # rerun, same identity
    rows = load_experiments_index(tmp_path)["experiments"]
    assert len(rows) == 1  # rerun updated the row, not appended

    # A second experiment whose dir sorts BEFORE the first, inserted AFTER it,
    # must still come out first -> proves the rows are sorted, not insertion-order.
    other = ExperimentMeta(
        attacker="aaa", target="tgt", claim="clm", model="test-model", scope=("external",)
    )
    _write_tree(tmp_path, other, [_make_task_result("goal two")])
    dirs = [r["dir"] for r in load_experiments_index(tmp_path)["experiments"]]
    assert len(dirs) == 2
    assert dirs == sorted(dirs) and dirs[0].startswith("aaa")


# ---------------------------------------------------------------------------
# (6) Reader API over a written tree
# ---------------------------------------------------------------------------


def test_reader_api_over_written_tree(tmp_path: Path) -> None:
    root = tmp_path / "results"
    task_results = [
        _make_task_result("goal one", score=0.8, success=True, stop_reason="done"),
        _make_task_result("goal two", score=0.0, success=False, stop_reason="max_runs"),
    ]
    exp_dir = _write_tree(root, BASE_META, task_results)

    # iter_task_dirs / iter_tasks ordered by index.
    dirs = iter_task_dirs(exp_dir)
    assert len(dirs) == 2
    assert [d.name for d in dirs] == sorted(d.name for d in dirs)

    views = iter_tasks(exp_dir)
    assert [v.index for v in views] == [1, 2]
    assert views[0].goal == "goal one"
    assert views[0].success is True
    assert views[0].best_score == pytest.approx(0.8)
    assert views[1].stop_reason == "max_runs"

    # load_task / load_iterations / load_trajectory.
    task = load_task(dirs[0])
    assert task["schema_version"] == SCHEMA_VERSION
    assert task["index"] == 1
    assert task["scope"] == ["external"]
    assert task["llm_config"] == {"model": "test-model"}

    iters = load_iterations(dirs[0])
    assert len(iters["runs"]) == 1
    assert iters["runs"][0]["run_number"] == 1

    traj = load_trajectory(dirs[0], 1)
    assert traj["run_number"] == 1
    assert traj["trajectory"] == []

    # manifest + result.
    manifest = load_manifest(exp_dir)
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["status"] == "complete"
    assert len(manifest["tasks"]) == 2

    result = load_result(exp_dir)
    assert result["schema_version"] == SCHEMA_VERSION
    assert result["summary"]["n_tasks"] == 2
    assert result["summary"]["n_success"] == 1


def test_written_tree_never_leaks_secrets(tmp_path: Path) -> None:
    root = tmp_path / "results"
    exp_dir = _write_tree(root, BASE_META, [_make_task_result("g")])
    for path in exp_dir.rglob("*.json"):
        text = path.read_text()
        assert "sk-test" not in text
        assert "api_key" not in text
        assert "api_base" not in text


def test_written_tree_skipped_task_in_manifest(tmp_path: Path) -> None:
    root = tmp_path / "results"
    exp_dir = _write_tree(root, BASE_META, [_make_task_result("g1")], skipped=[(2, "skipped goal")])
    manifest = load_manifest(exp_dir)
    skipped = [t for t in manifest["tasks"] if t["status"] == "skipped"]
    assert len(skipped) == 1
    assert skipped[0]["dir"] is None
    assert manifest["summary"]["n_skipped"] == 1


# ---------------------------------------------------------------------------
# (7) _compute_summary
# ---------------------------------------------------------------------------


def test_compute_summary_counts_and_asr() -> None:
    views = [
        _make_task_view(1, success=True, stop_reason="done", best_score=1.0),
        _make_task_view(2, success=False, stop_reason="max_runs", best_score=0.0),
        _make_task_view(3, success=False, stop_reason="budget_exhausted", best_score=0.3),
        _make_task_view(4, success=False, stop_reason="error", best_score=0.0, error="boom"),
    ]
    summary = _compute_summary(views, n_skipped=2)

    assert summary["n_tasks"] == 4
    assert summary["n_success"] == 1
    # completed = done + max_runs + budget_exhausted (error excluded).
    assert summary["n_completed"] == 3
    assert summary["n_failed"] == 2
    assert summary["n_error"] == 1
    assert summary["n_budget_exhausted"] == 1
    assert summary["n_skipped"] == 2
    assert summary["asr"] == pytest.approx(1 / 3)
    assert summary["max_primary_score"] == pytest.approx(1.0)
    assert summary["mean_primary_score"] == pytest.approx((1.0 + 0.0 + 0.3 + 0.0) / 4)


def test_compute_summary_asr_none_when_no_completed() -> None:
    views = [_make_task_view(1, success=False, stop_reason="error", best_score=0.0)]
    summary = _compute_summary(views, n_skipped=0)
    assert summary["asr"] is None
    assert summary["n_completed"] == 0


def test_compute_summary_empty() -> None:
    summary = _compute_summary([], n_skipped=0)
    assert summary["asr"] is None
    assert summary["n_tasks"] == 0
    assert summary["max_primary_score"] is None
    assert summary["mean_primary_score"] is None
    assert summary["total_llm_usage"] == {"calls": 0, "cost": 0}


def test_compute_summary_totals_usage() -> None:
    views = [
        _make_task_view(1, success=True, stop_reason="done", best_score=1.0, calls=2, cost=0.1),
        _make_task_view(2, success=True, stop_reason="done", best_score=1.0, calls=3, cost=0.2),
    ]
    summary = _compute_summary(views, n_skipped=0)
    assert summary["total_llm_usage"]["calls"] == 5
    assert summary["total_llm_usage"]["cost"] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# (8) _atomic_write_json
# ---------------------------------------------------------------------------


def test_atomic_write_json_writes_valid_json_no_tmp(tmp_path: Path) -> None:
    path = tmp_path / "out.json"
    payload = {"a": 1, "b": ["x", "y"], "nested": {"k": True}}
    _atomic_write_json(path, payload)
    assert json.loads(path.read_text()) == payload
    assert not (tmp_path / "out.json.tmp").exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_write_json_serializes_datetime(tmp_path: Path) -> None:
    path = tmp_path / "dt.json"
    dt = datetime(2026, 1, 1, tzinfo=UTC)
    _atomic_write_json(path, {"when": dt})  # type: ignore[dict-item]
    assert json.loads(path.read_text())["when"] == dt.isoformat()


# ---------------------------------------------------------------------------
# (9) resolve_results_root
# ---------------------------------------------------------------------------


def test_resolve_results_root_explicit_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANONFRAMEWORK_RESULTS_DIR", "/env/dir")
    assert resolve_results_root("/explicit") == Path("/explicit")


def test_resolve_results_root_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANONFRAMEWORK_RESULTS_DIR", "/env/dir")
    assert resolve_results_root(None) == Path("/env/dir")


def test_resolve_results_root_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANONFRAMEWORK_RESULTS_DIR", raising=False)
    assert resolve_results_root(None) == Path(DEFAULT_RESULTS_ROOT)


# Reference imports so unused-tag lint stays quiet and tags are exercised.
_TAGS = (EXTERNAL_TAG, INTERNAL_TAG, ROOT_TAG)


def test_unreadable_task_record_is_skipped_not_fatal(tmp_path: Path) -> None:
    """A damaged task.json must not make the whole experiment unopenable.

    ``_scan_prior_tasks`` already tolerates one (the task just counts as
    outstanding), but ``iter_tasks`` used to raise, and ``ExperimentSession.open``
    calls it before any task runs -- so a single truncated write, from a crash or
    a full disk, permanently blocked that experiment: it could never start again
    to repair itself.
    """
    root = tmp_path / "results"
    task_results = [
        _make_task_result("goal one", score=0.8, success=True, stop_reason="done"),
        _make_task_result("goal two", score=0.0, success=False, stop_reason="done"),
    ]
    exp_dir = _write_tree(root, BASE_META, task_results)
    goals = [tr.task.goal.description for tr in task_results]

    # Truncate the first task's record the way an interrupted write would.
    damaged = iter_task_dirs(exp_dir)[0] / "task.json"
    raw = damaged.read_text(encoding="utf-8")
    damaged.write_text(raw[: len(raw) // 2], encoding="utf-8")

    views = iter_tasks(exp_dir)
    assert [v.index for v in views] == [2], "the readable task must still be reported"

    plan = plan_resume(exp_dir, goals, overwrite=False)
    assert plan.rerun == frozenset({1}), "the damaged task must be recomputed"
    assert plan.keep == frozenset({2}), "its neighbour must still be kept"

    # The real consequence: the experiment can be opened again, so the next run
    # repairs it instead of dying on it.
    session = ExperimentSession.open(root, BASE_META, goals)
    try:
        assert session.plan.rerun == frozenset({1})
    finally:
        session.abort()
