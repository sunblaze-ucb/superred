"""End-to-end persistence tests for the v4 result tree (``anonframework.core.persistence``).

A ``Controller(persist=True, results_dir=tmp_path, ...)`` writes a self-describing
tree under ``{results_dir}/{slug}-{hash8}/``. These tests drive a real Controller
with stub target/optimizer/tasks and assert the FULL v4 layout via the public
reader API, plus the secret allowlist (no credentials ever land on disk).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anonframework.core.controller import Controller, TargetFactory
from anonframework.core.interfaces.security_claim import SecurityClaim
from anonframework.core.persistence import (
    SCHEMA_VERSION,
    ExperimentMeta,
    iter_task_dirs,
    iter_tasks,
    load_experiments_index,
    load_iterations,
    load_manifest,
    load_result,
    load_task,
    load_trajectory,
)
from anonframework.core.types.security_domain import Scope

from .conftest import (
    EXTERNAL_TAG,
    STUB_LLM_CONFIG,
    CountingOptimizer,
    NotApplicableTask,
    StubTarget,
    StubTask,
)

EXTERNAL_SCOPE: Scope = frozenset({EXTERNAL_TAG})

ATTACKER_LABEL = "atk"
TARGET_LABEL = "tgt"
CLAIM_LABEL = "clm"


def _expected_dirname(n_tasks: int) -> str:
    """The experiment dir the Controller below writes to.

    ``identity_hash`` excludes ``concurrency`` and ``n_tasks``, so building a
    matching :class:`ExperimentMeta` reproduces the exact folder name.
    """
    meta = ExperimentMeta(
        attacker=ATTACKER_LABEL,
        target=TARGET_LABEL,
        claim=CLAIM_LABEL,
        model="test-model",
        scope=("external",),
        read_only=(),
        scope_label=None,
        task_cost_cap_usd=None,
        max_runs_per_task=1,
        include_feedback=True,
        n_tasks=n_tasks,
    )
    return meta.dirname()


def _make_controller(tmp_path: Path, tasks: list[StubTask], **overrides: object) -> Controller:
    kwargs: dict[str, object] = dict(
        scope=EXTERNAL_SCOPE,
        optimizer_factory=lambda: CountingOptimizer(stop_after=1),
        target_factory=TargetFactory(create=StubTarget, concurrency=1),
        security_claim=SecurityClaim.from_tasks(tasks),
        llm_config=STUB_LLM_CONFIG,
        results_dir=tmp_path,
        persist=True,
        report=False,
        max_runs_per_task=1,
        attacker_label=ATTACKER_LABEL,
        target_label=TARGET_LABEL,
        claim_label=CLAIM_LABEL,
    )
    kwargs.update(overrides)
    return Controller(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Full v4 tree
# ---------------------------------------------------------------------------


async def test_full_v4_tree(tmp_path: Path) -> None:
    """One run of two successful tasks produces the complete v4 layout."""
    tasks = [
        StubTask(score=0.8, success=True, goal_text="alpha goal"),
        StubTask(score=1.0, success=True, goal_text="beta goal"),
    ]
    result = await _make_controller(tmp_path, tasks).run()

    # -- experiment dir named {slug}-{hash8} -------------------------------
    exp_dir = tmp_path / _expected_dirname(n_tasks=2)
    assert exp_dir.is_dir()
    # It is the only experiment dir written.
    subdirs = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert subdirs == [exp_dir]

    # -- manifest.json -----------------------------------------------------
    manifest = load_manifest(exp_dir)
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["status"] == "complete"
    m_summary = manifest["summary"]
    assert m_summary["n_tasks"] == 2
    assert m_summary["n_success"] == 2
    assert m_summary["n_completed"] == 2
    assert m_summary["n_failed"] == 0
    assert m_summary["n_error"] == 0
    assert m_summary["n_skipped"] == 0
    assert m_summary["asr"] == 1.0
    assert m_summary["max_primary_score"] == 1.0
    assert m_summary["mean_primary_score"] == pytest.approx(0.9)
    assert m_summary["total_llm_usage"] == {"calls": 0, "cost": 0.0}
    # one manifest task entry per task, each with a dir + success status.
    m_tasks = manifest["tasks"]
    assert [t["index"] for t in m_tasks] == [1, 2]
    for entry in m_tasks:
        assert entry["dir"] is not None
        assert entry["status"] == "success"
        assert entry["success"] is True
        assert entry["has_error"] is False
    assert manifest["experiment"]["hash"] == exp_dir.name.rsplit("-", 1)[1]

    # -- result.json (completion marker) -----------------------------------
    assert (exp_dir / "result.json").exists()
    result_json = load_result(exp_dir)
    assert result_json["schema_version"] == SCHEMA_VERSION
    assert result_json["summary"]["asr"] == 1.0
    assert result_json["summary"]["n_success"] == 2
    assert result_json["timing"]["started_at"] is not None
    assert result_json["timing"]["completed_at"] is not None

    # -- per-task task.json ------------------------------------------------
    views = iter_tasks(exp_dir)
    assert [v.index for v in views] == [1, 2]
    task_dirs = iter_task_dirs(exp_dir)
    assert len(task_dirs) == 2

    task1 = load_task(task_dirs[0])
    assert task1["schema_version"] == SCHEMA_VERSION
    assert task1["index"] == 1
    assert task1["goal"] == "alpha goal"
    assert task1["status"] == "success"
    assert task1["success"] is True
    assert task1["stop_reason"] == "success"
    assert task1["scope"] == ["external"]
    assert task1["read_only"] == []
    assert task1["n_runs"] == 1
    assert task1["best_score"]["value"] == 0.8
    assert task1["llm_usage"] == {"calls": 0, "cost": 0.0}
    assert task1["timing"]["started_at"] is not None
    assert task1["timing"]["ended_at"] is not None
    assert task1["timing"]["duration_s"] is not None
    assert task1["error"] is None

    # -- iterations.json (one run entry per run) ---------------------------
    iters = load_iterations(task_dirs[0])
    assert iters["schema_version"] == SCHEMA_VERSION
    assert iters["index"] == 1
    assert len(iters["runs"]) == 1
    run = iters["runs"][0]
    assert run["run_number"] == 1
    assert run["primary_score"] == 0.8
    assert run["success"] is True
    assert run["evaluated"] is True
    assert run["success"] is True
    assert run["usage_delta"] == {"calls": 0, "cost": 0.0}
    assert run["usage_cumulative"] == {"calls": 0, "cost": 0.0}
    assert run["trajectory"] == "trajectories/run_00001.json"

    # -- trajectories/run_00001.json ---------------------------------------
    traj = load_trajectory(task_dirs[0], 1)
    assert traj["schema_version"] == SCHEMA_VERSION
    assert traj["run_number"] == 1
    assert isinstance(traj["trajectory"], list)
    assert traj["trajectory"][-1]["type"] == "RunEndEvent"

    # -- logs/ dirs --------------------------------------------------------
    assert (exp_dir / "logs").is_dir()
    assert (task_dirs[0] / "logs").is_dir()

    # -- experiments.json at the root --------------------------------------
    index = load_experiments_index(tmp_path)
    assert index["schema_version"] == SCHEMA_VERSION
    assert len(index["experiments"]) == 1
    row = index["experiments"][0]
    assert row["dir"] == exp_dir.name
    assert row["status"] == "complete"
    assert row["summary"]["n_success"] == 2
    assert row["summary"]["asr"] == 1.0

    # -- returned ThreatModelResult mirrors disk ---------------------------
    assert len(result.task_results) == 2
    assert result.skipped_tasks == []


# ---------------------------------------------------------------------------
# Secret allowlist
# ---------------------------------------------------------------------------


async def test_secret_allowlist(tmp_path: Path) -> None:
    """Credentials never appear in any written file; llm_config is model-only."""
    tasks = [StubTask(goal_text="secret probe")]
    await _make_controller(tmp_path, tasks).run()

    json_files = list(tmp_path.rglob("*.json"))
    assert json_files  # sanity: something was written

    for path in json_files:
        text = path.read_text(encoding="utf-8")
        assert "sk-test" not in text, path
        assert "api_key" not in text, path
        assert "api_base" not in text, path
        assert STUB_LLM_CONFIG.api_base not in text, path

    # Every serialized llm_config is exactly {"model": ...} or null.
    def _check_llm_config(obj: object) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == "llm_config" and value is not None:
                    assert set(value.keys()) == {"model"}, value
                _check_llm_config(value)
        elif isinstance(obj, list):
            for item in obj:
                _check_llm_config(item)

    exp_dir = tmp_path / _expected_dirname(n_tasks=1)
    task_dir = iter_task_dirs(exp_dir)[0]
    task_json = load_task(task_dir)
    assert task_json["llm_config"] == {"model": "test-model"}
    for path in json_files:
        _check_llm_config(json.loads(path.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Skipped task (NotApplicable)
# ---------------------------------------------------------------------------


async def test_skipped_task_appears_in_manifest_and_result(tmp_path: Path) -> None:
    """A NotApplicable task is recorded as skipped (dir=null) and never runs."""
    tasks: list[StubTask] = [
        StubTask(goal_text="applicable"),
        NotApplicableTask(),  # type: ignore[list-item]
    ]
    result = await _make_controller(tmp_path, tasks).run()

    exp_dir = tmp_path / _expected_dirname(n_tasks=2)
    manifest = load_manifest(exp_dir)

    # Manifest lists both tasks in index order; index 2 is the skipped one.
    m_tasks = manifest["tasks"]
    assert [t["index"] for t in m_tasks] == [1, 2]
    applicable, skipped = m_tasks
    assert applicable["status"] == "success"
    assert applicable["dir"] is not None
    assert skipped["status"] == "skipped"
    assert skipped["dir"] is None

    # Summary counts the skip separately; only the run task is a "view".
    assert manifest["summary"]["n_skipped"] == 1
    assert manifest["summary"]["n_tasks"] == 1

    # Only the applicable task produced a task dir on disk.
    assert len(iter_task_dirs(exp_dir)) == 1

    # The skipped task surfaces on the returned ThreatModelResult.
    assert len(result.skipped_tasks) == 1
    assert len(result.task_results) == 1


# ---------------------------------------------------------------------------
# persist=False writes nothing
# ---------------------------------------------------------------------------


async def test_persist_false_writes_nothing(tmp_path: Path) -> None:
    """With persist=False, not a single file lands under results_dir."""
    controller = _make_controller(tmp_path, [StubTask()], persist=False)
    result = await controller.run()

    assert list(tmp_path.iterdir()) == []
    # The run still completed normally and returned a result.
    assert len(result.task_results) == 1
    assert result.task_results[0].stop_reason == "success"
