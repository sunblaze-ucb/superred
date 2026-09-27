"""Persistence for ``ThreatModelResult`` artifacts (schema v4).

Module-private writers plus a small public reader API.  Persistence is
opt-out (``Controller(persist=True)`` by default); one experiment lands in
a self-describing directory tree that a static website (and the resume
engine) can consume without globbing.

Layout (per experiment)::

    {results_root}/
      experiments.json                 # cross-experiment index (sweep landing)
      {slug}-{hash8}/                  # one experiment (one threat model)
        manifest.json                  # index: params + summary + tasks[]
        result.json                    # claim-level final aggregate (marker)
        logs/diagnostics.log           # experiment-level (unattributed) diagnostics
        tasks/
          00001__{goalslug}/           # CURRENT (latest) result for the task
            task.json                  # per-task final result + metrics
            iterations.json            # per-run score/metric progression
            logs/diagnostics.log       # this task's diagnostics (JSONL)
            trajectories/run_00001.json
        previous_01/                   # immutable complete snapshot of a prior run
          result.json  tasks/...

**Naming.** ``{slug}`` is a short human label
(``{attacker}__{target}__{claim}__{model}``); ``{hash8}`` is 8 hex chars of a
sha256 over the *measurement identity* (attacker/target/claim/model, scope,
read_only, budget, max_runs, feedback) so two distinct threat models never
collide, and a rerun of identical params resolves to the same folder (and
resumes).  The schema version is deliberately excluded from the identity so a
framework upgrade still resumes a prior run.

**Current vs history.** The latest result lives at the *direct* task path
(``tasks/00001__slug/task.json``).  On a resume that reruns errored tasks, the
prior complete state is snapshotted into the next ``previous_NN/`` before any
current file is touched.  Unchanged (kept) tasks are shared into that snapshot
by hardlink (copy fallback), so they are part of *both* the snapshot and the
current state at no extra disk cost, and prior snapshots stay immutable.

**Crash safety.** Every file is written tmp + ``os.replace``; every task dir is
published tmp-dir + ``rename``; the ``previous_NN`` snapshot is taken *before*
any current file changes, so an interrupted rerun can never destroy the only
copy of a prior result.  ``result.json`` is written last and is the completion
marker.

**Secrets.** ``LLMConfig`` serializes ``{model}`` only (``api_key``/``api_base``
never written).  Trajectory contents are NOT scrubbed: this is a red-teaming
framework and persisted trajectories contain jailbreaks, planted secrets, and
exfiltrated content.  Treat the results root as sensitive; it is gitignored.
"""

from __future__ import annotations

import hashlib
import importlib.resources
import json
import logging
import os
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from anonframework.core.types.controllable import Controllable
from anonframework.core.types.evaluation import EvaluationResult, Score
from anonframework.core.types.event import Event, EventResponse
from anonframework.core.types.events import (
    ControllableInjection,
    ControllableNoInjection,
    ControllablePostCallEvent,
    ControllablePreCallEvent,
    ObservableEvent,
    RunEndEvent,
    RunEndResponse,
)
from anonframework.core.types.llm import LLMConfig, LLMUsage
from anonframework.core.types.observable import Observable
from anonframework.core.types.security_domain import Scope, SecurityDomainTag
from anonframework.core.types.trajectory import Trajectory

if TYPE_CHECKING:
    from anonframework.core.controller import TaskResult
    from anonframework.core.interfaces.task import Task

try:  # POSIX advisory locking; absent on Windows (lock becomes a no-op there).
    import fcntl
except ImportError:  # pragma: no cover - platform-specific
    fcntl = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

# 5: ``ControllableNoInjection`` entries carry ``declined_by``. A tree at
# version 4 or lower does NOT record who declined, and the field must be
# read as UNKNOWN there, never defaulted to "optimizer": defaulting would
# silently re-commit the very conflation the field exists to prevent, by
# reporting the framework's own scope policy as attacker behaviour.
SCHEMA_VERSION = 5

_SAFE_SEGMENT_RE = re.compile(r"[^a-z0-9._-]")
_SLUG_MAXLEN = 24
_TASK_GOAL_MAXLEN = 50
_HASH_LEN = 8
_GOAL_HASH_LEN = 16

# Windows reserved device names (guarded even on POSIX so trees stay portable).
_RESERVED = {"", ".", ".."} | {"con", "prn", "aux", "nul"}
_RESERVED |= {f"com{i}" for i in range(1, 10)} | {f"lpt{i}" for i in range(1, 10)}

# Task statuses whose record holds a measurement and is therefore kept (not
# rerun) on resume.  ``"timeout"`` appears here only in its truncated form: see
# :func:`_status_of`, which splits a wall-clock cancellation into ``"timeout"``
# (at least one run completed and was judged, so the record holds a real, if
# truncated, measurement) and ``"timeout_empty"`` (nothing was measured, so the
# task must be recomputed).
_KEPT_STATUSES = frozenset({"success", "failed", "budget_exhausted", "timeout"})

DEFAULT_RESULTS_ROOT = "anonframework-results"


def is_kept(status: str, n_runs: int) -> bool:
    """Whether a persisted task is kept (not recomputed) by a resume.

    The single authority on the resume rule, for the framework and for any
    experiment driver that needs to predict what a resume will do.

    *n_runs* is the task record's run count.  It only ever matters for
    ``"timeout"``, and only to reject records written before a timed-out task
    preserved the runs it had completed: those always carry ``n_runs == 0``,
    and a zero-run record holds no measurement whatever its status says.
    """
    if status not in _KEPT_STATUSES:
        return False
    return status != "timeout" or n_runs >= 1


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------


def _slug_segment(s: str, maxlen: int = _SLUG_MAXLEN) -> str:
    """Lowercase, path-safe, truncated slug for one identity segment."""
    lowered = s.strip().lower().replace(" ", "-").replace("/", "-")
    cleaned = _SAFE_SEGMENT_RE.sub("-", lowered)
    cleaned = re.sub(r"-+", "-", cleaned).strip("-._")[:maxlen].strip("-._")
    if cleaned in _RESERVED:
        return "x"
    return cleaned or "x"


def goal_hash(goal: str) -> str:
    """Stable short content hash of a task goal (for resume identity)."""
    return hashlib.sha256(goal.encode("utf-8")).hexdigest()[:_GOAL_HASH_LEN]


def _goal_slug(goal: str) -> str:
    cleaned = _SAFE_SEGMENT_RE.sub("_", goal.strip().lower())[:_TASK_GOAL_MAXLEN].strip("_")
    return cleaned or "task"


def _task_dirname(index: int, goal: str) -> str:
    return f"{index:05d}__{_goal_slug(goal)}"


@dataclass(frozen=True)
class ExperimentMeta:
    """Identity + display parameters of one threat model (one experiment).

    The controller builds this from its labels and configuration.  It drives
    the folder name (``dirname``) and the manifest ``experiment`` block.
    """

    attacker: str
    target: str
    claim: str
    model: str | None
    scope: tuple[str, ...] = ()
    read_only: tuple[str, ...] = ()
    scope_label: str | None = None
    task_cost_cap_usd: float | None = None
    # Recorded in the output but NOT in identity_hash(): the wall-clock cap is a
    # property of the host, not of the measurement.  A truncated ("timeout")
    # task IS kept, so the reader needs to know which cap truncated it -- hence
    # it lands in experiment_block() and in every task record.
    task_time_cap_s: float | None = None
    max_runs_per_task: int = 0
    include_feedback: bool = True
    concurrency: int = 1
    n_tasks: int | None = None

    def slug(self) -> str:
        model = self.model or "no-llm"
        return "__".join(_slug_segment(s) for s in (self.attacker, self.target, self.claim, model))

    def identity_hash(self) -> str:
        """8 hex chars over the measurement identity (schema version excluded).

        ``task_time_cap_s`` is deliberately absent: it is a host property, and
        folding it in would re-key every result when the sweep moves machine.
        It is recorded in ``experiment_block()`` and in each task instead, so a
        truncated task still says which cap truncated it.
        """
        identity = {
            "attacker": self.attacker,
            "target": self.target,
            "claim": self.claim,
            "model": self.model,
            "scope": sorted(self.scope),
            "read_only": sorted(self.read_only),
            "scope_label": self.scope_label,
            "task_cost_cap_usd": self.task_cost_cap_usd,
            "max_runs_per_task": self.max_runs_per_task,
            "include_feedback": self.include_feedback,
        }
        canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:_HASH_LEN]

    def dirname(self) -> str:
        return f"{self.slug()}-{self.identity_hash()}"

    def experiment_block(self) -> dict[str, Any]:
        return {
            "slug": self.slug(),
            "hash": self.identity_hash(),
            "attacker": self.attacker,
            "target": self.target,
            "claim": self.claim,
            "model": self.model,
            "scope": sorted(self.scope),
            "read_only": sorted(self.read_only),
            "scope_label": self.scope_label,
            "task_cost_cap_usd": self.task_cost_cap_usd,
            "task_time_cap_s": self.task_time_cap_s,
            "max_runs_per_task": self.max_runs_per_task,
            "include_feedback": self.include_feedback,
            "concurrency": self.concurrency,
            "n_tasks": self.n_tasks,
        }


# ---------------------------------------------------------------------------
# Per-type serializers (trajectory item keys held byte-stable from v3)
# ---------------------------------------------------------------------------


def _tag_name(tag: SecurityDomainTag | None) -> str | None:
    return tag.name if tag is not None else None


def _sorted_names(scope: Scope) -> list[str]:
    return sorted(t.name for t in scope)


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt is not None else None


def _duration_s(started: datetime | None, ended: datetime | None) -> float | None:
    if started is None or ended is None:
        return None
    return (ended - started).total_seconds()


def _serialize_controllable(c: Controllable) -> dict[str, Any]:
    return {
        "name": c.name,
        "security_domain": c.security_domain.name,
        "description": c.description,
        "value_type": c.value_type,
    }


def _serialize_observable(o: Observable) -> dict[str, Any]:
    return {
        "name": o.name,
        "security_domain": o.security_domain.name,
        "description": o.description,
        "observable_type": o.observable_type,
    }


def _serialize_score(s: Score) -> dict[str, Any]:
    return {"name": s.name, "value": s.value, "security_domain": _tag_name(s.security_domain)}


def _serialize_evaluation(e: EvaluationResult) -> dict[str, Any]:
    return {
        "success": e.success,
        "primary_score": _serialize_score(e.primary_score),
        "sub_scores": {k: _serialize_score(v) for k, v in e.sub_scores.items()},
        "rationale": e.rationale,
    }


def _serialize_llm_config(cfg: LLMConfig | None) -> dict[str, Any] | None:
    """Allowlist serialization. Excludes ``api_key`` and ``api_base``."""
    if cfg is None:
        return None
    return {"model": cfg.model}


def _serialize_llm_usage(u: LLMUsage) -> dict[str, Any]:
    return {"calls": u.calls, "cost": u.cost}


def _event_base(event: Event) -> dict[str, Any]:
    return {
        "kind": "event",
        "type": type(event).__name__,
        "event_id": event.event_id,
        "timestamp": event.timestamp.isoformat(),
        "security_domain": _tag_name(event.security_domain),
    }


def _serialize_event(event: Event) -> dict[str, Any]:
    payload = _event_base(event)
    if isinstance(event, ControllablePreCallEvent):
        payload["controllable"] = _serialize_controllable(event.controllable)
        payload["request"] = event.request
    elif isinstance(event, ControllablePostCallEvent):
        payload["controllable"] = _serialize_controllable(event.controllable)
        payload["request"] = event.request
        payload["answer"] = event.answer
    elif isinstance(event, ObservableEvent):
        payload["observable"] = _serialize_observable(event.observable)
        payload["content"] = event.content
    elif isinstance(event, RunEndEvent):
        payload["evaluation"] = (
            _serialize_evaluation(event.evaluation) if event.evaluation is not None else None
        )
    return payload


def _serialize_response(resp: EventResponse) -> dict[str, Any]:
    payload: dict[str, Any] = {"kind": "response", "type": type(resp).__name__}
    if isinstance(resp, ControllableInjection):
        payload["controllable"] = resp.controllable.name
        payload["value"] = resp.value
    elif isinstance(resp, ControllableNoInjection):
        payload["controllable"] = resp.controllable.name
        payload["declined_by"] = resp.declined_by
    elif isinstance(resp, RunEndResponse):
        payload["done"] = resp.done
    return payload


def _serialize_trajectory(trajectory: Trajectory) -> list[dict[str, Any]]:
    return [
        _serialize_event(item) if isinstance(item, Event) else _serialize_response(item)
        for item in trajectory.snapshot()
    ]


# ---------------------------------------------------------------------------
# Per-file builders
# ---------------------------------------------------------------------------


def _status_of(success: bool, stop_reason: str, n_measured_runs: int) -> str:
    """Classify a finished task for the record, and thereby for resume.

    *n_measured_runs* is the number of runs that completed AND were judged.
    It splits the wall-clock cancellation in two, because the two halves are
    different kinds of thing:

    - ``"timeout"``: the cap cut the task short after it had already produced
      at least one judged run.  That is a real measurement, truncated -- the
      same shape as ``"budget_exhausted"``, a resource bound reached with the
      completed work retained -- so it is kept on resume.
    - ``"timeout_empty"``: the cap cut the task short with nothing judged.  The
      record holds no measurement, so it is never kept: a resume recomputes it.

    Both are held out of the ASR numerator and denominator (see
    ``_compute_summary``): a truncated task is a lower bound on what the
    attacker would have achieved, not a verdict.
    """
    if stop_reason == "error":
        return "error"
    if stop_reason == "timeout":
        return "timeout" if n_measured_runs >= 1 else "timeout_empty"
    if success:
        return "success"
    if stop_reason == "budget_exhausted":
        return "budget_exhausted"
    return "failed"


def _model_llm_config(meta: ExperimentMeta) -> LLMConfig | None:
    """Model-only LLMConfig for serialization (credentials never persisted)."""
    if meta.model is None:
        return None
    return LLMConfig(model=meta.model, api_base="", api_key="")


def _n_measured_runs(tr: TaskResult) -> int:
    """Runs that completed AND were judged (an errored partial run is neither)."""
    return sum(1 for r in tr.runs if r.evaluated)


def _build_task_json(tr: TaskResult, index: int, meta: ExperimentMeta) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "index": index,
        "goal": tr.task.goal.description,
        "goal_hash": goal_hash(tr.task.goal.description),
        "scope": _sorted_names(tr.scope),
        "read_only": _sorted_names(tr.read_only),
        "llm_config": _serialize_llm_config(_model_llm_config(meta)),
        "task_cost_cap_usd": meta.task_cost_cap_usd,
        "task_time_cap_s": meta.task_time_cap_s,
        "status": _status_of(tr.success, tr.stop_reason, _n_measured_runs(tr)),
        "success": tr.success,
        "stop_reason": tr.stop_reason,
        "best_score": _serialize_score(tr.best_score),
        "best_evaluation": _serialize_evaluation(tr.best_evaluation),
        "n_runs": len(tr.runs),
        "n_measured_runs": _n_measured_runs(tr),
        "llm_usage": _serialize_llm_usage(tr.llm_usage),
        "timing": {
            "started_at": _iso(tr.started_at),
            "ended_at": _iso(tr.ended_at),
            "duration_s": _duration_s(tr.started_at, tr.ended_at),
        },
        "error": tr.error,
    }


def _build_iterations_json(tr: TaskResult, index: int) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for i, run in enumerate(tr.runs, start=1):
        runs.append(
            {
                "run_number": i,
                "primary_score": run.evaluation.primary_score.value,
                "success": run.evaluation.success,
                "evaluated": run.evaluated,
                "errored": run.errored,
                "done": run.done,
                "usage_delta": _serialize_llm_usage(run.run_usage_delta),
                "usage_cumulative": _serialize_llm_usage(run.llm_usage),
                "timing": {
                    "started_at": _iso(run.started_at),
                    "ended_at": _iso(run.ended_at),
                    "duration_s": _duration_s(run.started_at, run.ended_at),
                },
                "trajectory": f"trajectories/run_{i:05d}.json",
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "index": index,
        "goal": tr.task.goal.description,
        "runs": runs,
    }


def _build_trajectory_json(trajectory: Trajectory, run_number: int) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_number": run_number,
        "trajectory": _serialize_trajectory(trajectory),
    }


# ---------------------------------------------------------------------------
# Summary / manifest builders
# ---------------------------------------------------------------------------


def _compute_summary(views: list[TaskView], n_skipped: int) -> dict[str, Any]:
    # A truncated task counts as a completed measurement. The per-task wall
    # clock is a threat-model parameter, not an accident of the harness: an
    # attacker that has not succeeded within its time budget has failed under
    # the threat model being measured, exactly as one that exhausted its cost
    # budget has. Both therefore enter the ASR denominator as non-successes.
    #
    # "timeout_empty" is deliberately NOT here. That is a task the cap cut short
    # with nothing judged at all, which after a full time budget is far more
    # likely a hung provider call than an attacker working to the wire -- and an
    # outage counted as attacker failure is the one error this must not make. It
    # is recomputed instead, so it can become a real measurement or a real error.
    # "success" is the reason a won task ends.  It must be here:
    # omitting it would drop every win out of the numerator AND the denominator,
    # reporting a perfect sweep as 0/0.
    completed_reasons = ("success", "done", "max_runs", "budget_exhausted")

    def _is_completed(v: TaskView) -> bool:
        """Whether this task is a measurement that belongs in the ASR.

        Both kinds of wall-clock cancellation share stop_reason "timeout", so
        only the STATUS separates them, and they belong on opposite sides:
        "timeout" was truncated with judged runs behind it, "timeout_empty" has
        nothing at all.
        """
        if v.status == "timeout":
            return True
        if v.status == "timeout_empty":
            return False
        return v.stop_reason in completed_reasons

    # Count a success only among completed tasks: a task can be success=True yet
    # stop_reason="error" (goal met, then reset_ephemeral_state failed), which
    # would otherwise make the numerator exceed the denominator (ASR > 100%).
    n_success = sum(1 for v in views if v.success and _is_completed(v))
    n_completed = sum(1 for v in views if _is_completed(v))
    n_budget = sum(1 for v in views if v.stop_reason == "budget_exhausted")
    n_error = sum(1 for v in views if v.stop_reason == "error")
    # Reported separately as well as counted above, because a truncated task is
    # a LOWER BOUND on what that attacker would have achieved: it is a real
    # failure under the time budget, but not evidence the attack cannot work
    # given more. n_timeout_empty is the subset a resume will recompute.
    n_timeout = sum(1 for v in views if v.status == "timeout")
    n_timeout_empty = sum(1 for v in views if v.status == "timeout_empty")
    scores = [v.best_score for v in views]
    return {
        "asr": (n_success / n_completed) if n_completed else None,
        "n_tasks": len(views),
        "n_success": n_success,
        "n_completed": n_completed,
        "n_failed": n_completed - n_success,
        "n_error": n_error,
        "n_budget_exhausted": n_budget,
        "n_timeout": n_timeout,
        "n_timeout_empty": n_timeout_empty,
        "n_skipped": n_skipped,
        "max_primary_score": max(scores) if scores else None,
        "mean_primary_score": (sum(scores) / len(scores)) if scores else None,
        "total_llm_usage": {
            "calls": sum(v.calls for v in views),
            "cost": sum(v.cost for v in views),
        },
    }


def _manifest_task_entry(v: TaskView) -> dict[str, Any]:
    return {
        "index": v.index,
        "goal": v.goal,
        "goal_hash": v.goal_hash,
        "dir": v.dir,
        "status": v.status,
        "success": v.success,
        "best_score": v.best_score,
        "stop_reason": v.stop_reason,
        "n_runs": v.n_runs,
        "cost_usd": v.cost,
        "has_error": v.error is not None,
        "started_at": v.started_at,
        "ended_at": v.ended_at,
    }


# ---------------------------------------------------------------------------
# Atomic filesystem primitives
# ---------------------------------------------------------------------------


def _json_fallback(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    return repr(obj)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_fallback),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _link_or_copy(src: Path, dst: Path) -> None:
    """Hardlink *src* to *dst* (immutable share), copying if link is unsupported."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _link_tree(src: Path, dst: Path) -> None:
    """Recreate *src* under *dst*, hardlinking files (copy fallback)."""
    dst.mkdir(parents=True, exist_ok=True)
    for entry in src.iterdir():
        target = dst / entry.name
        if entry.is_dir():
            _link_tree(entry, target)
        else:
            _link_or_copy(entry, target)


def _publish_dir(staging: Path, target: Path) -> None:
    """Atomically make *staging* the content at *target*.

    Safe to overwrite an existing *target*: the prior state has already been
    captured in a ``previous_NN`` snapshot before any rerun, so a crash in the
    swap window never loses the only copy.
    """
    if target.exists():
        old = target.with_name(target.name + ".old")
        if old.exists():
            shutil.rmtree(old)
        os.rename(target, old)
        os.rename(staging, target)
        shutil.rmtree(old, ignore_errors=True)
    else:
        os.rename(staging, target)


# ---------------------------------------------------------------------------
# Public reader API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskView:
    """Scalar view of one persisted task (from ``task.json``)."""

    index: int
    goal: str
    goal_hash: str
    dir: str
    status: str
    success: bool
    stop_reason: str
    best_score: float
    n_runs: int
    calls: int
    cost: float
    error: str | None
    started_at: str | None
    ended_at: str | None


def _task_view_from_json(data: dict[str, Any], dir_rel: str) -> TaskView:
    usage = data.get("llm_usage", {}) or {}
    timing = data.get("timing", {}) or {}
    score = data.get("best_score", {}) or {}
    return TaskView(
        index=int(data["index"]),
        goal=data.get("goal", ""),
        goal_hash=data.get("goal_hash", ""),
        dir=dir_rel,
        status=data.get("status", "failed"),
        success=bool(data.get("success", False)),
        stop_reason=data.get("stop_reason", "error"),
        best_score=float(score.get("value", 0.0)),
        n_runs=int(data.get("n_runs", 0)),
        calls=int(usage.get("calls", 0)),
        cost=float(usage.get("cost", 0.0)),
        error=data.get("error"),
        started_at=timing.get("started_at"),
        ended_at=timing.get("ended_at"),
    )


def _read_json(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def load_manifest(experiment_dir: str | Path) -> dict[str, Any]:
    """Load an experiment's ``manifest.json``."""
    return _read_json(Path(experiment_dir) / "manifest.json")


def load_result(experiment_dir: str | Path) -> dict[str, Any]:
    """Load an experiment's claim-level ``result.json`` (completion marker)."""
    return _read_json(Path(experiment_dir) / "result.json")


def load_task(task_dir: str | Path) -> dict[str, Any]:
    """Load a task's ``task.json``."""
    return _read_json(Path(task_dir) / "task.json")


def load_iterations(task_dir: str | Path) -> dict[str, Any]:
    """Load a task's ``iterations.json`` (per-run progression)."""
    return _read_json(Path(task_dir) / "iterations.json")


def load_trajectory(task_dir: str | Path, run_number: int) -> dict[str, Any]:
    """Load one run's trajectory file for a task."""
    return _read_json(Path(task_dir) / "trajectories" / f"run_{run_number:05d}.json")


def iter_task_dirs(experiment_dir: str | Path) -> list[Path]:
    """Task directories under an experiment, ordered by name (== index)."""
    tasks = Path(experiment_dir) / "tasks"
    if not tasks.is_dir():
        return []
    return sorted(d for d in tasks.iterdir() if d.is_dir() and (d / "task.json").exists())


def iter_tasks(experiment_dir: str | Path) -> list[TaskView]:
    """Scalar views of all current tasks under an experiment, ordered by index.

    A task record that cannot be read is SKIPPED, which is what
    :func:`_scan_prior_tasks` already does when it plans the resume: the index is
    simply absent, so that task counts as outstanding and is recomputed.  Raising
    instead would make one damaged file fatal to the entire experiment, forever:
    ``ExperimentSession.open`` calls this before any task runs, so the experiment
    could never start again to repair itself.
    """
    views = []
    for d in iter_task_dirs(experiment_dir):
        try:
            views.append(_task_view_from_json(load_task(d), f"tasks/{d.name}"))
        except Exception:
            logger.warning(
                "anonframework: unreadable task record at %s, treating that task as outstanding", d
            )
    return sorted(views, key=lambda v: v.index)


def load_experiments_index(results_root: str | Path) -> dict[str, Any]:
    """Load the cross-experiment ``experiments.json`` at a results root."""
    return _read_json(Path(results_root) / "experiments.json")


# ---------------------------------------------------------------------------
# Resume planning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResumePlan:
    """Which tasks to rerun vs keep when an experiment dir already exists.

    Attributes:
        keep: 1-based indices whose prior result is a valid measurement and is
            kept (not rerun); their current task dirs remain on disk.
        rerun: 1-based indices to (re)compute (error/interrupted/missing, or
            all under ``overwrite``).
        is_fresh: ``True`` when there is no prior experiment on disk.
    """

    keep: frozenset[int]
    rerun: frozenset[int]
    is_fresh: bool


def _scan_prior_tasks(experiment_dir: Path) -> dict[int, tuple[str, str, int]]:
    """Map index -> (goal_hash, status, n_runs) for prior current task dirs."""
    prior: dict[int, tuple[str, str, int]] = {}
    for d in iter_task_dirs(experiment_dir):
        try:
            data = load_task(d)
            prior[int(data["index"])] = (
                data.get("goal_hash", ""),
                data.get("status", "error"),
                int(data.get("n_runs", 0) or 0),
            )
        except Exception:
            continue
    return prior


def plan_resume(experiment_dir: Path, goals: list[str], overwrite: bool) -> ResumePlan:
    """Decide keep vs rerun for the live claim against the on-disk experiment.

    A live task at position ``i`` (1-based) is KEPT iff a prior current task at
    the same index has a matching ``goal_hash`` and a record that holds a
    measurement (see :func:`is_kept`: success / failed / budget_exhausted, plus
    a ``timeout`` that retained at least one run).  Everything else reruns --
    including ``timeout_empty``, a task the wall-clock cap cancelled before
    anything was judged.  Under ``overwrite`` every task reruns.  Appending
    tasks to a claim resumes (the new tasks are missing -> rerun); reordering
    existing tasks reruns them (index no longer matches) but never reuses a
    wrong result.
    """
    indices = list(range(1, len(goals) + 1))
    fresh = (
        not (experiment_dir / "manifest.json").exists() and not (experiment_dir / "tasks").exists()
    )
    if fresh or overwrite:
        return ResumePlan(keep=frozenset(), rerun=frozenset(indices), is_fresh=fresh)
    prior = _scan_prior_tasks(experiment_dir)
    keep: set[int] = set()
    for i, g in zip(indices, goals, strict=True):
        entry = prior.get(i)
        if entry is not None and entry[0] == goal_hash(g) and is_kept(entry[1], entry[2]):
            keep.add(i)
    rerun = frozenset(i for i in indices if i not in keep)
    return ResumePlan(keep=frozenset(keep), rerun=rerun, is_fresh=False)


# ---------------------------------------------------------------------------
# Snapshots + experiments index
# ---------------------------------------------------------------------------


def _next_previous_index(experiment_dir: Path) -> int:
    n = 0
    for d in experiment_dir.glob("previous_*"):
        m = re.fullmatch(r"previous_(\d+)", d.name)
        if m and d.is_dir():
            n = max(n, int(m.group(1)))
    return n + 1


def snapshot_current(experiment_dir: Path) -> Path | None:
    """Snapshot the current experiment state into the next ``previous_NN/``.

    Hardlinks (copy fallback) ``result.json``, ``manifest.json``, ``logs/`` and
    ``tasks/`` so kept tasks are shared into the immutable snapshot at no extra
    disk cost.  Returns the snapshot path, or ``None`` if there is nothing to
    snapshot.  Written to a ``.wip`` dir then renamed so a crash leaves only a
    stray ``.wip`` to discard.
    """
    have = [
        p
        for p in (
            experiment_dir / "result.json",
            experiment_dir / "manifest.json",
            experiment_dir / "logs",
            experiment_dir / "tasks",
        )
        if p.exists()
    ]
    if not have:
        return None
    n = _next_previous_index(experiment_dir)
    final = experiment_dir / f"previous_{n:02d}"
    staging = experiment_dir / f"previous_{n:02d}.wip"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for p in have:
        if p.name == "logs" and p.is_dir():
            # The experiment-level diagnostics log is appended in place during a
            # resume (open("a")), so hardlinking it would mutate the snapshot
            # through the shared inode. Copy it (small vs trajectories). Task
            # dirs are safe to hardlink: a rerun republishes a fresh inode and
            # never writes back into the old one.
            shutil.copytree(p, staging / p.name)
        elif p.is_dir():
            _link_tree(p, staging / p.name)
        else:
            _link_or_copy(p, staging / p.name)
    os.rename(staging, final)
    return final


def _stale_task_dirs(experiment_dir: Path, goals: list[str]) -> list[Path]:
    """Current task dirs that do not belong to the live claim.

    On a resume against an edited / reordered / removed goal, the prior run's
    dir for the old ``(index, goal)`` survives under a different name than the
    live claim's :func:`_task_dirname`, and nothing else prunes it, so
    :func:`iter_tasks` would count the orphan and inflate ``result.json`` plus
    the ``experiments.json`` sweep row.  Reserved staging suffixes
    (``.wip`` / ``.old``) are excluded (handled by :func:`_gc_staging`).
    """
    tasks = experiment_dir / "tasks"
    if not tasks.is_dir():
        return []
    valid = {_task_dirname(i, g) for i, g in enumerate(goals, start=1)}
    return [
        d
        for d in tasks.iterdir()
        if d.is_dir() and d.name not in valid and not d.name.endswith((".wip", ".old"))
    ]


def update_experiments_index(
    results_root: Path,
    meta: ExperimentMeta,
    summary: dict[str, Any],
    status: str,
    completed_at: str | None,
) -> None:
    """Append/update this experiment's row in ``experiments.json`` (atomic).

    The read-modify-write is guarded by a root-level ``.experiments.lock`` flock
    so two processes finalizing *different* experiments into the same results
    root cannot clobber each other's row (nor collide on the shared temp file).
    """
    path = results_root / "experiments.json"
    lock_fh = None
    if fcntl is not None:
        lock_fh = open(results_root / ".experiments.lock", "w")
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
    try:
        try:
            rows = _read_json(path).get("experiments", [])
        except Exception:
            rows = []
        rows = [r for r in rows if r.get("dir") != meta.dirname()]
        rows.append(
            {
                "dir": meta.dirname(),
                "slug": meta.slug(),
                "hash": meta.identity_hash(),
                "status": status,
                "experiment": meta.experiment_block(),
                "summary": {
                    "asr": summary.get("asr"),
                    "n_success": summary.get("n_success"),
                    "n_completed": summary.get("n_completed"),
                    "total_llm_usage": summary.get("total_llm_usage"),
                },
                "completed_at": completed_at,
            }
        )
        rows.sort(key=lambda r: r.get("dir", ""))
        _atomic_write_json(path, {"schema_version": SCHEMA_VERSION, "experiments": rows})
    finally:
        if lock_fh is not None:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
            lock_fh.close()


def write_dashboard(dest_dir: Path) -> None:
    """Drop the bundled static results dashboard into ``dest_dir/dashboard.html``.

    The dashboard is a single self-contained page that fetches the JSON in its
    own directory: at a results root it reads ``experiments.json`` (the sweep
    index); inside an experiment dir it reads ``manifest.json`` and drills into
    the per-task files.  Best-effort: a missing asset or write error never
    breaks a run (the JSON is the source of truth; the page is a convenience).
    """
    try:
        data = (importlib.resources.files("anonframework.core") / "dashboard.html").read_bytes()
        (dest_dir / "dashboard.html").write_bytes(data)
    except Exception:  # pragma: no cover - best-effort convenience asset
        pass


# ---------------------------------------------------------------------------
# Locking
# ---------------------------------------------------------------------------


class _ExperimentLock:
    """Non-blocking advisory lock (flock) on an experiment dir's ``.lock``."""

    def __init__(self, experiment_dir: Path) -> None:
        self._path = experiment_dir / ".lock"
        self._fh: Any = None

    def acquire(self) -> None:
        if fcntl is None:  # pragma: no cover - platform-specific
            return
        self._fh = open(self._path, "w")
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._fh.close()
            self._fh = None
            raise RuntimeError(f"experiment {self._path.parent} is locked by another run") from exc

    def release(self) -> None:
        if self._fh is not None and fcntl is not None:
            try:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            finally:
                self._fh.close()
                self._fh = None


# ---------------------------------------------------------------------------
# Experiment session (writer)
# ---------------------------------------------------------------------------


def resolve_results_root(results_dir: str | Path | None) -> Path:
    """Resolve the results root for on-by-default persistence.

    ``results_dir`` set -> used verbatim.  Otherwise ``ANONFRAMEWORK_RESULTS_DIR`` if
    set, else ``./anonframework-results/``.
    """
    if results_dir is not None:
        return Path(results_dir)
    return Path(os.environ.get("ANONFRAMEWORK_RESULTS_DIR", DEFAULT_RESULTS_ROOT))


def _gc_staging(experiment_dir: Path) -> None:
    """Remove stray ``*.wip`` / ``*.old`` dirs left by a prior crash."""
    globs = list(experiment_dir.glob("*.wip"))
    tasks = experiment_dir / "tasks"
    if tasks.is_dir():
        globs += list(tasks.glob("*.wip")) + list(tasks.glob("*.old"))
    for d in globs:
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)


class ExperimentSession:
    """Owns one experiment's on-disk lifecycle: prepare -> per-task -> finalize.

    Created by :meth:`open` (which resolves the dir, locks it, plans the
    resume, and snapshots any prior state to ``previous_NN`` before a rerun).
    Per reran task the controller calls :meth:`begin_task` (staging dir + logs
    sink), then :meth:`publish_task`; skipped tasks use :meth:`mark_skipped`;
    the run ends with :meth:`finalize` (or :meth:`abort`).
    """

    def __init__(
        self,
        results_root: Path,
        experiment_dir: Path,
        meta: ExperimentMeta,
        plan: ResumePlan,
        lock: _ExperimentLock,
    ) -> None:
        self.results_root = results_root
        self.experiment_dir = experiment_dir
        self.meta = meta
        self.plan = plan
        self._lock = lock
        self._staging: dict[int, Path] = {}
        self._skipped: list[tuple[int, str]] = []

    @classmethod
    def open(
        cls,
        results_root: str | Path,
        meta: ExperimentMeta,
        goals: list[str],
        *,
        overwrite: bool = False,
    ) -> ExperimentSession:
        root = Path(results_root)
        experiment_dir = root / meta.dirname()
        experiment_dir.mkdir(parents=True, exist_ok=True)
        lock = _ExperimentLock(experiment_dir)
        lock.acquire()
        try:
            _gc_staging(experiment_dir)
            plan = plan_resume(experiment_dir, goals, overwrite)
            # Snapshot prior state before touching current, then reconcile
            # current with the live claim.  A rerun republishes task dirs, and a
            # resume against an edited / reordered / shrunk claim leaves prior
            # dirs whose (index, goal) no longer matches under a stale name that
            # nothing else prunes (so iter_tasks would double-count and inflate
            # result.json + the experiments.json row).  Both mutate current, so
            # snapshot when either applies (history stays complete), then drop
            # the orphans.
            if not plan.is_fresh:
                stale = _stale_task_dirs(experiment_dir, goals)
                if plan.rerun or stale:
                    snapshot_current(experiment_dir)
                for d in stale:
                    shutil.rmtree(d, ignore_errors=True)
            (experiment_dir / "tasks").mkdir(exist_ok=True)
            (experiment_dir / "logs").mkdir(exist_ok=True)
            session = cls(root, experiment_dir, meta, plan, lock)
            session._write_manifest(status="in_progress", completed_at=None)
            return session
        except Exception:
            lock.release()
            raise

    # -- Per-task ----------------------------------------------------------

    def experiment_log_path(self) -> Path:
        return self.experiment_dir / "logs" / "diagnostics.log"

    def task_log_path(self, index: int | None) -> Path:
        """Where a diagnostic lands: the task's staging log, else the experiment log."""
        if index is not None:
            staging = self._staging.get(index)
            if staging is not None:
                return staging / "logs" / "diagnostics.log"
        return self.experiment_log_path()

    def begin_task(self, index: int, goal: str) -> Path:
        """Create a fresh staging dir (with ``logs/`` + ``trajectories/``).

        The returned dir's ``logs/diagnostics.log`` is where the logging bridge
        writes this task's diagnostics during the run; it is published into the
        current task dir by :meth:`publish_task`.
        """
        staging = self.experiment_dir / "tasks" / f"{_task_dirname(index, goal)}.wip"
        if staging.exists():
            shutil.rmtree(staging)
        (staging / "logs").mkdir(parents=True)
        (staging / "trajectories").mkdir()
        self._staging[index] = staging
        return staging

    def publish_task(self, index: int, tr: TaskResult) -> None:
        """Write the task result into its staging dir and publish it to current."""
        staging = self._staging.get(index)
        if staging is None:
            staging = self.begin_task(index, tr.task.goal.description)
        _atomic_write_json(staging / "task.json", _build_task_json(tr, index, self.meta))
        _atomic_write_json(staging / "iterations.json", _build_iterations_json(tr, index))
        for i, run in enumerate(tr.runs, start=1):
            _atomic_write_json(
                staging / "trajectories" / f"run_{i:05d}.json",
                _build_trajectory_json(run.trajectory, i),
            )
        current = self.experiment_dir / "tasks" / _task_dirname(index, tr.task.goal.description)
        _publish_dir(staging, current)
        self._staging.pop(index, None)
        self._write_manifest(status="in_progress", completed_at=None)

    def mark_skipped(self, index: int, goal: str) -> None:
        self._skipped.append((index, goal))
        # A skipped task's ``begin_task`` staging dir is never published; drop it
        # so a skip leaves no stray ``*.wip`` in the tree.
        staging = self._staging.pop(index, None)
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

    # -- Finalize ----------------------------------------------------------

    def finalize(self, started_at: datetime | None, ended_at: datetime | None) -> Path:
        """Write ``result.json`` (completion marker) + final manifest + index."""
        try:
            views = iter_tasks(self.experiment_dir)
            summary = _compute_summary(views, n_skipped=len(self._skipped))
            completed = _iso(ended_at) or datetime.now(UTC).isoformat()
            result = {
                "schema_version": SCHEMA_VERSION,
                "experiment": self.meta.experiment_block(),
                "timing": {"started_at": _iso(started_at), "completed_at": completed},
                "summary": summary,
            }
            _atomic_write_json(self.experiment_dir / "result.json", result)
            self._write_manifest(
                status="complete",
                completed_at=completed,
                views=views,
                summary=summary,
                started_at=started_at,
            )
            update_experiments_index(self.results_root, self.meta, summary, "complete", completed)
            # Drop the static browser dashboard next to the JSON it reads (both
            # the experiment view and the sweep index at the root).
            write_dashboard(self.experiment_dir)
            write_dashboard(self.results_root)
            return self.experiment_dir / "result.json"
        finally:
            self._lock.release()

    def abort(self) -> None:
        """Release the lock without finalizing (interrupted run)."""
        self._lock.release()

    # -- Manifest ----------------------------------------------------------

    def _write_manifest(
        self,
        *,
        status: str,
        completed_at: str | None,
        views: list[TaskView] | None = None,
        summary: dict[str, Any] | None = None,
        started_at: datetime | None = None,
    ) -> None:
        if views is None:
            views = iter_tasks(self.experiment_dir)
        if summary is None:
            summary = _compute_summary(views, n_skipped=len(self._skipped))
        task_entries = [_manifest_task_entry(v) for v in views]
        for index, goal in self._skipped:
            task_entries.append(
                {
                    "index": index,
                    "goal": goal,
                    "goal_hash": goal_hash(goal),
                    "dir": None,
                    "status": "skipped",
                    "success": False,
                    "best_score": 0.0,
                    "stop_reason": "skipped",
                    "n_runs": 0,
                    "cost_usd": 0.0,
                    "has_error": False,
                    "started_at": None,
                    "ended_at": None,
                }
            )
        task_entries.sort(key=lambda e: e["index"])
        _atomic_write_json(
            self.experiment_dir / "manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": status,
                "experiment": self.meta.experiment_block(),
                "timing": {"started_at": _iso(started_at), "completed_at": completed_at},
                "summary": summary,
                "tasks": task_entries,
            },
        )


# ---------------------------------------------------------------------------
# Kept-task reconstruction (for the returned ThreatModelResult)
# ---------------------------------------------------------------------------


def reconstruct_kept_task_result(experiment_dir: Path, index: int, task: Task[Any]) -> TaskResult:
    """Rebuild a lightweight :class:`TaskResult` for a kept task from disk.

    Scalar metrics are faithful; the run list is empty (the full per-run
    trajectories remain on disk and are not re-loaded into memory).  Used to
    include kept tasks in the returned ``ThreatModelResult`` and the printed
    summary without rewriting or re-reading their trajectories.
    """
    from anonframework.core.controller import TaskResult as _TaskResult

    for d in iter_task_dirs(experiment_dir):
        try:
            data = load_task(d)
        except Exception:
            # A damaged neighbouring record must not hide the one we want.
            continue
        if int(data.get("index", -1)) != index:
            continue
        score = data.get("best_score", {}) or {}
        best = Score(value=float(score.get("value", 0.0)), name=score.get("name", "primary"))
        eval_data = data.get("best_evaluation") or {}
        best_eval = EvaluationResult(
            success=bool(eval_data.get("success", data.get("success", False))),
            primary_score=best,
            sub_scores={},
            rationale=(eval_data.get("rationale", "") or ""),
        )
        usage = data.get("llm_usage", {}) or {}
        timing = data.get("timing", {}) or {}
        return _TaskResult(
            task=task,
            runs=[],
            best_score=best,
            best_evaluation=best_eval,
            success=bool(data.get("success", False)),
            llm_usage=LLMUsage(
                calls=int(usage.get("calls", 0)), cost=float(usage.get("cost", 0.0))
            ),
            stop_reason=data.get("stop_reason", "error"),
            error=data.get("error"),
            started_at=_parse_iso(timing.get("started_at")),
            ended_at=_parse_iso(timing.get("ended_at")),
            # runs=[] is intentional (trajectories stay on disk); carry the real
            # count so the reporter + returned result match a fresh run.
            n_runs=int(data.get("n_runs", 0)),
        )
    raise FileNotFoundError(f"no persisted task with index {index} in {experiment_dir}")


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None
