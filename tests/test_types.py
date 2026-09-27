"""Unit tests for core value types: Goal, Score, EvaluationResult,
Controllable, Observable, ObservableValue, ConfigSpec, QuerySpec, QueryParam,
and Event hierarchy.

Each test verifies a real behavioral contract of the type system.
"""

from __future__ import annotations

import uuid
from dataclasses import FrozenInstanceError
from datetime import datetime

import pytest

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
    RunStartEvent,
)
from anonframework.core.types.goal import Goal
from anonframework.core.types.observable import Observable, ObservableValue
from anonframework.core.types.security_domain import SecurityDomainTag
from anonframework.core.types.state import ConfigSpec, QueryParam, QuerySpec
from anonframework.core.types.trajectory import Trajectory

# ---------------------------------------------------------------------------
# Goal
# ---------------------------------------------------------------------------


class TestGoal:
    def test_construction_requires_description(self) -> None:
        g = Goal(description="Extract secret")
        assert g.description == "Extract secret"

    def test_frozen(self) -> None:
        g = Goal(description="test")
        with pytest.raises(FrozenInstanceError):
            g.description = "other"  # type: ignore[misc]

    def test_equality(self) -> None:
        assert Goal(description="a") == Goal(description="a")
        assert Goal(description="a") != Goal(description="b")


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------


class TestScore:
    def test_construction(self) -> None:
        tag = SecurityDomainTag("ext")
        s = Score(value=0.5, security_domain=tag)
        assert s.value == 0.5
        assert s.security_domain is tag
        assert s.name == "primary"

    def test_custom_name(self) -> None:
        tag = SecurityDomainTag("ext")
        s = Score(value=1.0, security_domain=tag, name="asr")
        assert s.name == "asr"

    def test_frozen(self) -> None:
        tag = SecurityDomainTag("ext")
        s = Score(value=0.5, security_domain=tag)
        with pytest.raises(FrozenInstanceError):
            s.value = 1.0  # type: ignore[misc]

    @pytest.mark.parametrize("value", [0.0, -1.0, 1.0, float("inf"), float("-inf")])
    def test_accepts_any_float(self, value: float) -> None:
        """Score does not constrain value range -- that's the evaluator's job."""
        tag = SecurityDomainTag("ext")
        s = Score(value=value, security_domain=tag)
        assert s.value == value

    def test_security_domain_defaults_to_none(self) -> None:
        s = Score(value=0.5)
        assert s.security_domain is None


# ---------------------------------------------------------------------------
# EvaluationResult
# ---------------------------------------------------------------------------


class TestEvaluationResult:
    def test_required_fields(self) -> None:
        er = EvaluationResult(
            success=True,
            primary_score=Score(value=1.0),
        )
        assert er.success is True
        assert er.primary_score.value == 1.0
        assert er.sub_scores == {}
        assert er.rationale == ""

    def test_with_sub_scores_and_rationale(self) -> None:
        tag = SecurityDomainTag("ext")
        sub = {"asr": Score(value=0.9, security_domain=tag, name="asr")}
        er = EvaluationResult(
            success=False,
            primary_score=Score(value=0.5),
            sub_scores=sub,
            rationale="Partial extraction",
        )
        assert er.sub_scores["asr"].value == 0.9
        assert er.rationale == "Partial extraction"

    def test_frozen(self) -> None:
        er = EvaluationResult(
            success=True,
            primary_score=Score(value=1.0),
        )
        with pytest.raises(FrozenInstanceError):
            er.success = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ObservableEvent
# ---------------------------------------------------------------------------


class TestObservableEvent:
    def test_fields(self) -> None:
        tag = SecurityDomainTag("ext")
        obs = Observable(name="model_request", security_domain=tag, description="User message")
        e = ObservableEvent(observable=obs, content="hello")
        assert e.observable is obs
        assert e.content == "hello"
        assert e.security_domain is tag
        assert isinstance(e, Event)

    def test_frozen(self) -> None:
        tag = SecurityDomainTag("ext")
        obs = Observable(name="x", security_domain=tag)
        e = ObservableEvent(observable=obs, content="hello")
        with pytest.raises(FrozenInstanceError):
            e.content = "other"  # type: ignore[misc]

    def test_security_domain_auto_derived(self) -> None:
        tag = SecurityDomainTag("ext")
        obs = Observable(name="x", security_domain=tag)
        e = ObservableEvent(observable=obs, content="hello")
        assert e.security_domain is tag


# ---------------------------------------------------------------------------
# Controllable
# ---------------------------------------------------------------------------


class TestControllable:
    def test_construction(self) -> None:
        tag = SecurityDomainTag("ext")
        c = Controllable(name="input", security_domain=tag)
        assert c.name == "input"
        assert c.security_domain is tag
        assert c.description == ""
        assert c.value_type == "text"

    def test_frozen(self) -> None:
        tag = SecurityDomainTag("ext")
        c = Controllable(name="input", security_domain=tag)
        with pytest.raises(FrozenInstanceError):
            c.name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Observable / ObservableValue
# ---------------------------------------------------------------------------


class TestObservable:
    def test_construction_defaults(self) -> None:
        tag = SecurityDomainTag("ext")
        obs = Observable(name="sys_desc", security_domain=tag)
        assert obs.name == "sys_desc"
        assert obs.description == ""
        assert obs.observable_type == "text"

    def test_frozen(self) -> None:
        tag = SecurityDomainTag("ext")
        obs = Observable(name="sys_desc", security_domain=tag)
        with pytest.raises(FrozenInstanceError):
            obs.name = "other"  # type: ignore[misc]


class TestObservableValue:
    def test_wraps_observable_with_content(self) -> None:
        tag = SecurityDomainTag("ext")
        obs = Observable(name="code", security_domain=tag)
        ov = ObservableValue(observable=obs, content="print('hello')")
        assert ov.observable is obs
        assert ov.content == "print('hello')"

    def test_default_content_is_none(self) -> None:
        tag = SecurityDomainTag("ext")
        obs = Observable(name="code", security_domain=tag)
        ov = ObservableValue(observable=obs)
        assert ov.content is None


# ---------------------------------------------------------------------------
# ConfigSpec / QuerySpec / QueryParam
# ---------------------------------------------------------------------------


class TestConfigSpec:
    def test_construction(self) -> None:
        tag = SecurityDomainTag("ext")
        cs = ConfigSpec(name="db_seed", security_domain=tag, description="SQL seed data")
        assert cs.name == "db_seed"
        assert cs.security_domain is tag
        assert cs.description == "SQL seed data"

    def test_frozen(self) -> None:
        tag = SecurityDomainTag("ext")
        cs = ConfigSpec(name="db_seed", security_domain=tag, description="desc")
        with pytest.raises(FrozenInstanceError):
            cs.name = "other"  # type: ignore[misc]


class TestQueryParam:
    def test_construction(self) -> None:
        qp = QueryParam(name="limit", description="Max results")
        assert qp.name == "limit"
        assert qp.description == "Max results"


class TestQuerySpec:
    def test_construction_with_defaults(self) -> None:
        qs = QuerySpec(name="get_logs", description="Retrieve audit logs")
        assert qs.name == "get_logs"
        assert qs.params == []

    def test_construction_with_params(self) -> None:
        p = QueryParam(name="limit", description="Max results")
        qs = QuerySpec(name="get_logs", description="desc", params=[p])
        assert len(qs.params) == 1
        assert qs.params[0] is p


# ---------------------------------------------------------------------------
# Event hierarchy
# ---------------------------------------------------------------------------


class TestEvent:
    def test_auto_fields(self) -> None:
        """Event auto-generates event_id (UUID) and timestamp."""
        e = Event()
        uuid.UUID(e.event_id)  # validates format, raises if invalid
        assert isinstance(e.timestamp, datetime)

    def test_unique_ids(self) -> None:
        """Each Event instance gets a unique event_id."""
        e1 = Event()
        e2 = Event()
        assert e1.event_id != e2.event_id

    def test_frozen(self) -> None:
        e = Event()
        with pytest.raises(FrozenInstanceError):
            e.event_id = "fake"  # type: ignore[misc]


class TestEventResponse:
    def test_references_event(self) -> None:
        e = Event()
        r = EventResponse(event=e)
        assert r.event is e

    def test_frozen(self) -> None:
        e = Event()
        r = EventResponse(event=e)
        with pytest.raises(FrozenInstanceError):
            r.event = Event()  # type: ignore[misc]


class TestControllableEvents:
    def test_pre_call_event_fields(self) -> None:
        tag = SecurityDomainTag("ext")
        c = Controllable(name="input", security_domain=tag)
        e = ControllablePreCallEvent(controllable=c, request="hello")
        assert e.controllable is c
        assert e.request == "hello"
        assert isinstance(e, Event)

    def test_post_call_event_fields(self) -> None:
        tag = SecurityDomainTag("ext")
        c = Controllable(name="input", security_domain=tag)
        e = ControllablePostCallEvent(controllable=c, request="hello", answer="world")
        assert e.answer == "world"
        assert isinstance(e, Event)

    def test_injection_response(self) -> None:
        tag = SecurityDomainTag("ext")
        c = Controllable(name="input", security_domain=tag)
        e = ControllablePreCallEvent(controllable=c, request="hello")
        inj = ControllableInjection(event=e, controllable=c, value="payload")
        assert inj.value == "payload"
        assert inj.controllable is c
        assert inj.event is e
        assert isinstance(inj, EventResponse)

    def test_no_injection_response(self) -> None:
        tag = SecurityDomainTag("ext")
        c = Controllable(name="input", security_domain=tag)
        e = ControllablePreCallEvent(controllable=c, request="hello")
        nm = ControllableNoInjection(event=e, controllable=c)
        assert nm.controllable is c
        assert nm.event is e
        assert isinstance(nm, EventResponse)

    def test_pre_call_frozen(self) -> None:
        tag = SecurityDomainTag("ext")
        c = Controllable(name="input", security_domain=tag)
        e = ControllablePreCallEvent(controllable=c, request="hello")
        with pytest.raises(FrozenInstanceError):
            e.request = "other"  # type: ignore[misc]

    def test_post_call_frozen(self) -> None:
        tag = SecurityDomainTag("ext")
        c = Controllable(name="input", security_domain=tag)
        e = ControllablePostCallEvent(controllable=c, request="hello", answer="world")
        with pytest.raises(FrozenInstanceError):
            e.answer = "other"  # type: ignore[misc]

    def test_injection_frozen(self) -> None:
        tag = SecurityDomainTag("ext")
        c = Controllable(name="input", security_domain=tag)
        e = ControllablePreCallEvent(controllable=c, request="hello")
        inj = ControllableInjection(event=e, controllable=c, value="x")
        with pytest.raises(FrozenInstanceError):
            inj.value = "other"  # type: ignore[misc]

    def test_no_injection_frozen(self) -> None:
        tag = SecurityDomainTag("ext")
        c = Controllable(name="input", security_domain=tag)
        e = ControllablePreCallEvent(controllable=c, request="hello")
        nm = ControllableNoInjection(event=e, controllable=c)
        with pytest.raises(FrozenInstanceError):
            nm.controllable = c  # type: ignore[misc]


class TestRunLifecycleEvents:
    def test_run_start_event(self) -> None:
        t = Trajectory()
        e = RunStartEvent(trajectory=t)
        assert e.trajectory is t
        assert isinstance(e, Event)

    def test_run_end_event(self) -> None:
        e = RunEndEvent()
        assert e.evaluation is None

    def test_run_end_event_with_evaluation(self) -> None:
        from anonframework.core.types.evaluation import EvaluationResult, Score

        ev = EvaluationResult(success=False, primary_score=Score(0.5))
        e = RunEndEvent(evaluation=ev)
        assert e.evaluation is ev

    def test_run_end_response_default_not_done(self) -> None:
        e = RunEndEvent()
        r = RunEndResponse(event=e, done=False)
        assert r.done is False

    def test_run_end_response_done(self) -> None:
        e = RunEndEvent()
        r = RunEndResponse(event=e, done=True)
        assert r.done is True

    def test_run_start_frozen(self) -> None:
        t = Trajectory()
        e = RunStartEvent(trajectory=t)
        with pytest.raises(FrozenInstanceError):
            e.trajectory = Trajectory()  # type: ignore[misc]

    def test_run_end_frozen(self) -> None:
        e = RunEndEvent()
        with pytest.raises(FrozenInstanceError):
            e.evaluation = None  # type: ignore[misc]

    def test_run_end_response_frozen(self) -> None:
        e = RunEndEvent()
        r = RunEndResponse(event=e, done=False)
        with pytest.raises(FrozenInstanceError):
            r.done = True  # type: ignore[misc]
