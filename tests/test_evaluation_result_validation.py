"""Validation tests for the ``EvaluationResult`` primary-score guard.

A ``primary_score`` is the optimizer-facing, scope-spanning score for a run.
Tagging it with a ``security_domain`` is meaningless (it is never filtered by
scope) and signals a modelling mistake, so construction must reject it. The
guard applies only to ``primary_score``: ``sub_scores`` tags are load-bearing
(the controller filters them by scope) and must be preserved untouched.
"""

from __future__ import annotations

import pytest

from anonframework.core.types.evaluation import EvaluationResult, Score
from anonframework.core.types.security_domain import SecurityDomainTag


class TestPrimaryScoreDomainGuard:
    def test_tagged_primary_score_rejected(self) -> None:
        """A ``primary_score`` carrying a ``security_domain`` raises ValueError."""
        tag = SecurityDomainTag("ext")
        with pytest.raises(ValueError, match="primary_score"):
            EvaluationResult(
                success=True,
                primary_score=Score(value=1.0, security_domain=tag),
            )

    def test_untagged_primary_score_accepted(self) -> None:
        """An untagged ``primary_score`` (the default) constructs cleanly."""
        er = EvaluationResult(
            success=True,
            primary_score=Score(value=1.0),
        )
        assert er.primary_score.security_domain is None
        assert er.primary_score.value == 1.0

    def test_sub_score_may_keep_security_domain(self) -> None:
        """The guard leaves ``sub_scores`` tags intact; they round-trip."""
        tag = SecurityDomainTag("ext")
        er = EvaluationResult(
            success=False,
            primary_score=Score(value=0.5),
            sub_scores={"x": Score(value=0.9, security_domain=tag, name="x")},
        )
        assert er.sub_scores["x"].security_domain is tag
        assert er.sub_scores["x"].value == 0.9
