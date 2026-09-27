"""Evaluation types produced by task modules.

Feedback originates from the task module's evaluator and flows through the
controller to the optimizer. It includes a primary score for optimization
and optional sub-scores for multi-objective analysis (e.g. Pareto frontiers).

Two orthogonal controls govern feedback: (1) the controller ``include_feedback``
flag enables or disables feedback as a whole (``RunEndEvent.evaluation`` is the
full :class:`EvaluationResult` or ``None``); (2) scope filtering prunes only
``sub_scores``, and only those carrying an out-of-scope ``security_domain`` (an
untagged sub-score, ``security_domain=None``, is always visible). The
``primary_score`` is never scope-filtered and carries no ``security_domain``: it
is the unscoped optimization signal always delivered to the optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from anonframework.core.types.security_domain import SecurityDomainTag


@dataclass(frozen=True)
class Score:
    """A single evaluation score.

    Attributes:
        value: Numeric score. Meaning is relative/comparative only. Higher values are better.
        name: Name of this score dimension (e.g. "asr", "utility_degradation").
        security_domain: The security domain this score pertains to. The tag
            scopes ``sub_scores`` and must be ``None`` on a ``primary_score``.
            ``None`` means always visible (unscoped).
    """

    value: float
    security_domain: SecurityDomainTag | None = None
    name: str = "primary"


@dataclass(frozen=True)
class EvaluationResult:
    """The result of evaluating a single run against a task.

    Attributes:
        success: Whether the adversarial goal was achieved (binary).
        primary_score: The main optimization score, always delivered to the
            optimizer (subject only to ``include_feedback``). It MUST be
            unscoped: ``security_domain`` must be ``None`` (attach domains to
            ``sub_scores`` instead).
        sub_scores: Named sub-scores for multi-objective analysis, keyed by
            what each score evaluates (e.g. ``{"asr": Score(...), ...}``).
            Each score carries its own ``security_domain``.
        rationale: Optional free-text explanation from the evaluator.
    """

    success: bool
    primary_score: Score
    sub_scores: dict[str, Score] = field(default_factory=dict)
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.primary_score.security_domain is not None:
            raise ValueError(
                "primary_score must not carry a security_domain "
                f"(got {self.primary_score.security_domain!r}). The primary score is the "
                "unscoped optimization signal always delivered to the optimizer; attach a "
                "security_domain to sub_scores instead."
            )
