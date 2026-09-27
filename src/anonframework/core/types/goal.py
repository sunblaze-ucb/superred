"""Goal specification for the optimizer in the review artifact."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Goal:
    """The adversarial goal for an optimization run.

    Attributes:
        description: Free-text description of the goal.
    """

    description: str
