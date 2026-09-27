"""Unit tests for SecurityClaim: factory methods, iteration, composition."""

from __future__ import annotations

import pytest

from anonframework.core.interfaces.security_claim import SecurityClaim

from .conftest import StubTask

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestSecurityClaimConstruction:
    def test_direct_init_raises_type_error(self) -> None:
        """SecurityClaim() must be constructed via factory classmethods."""
        with pytest.raises(TypeError, match="from_tasks.*from_claims"):
            SecurityClaim()

    def test_from_tasks_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="At least one task"):
            SecurityClaim.from_tasks([])

    def test_from_claims_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="At least one claim"):
            SecurityClaim.from_claims([])


# ---------------------------------------------------------------------------
# from_tasks iteration
# ---------------------------------------------------------------------------


class TestSecurityClaimFromTasks:
    def test_iterates_all_tasks(self) -> None:
        t1, t2 = StubTask(goal_text="a"), StubTask(goal_text="b")
        claim = SecurityClaim.from_tasks([t1, t2])
        tasks = list(claim)
        assert tasks == [t1, t2]

    def test_re_iterable(self) -> None:
        """Claims can be iterated multiple times (tasks are stateless)."""
        t1 = StubTask(goal_text="a")
        claim = SecurityClaim.from_tasks([t1])
        assert list(claim) == [t1]
        assert list(claim) == [t1]

    def test_defensive_copy(self) -> None:
        """Mutating the original list does not affect the claim."""
        tasks = [StubTask(goal_text="a")]
        claim = SecurityClaim.from_tasks(tasks)
        tasks.clear()
        assert len(list(claim)) == 1


# ---------------------------------------------------------------------------
# from_claims composition
# ---------------------------------------------------------------------------


class TestSecurityClaimFromClaims:
    def test_chains_two_claims(self) -> None:
        t1, t2 = StubTask(goal_text="a"), StubTask(goal_text="b")
        c1 = SecurityClaim.from_tasks([t1])
        c2 = SecurityClaim.from_tasks([t2])
        combined = SecurityClaim.from_claims([c1, c2])
        tasks = list(combined)
        assert tasks == [t1, t2]

    def test_nested_claims(self) -> None:
        """Claims of claims chain lazily."""
        t1, t2, t3 = (
            StubTask(goal_text="a"),
            StubTask(goal_text="b"),
            StubTask(goal_text="c"),
        )
        inner = SecurityClaim.from_claims(
            [
                SecurityClaim.from_tasks([t1]),
                SecurityClaim.from_tasks([t2]),
            ]
        )
        outer = SecurityClaim.from_claims([inner, SecurityClaim.from_tasks([t3])])
        tasks = list(outer)
        assert tasks == [t1, t2, t3]

    def test_re_iterable_claims(self) -> None:
        t1 = StubTask(goal_text="a")
        c1 = SecurityClaim.from_tasks([t1])
        combined = SecurityClaim.from_claims([c1])
        assert list(combined) == [t1]
        assert list(combined) == [t1]

    def test_defensive_copy_claims(self) -> None:
        """Mutating the original claims list does not affect the composed claim."""
        c1 = SecurityClaim.from_tasks([StubTask(goal_text="a")])
        claims = [c1]
        combined = SecurityClaim.from_claims(claims)
        claims.clear()
        assert len(list(combined)) == 1
