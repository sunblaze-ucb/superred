"""Tests for SecurityDomain.distinct_combinations — 100% branch coverage."""

from anonframework.core.types.security_domain import SecurityDomain, SecurityDomainTag


def _names(combo: frozenset[SecurityDomainTag]) -> frozenset[str]:
    return frozenset(t.name for t in combo)


def _all_names(combos: list[frozenset[SecurityDomainTag]]) -> set[frozenset[str]]:
    return {_names(c) for c in combos}


class TestEmptyDomain:
    """No tags at all — the root loop never executes."""

    def test_returns_only_empty_set(self) -> None:
        domain = SecurityDomain([])
        assert domain.distinct_combinations() == [frozenset()]


class TestSingleLeaf:
    """One root with no children — hits the leaf branch in _antichains."""

    def test_single_node(self) -> None:
        a = SecurityDomainTag("a")
        domain = SecurityDomain([a])
        result = _all_names(domain.distinct_combinations())
        assert result == {frozenset(), frozenset({"a"})}


class TestLinearChain:
    """Linear chain: root → child → grandchild.

    Covers: node-with-children branch, single-child product loop.
    """

    def test_three_deep(self) -> None:
        root = SecurityDomainTag("root")
        mid = SecurityDomainTag("mid", parent=root)
        leaf = SecurityDomainTag("leaf", parent=mid)
        domain = SecurityDomain([root, mid, leaf])
        result = _all_names(domain.distinct_combinations())
        assert result == {
            frozenset(),
            frozenset({"leaf"}),
            frozenset({"mid"}),
            frozenset({"root"}),
        }


class TestBranching:
    """Parent with multiple children — product loop iterates >1 time."""

    def test_two_siblings(self) -> None:
        root = SecurityDomainTag("root")
        left = SecurityDomainTag("left", parent=root)
        right = SecurityDomainTag("right", parent=root)
        domain = SecurityDomain([root, left, right])
        result = _all_names(domain.distinct_combinations())
        assert result == {
            frozenset(),
            frozenset({"left"}),
            frozenset({"right"}),
            frozenset({"left", "right"}),
            frozenset({"root"}),
        }


class TestForest:
    """Multiple independent roots — Cartesian product across trees."""

    def test_two_independent_roots(self) -> None:
        a = SecurityDomainTag("a")
        b = SecurityDomainTag("b")
        domain = SecurityDomain([a, b])
        result = _all_names(domain.distinct_combinations())
        assert result == {
            frozenset(),
            frozenset({"a"}),
            frozenset({"b"}),
            frozenset({"a", "b"}),
        }

    def test_forest_with_subtrees(self) -> None:
        # Tree 1: r1 -> c1
        r1 = SecurityDomainTag("r1")
        c1 = SecurityDomainTag("c1", parent=r1)
        # Tree 2: r2 (leaf)
        r2 = SecurityDomainTag("r2")
        domain = SecurityDomain([r1, c1, r2])
        # Tree 1 antichains: {}, {c1}, {r1}  (3)
        # Tree 2 antichains: {}, {r2}         (2)
        # Product: 3 * 2 = 6
        result = _all_names(domain.distinct_combinations())
        assert len(result) == 6
        assert frozenset({"c1", "r2"}) in result
        assert frozenset({"r1", "r2"}) in result


class TestParentOutsideDomain:
    """Tag whose parent is not in the domain — rejected at construction."""

    def test_orphan_tag_raises(self) -> None:
        import pytest

        outside = SecurityDomainTag("outside")
        child = SecurityDomainTag("child", parent=outside)
        with pytest.raises(ValueError, match="parent.*not in the domain"):
            SecurityDomain([child])
