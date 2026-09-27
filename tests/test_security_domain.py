"""Unit tests for SecurityDomainTag and SecurityDomain.

Covers: includes() traversal, SecurityDomain construction validation,
immutability enforcement, roots property, and distinct_combinations.
"""

from __future__ import annotations

import pytest

from anonframework.core.types.security_domain import (
    SecurityDomain,
    SecurityDomainTag,
)

# ---------------------------------------------------------------------------
# SecurityDomainTag.includes()
# ---------------------------------------------------------------------------


class TestSecurityDomainTagIncludes:
    def test_self_includes_self(self) -> None:
        tag = SecurityDomainTag("a")
        assert tag.includes(tag) is True

    def test_parent_includes_child(self) -> None:
        parent = SecurityDomainTag("parent")
        child = SecurityDomainTag("child", parent=parent)
        assert parent.includes(child) is True

    def test_child_does_not_include_parent(self) -> None:
        parent = SecurityDomainTag("parent")
        child = SecurityDomainTag("child", parent=parent)
        assert child.includes(parent) is False

    def test_grandparent_includes_grandchild(self) -> None:
        root = SecurityDomainTag("root")
        mid = SecurityDomainTag("mid", parent=root)
        leaf = SecurityDomainTag("leaf", parent=mid)
        assert root.includes(leaf) is True

    def test_siblings_do_not_include_each_other(self) -> None:
        parent = SecurityDomainTag("parent")
        a = SecurityDomainTag("a", parent=parent)
        b = SecurityDomainTag("b", parent=parent)
        assert a.includes(b) is False
        assert b.includes(a) is False

    def test_unrelated_roots_do_not_include_each_other(self) -> None:
        a = SecurityDomainTag("a")
        b = SecurityDomainTag("b")
        assert a.includes(b) is False


# ---------------------------------------------------------------------------
# SecurityDomain construction
# ---------------------------------------------------------------------------


class TestSecurityDomainConstruction:
    def test_empty_domain(self) -> None:
        domain = SecurityDomain([])
        assert domain.roots == []

    def test_single_root(self) -> None:
        root = SecurityDomainTag("root")
        domain = SecurityDomain([root])
        assert domain.roots == [root]

    def test_duplicate_tag_name_raises(self) -> None:
        a = SecurityDomainTag("same")
        b = SecurityDomainTag("same")
        with pytest.raises(ValueError, match="Duplicate tag name"):
            SecurityDomain([a, b])

    def test_orphan_tag_raises(self) -> None:
        outside = SecurityDomainTag("outside")
        child = SecurityDomainTag("child", parent=outside)
        with pytest.raises(ValueError, match="parent.*not in the domain"):
            SecurityDomain([child])


# ---------------------------------------------------------------------------
# SecurityDomain immutability
# ---------------------------------------------------------------------------


class TestSecurityDomainImmutability:
    def test_setattr_raises(self) -> None:
        domain = SecurityDomain([SecurityDomainTag("a")])
        with pytest.raises(AttributeError, match="immutable"):
            domain.x = 1  # type: ignore[attr-defined]

    def test_delattr_raises(self) -> None:
        domain = SecurityDomain([SecurityDomainTag("a")])
        with pytest.raises(AttributeError, match="immutable"):
            del domain._tags  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# SecurityDomain.roots
# ---------------------------------------------------------------------------


class TestSecurityDomainRoots:
    def test_multiple_roots(self) -> None:
        a = SecurityDomainTag("a")
        b = SecurityDomainTag("b")
        domain = SecurityDomain([a, b])
        root_names = {r.name for r in domain.roots}
        assert root_names == {"a", "b"}

    def test_child_not_in_roots(self) -> None:
        root = SecurityDomainTag("root")
        child = SecurityDomainTag("child", parent=root)
        domain = SecurityDomain([root, child])
        assert len(domain.roots) == 1
        assert domain.roots[0] is root


# ---------------------------------------------------------------------------
# SecurityDomain.distinct_combinations (supplement to test_distinct_combinations.py)
# ---------------------------------------------------------------------------


class TestScopeIncludes:
    """Tests for the scope_includes() helper function."""

    def test_empty_scope_includes_nothing(self) -> None:
        """An empty scope covers no tags."""
        from anonframework.core.types.security_domain import scope_includes

        tag = SecurityDomainTag("a")
        assert scope_includes(frozenset(), tag) is False

    def test_single_tag_includes_self(self) -> None:
        """A scope with one tag includes that tag."""
        from anonframework.core.types.security_domain import scope_includes

        tag = SecurityDomainTag("a")
        assert scope_includes(frozenset({tag}), tag) is True

    def test_single_tag_includes_descendant(self) -> None:
        """A scope with a parent tag includes its descendant."""
        from anonframework.core.types.security_domain import scope_includes

        parent = SecurityDomainTag("parent")
        child = SecurityDomainTag("child", parent=parent)
        assert scope_includes(frozenset({parent}), child) is True

    def test_single_tag_excludes_ancestor(self) -> None:
        """A scope with a child tag does NOT include its ancestor."""
        from anonframework.core.types.security_domain import scope_includes

        parent = SecurityDomainTag("parent")
        child = SecurityDomainTag("child", parent=parent)
        assert scope_includes(frozenset({child}), parent) is False

    def test_single_tag_excludes_sibling(self) -> None:
        """A scope with one child does NOT include its sibling."""
        from anonframework.core.types.security_domain import scope_includes

        parent = SecurityDomainTag("parent")
        child_a = SecurityDomainTag("a", parent=parent)
        child_b = SecurityDomainTag("b", parent=parent)
        assert scope_includes(frozenset({child_a}), child_b) is False

    def test_multi_tag_scope_any_match(self) -> None:
        """A scope with multiple tags includes if ANY tag includes the target."""
        from anonframework.core.types.security_domain import scope_includes

        root = SecurityDomainTag("root")
        alpha = SecurityDomainTag("alpha", parent=root)
        beta = SecurityDomainTag("beta", parent=root)
        alpha_child = SecurityDomainTag("alpha_child", parent=alpha)
        # {alpha, beta} should include alpha_child (via alpha)
        assert scope_includes(frozenset({alpha, beta}), alpha_child) is True

    def test_multi_tag_scope_none_match(self) -> None:
        """A scope with multiple tags excludes if NO tag includes the target."""
        from anonframework.core.types.security_domain import scope_includes

        root = SecurityDomainTag("root")
        alpha = SecurityDomainTag("alpha", parent=root)
        beta = SecurityDomainTag("beta", parent=root)
        # {alpha, beta} should NOT include root (neither is ancestor of root)
        assert scope_includes(frozenset({alpha, beta}), root) is False

    def test_multi_tag_scope_unrelated_trees(self) -> None:
        """Scope with tags from independent trees works correctly."""
        from anonframework.core.types.security_domain import scope_includes

        tree_a = SecurityDomainTag("tree_a")
        tree_b = SecurityDomainTag("tree_b")
        child_a = SecurityDomainTag("child_a", parent=tree_a)
        child_b = SecurityDomainTag("child_b", parent=tree_b)
        scope = frozenset({tree_a, child_b})
        assert scope_includes(scope, child_a) is True  # via tree_a
        assert scope_includes(scope, child_b) is True  # via child_b (self)
        assert scope_includes(scope, tree_b) is False  # child_b doesn't include parent


class TestDistinctCombinationsEdge:
    def test_result_always_contains_empty_set(self) -> None:
        """Every domain's combinations include the empty set (no tags selected)."""
        root = SecurityDomainTag("root")
        child = SecurityDomainTag("child", parent=root)
        domain = SecurityDomain([root, child])
        combos = domain.distinct_combinations()
        assert frozenset() in combos

    def test_no_ancestor_descendant_pair_in_any_combination(self) -> None:
        """No combination contains both a tag and one of its ancestors."""
        root = SecurityDomainTag("root")
        mid = SecurityDomainTag("mid", parent=root)
        leaf = SecurityDomainTag("leaf", parent=mid)
        domain = SecurityDomain([root, mid, leaf])

        for combo in domain.distinct_combinations():
            tags = list(combo)
            for i, t1 in enumerate(tags):
                for t2 in tags[i + 1 :]:
                    assert not t1.includes(t2), f"{t1.name} includes {t2.name}"
                    assert not t2.includes(t1), f"{t2.name} includes {t1.name}"
