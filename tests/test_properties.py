"""Property-based tests using Hypothesis.

Each test verifies an invariant that must hold for ALL valid inputs.
"""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from anonframework.core.types.events import ObservableEvent
from anonframework.core.types.observable import Observable
from anonframework.core.types.security_domain import SecurityDomain, SecurityDomainTag
from anonframework.core.types.trajectory import Trajectory

_TAG = SecurityDomainTag("prop_test")
_OBS = Observable(name="prop", security_domain=_TAG)


def _obs(content: str) -> ObservableEvent:
    return ObservableEvent(observable=_OBS, content=content)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


# Generate a linear chain of SecurityDomainTags (depth 1-5)
@st.composite
def linear_chain(draw: st.DrawFn) -> list[SecurityDomainTag]:
    """Generate a linear chain of tags: root -> child -> ... -> leaf."""
    depth = draw(st.integers(min_value=1, max_value=5))
    tags: list[SecurityDomainTag] = []
    for i in range(depth):
        parent = tags[-1] if tags else None
        tags.append(SecurityDomainTag(f"tag_{i}", parent=parent))
    return tags


@st.composite
def tag_tree(draw: st.DrawFn) -> list[SecurityDomainTag]:
    """Generate a random tree of SecurityDomainTags.

    Strategy: build nodes top-down. Each new node picks an existing node
    as parent (or is a root).
    """
    n_nodes = draw(st.integers(min_value=1, max_value=8))
    tags: list[SecurityDomainTag] = []
    for i in range(n_nodes):
        if not tags or draw(st.booleans()):
            # New root
            tags.append(SecurityDomainTag(f"t{i}"))
        else:
            # Child of an existing node
            parent = draw(st.sampled_from(tags))
            tags.append(SecurityDomainTag(f"t{i}", parent=parent))
    return tags


# ---------------------------------------------------------------------------
# SecurityDomainTag.includes() properties
# ---------------------------------------------------------------------------


class TestIncludesProperties:
    @given(linear_chain())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.differing_executors])
    def test_includes_is_reflexive(self, chain: list[SecurityDomainTag]) -> None:
        """Every tag includes itself."""
        for tag in chain:
            assert tag.includes(tag)

    @given(linear_chain())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.differing_executors])
    def test_ancestor_includes_all_descendants(self, chain: list[SecurityDomainTag]) -> None:
        """In a linear chain, the root includes every node below it."""
        root = chain[0]
        for tag in chain:
            assert root.includes(tag)

    @given(linear_chain())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.differing_executors])
    def test_descendant_does_not_include_ancestor(self, chain: list[SecurityDomainTag]) -> None:
        """A descendant never includes a strict ancestor."""
        if len(chain) < 2:
            return
        leaf = chain[-1]
        for ancestor in chain[:-1]:
            assert not leaf.includes(ancestor)


# ---------------------------------------------------------------------------
# SecurityDomain.distinct_combinations() properties
# ---------------------------------------------------------------------------


class TestDistinctCombinationsProperties:
    @given(tag_tree())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.differing_executors])
    def test_always_contains_empty_set(self, tags: list[SecurityDomainTag]) -> None:
        """Every domain's combinations include the empty set."""
        domain = SecurityDomain(tags)
        combos = domain.distinct_combinations()
        assert frozenset() in combos

    @given(tag_tree())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.differing_executors])
    def test_no_duplicates(self, tags: list[SecurityDomainTag]) -> None:
        """No two combinations are identical."""
        domain = SecurityDomain(tags)
        combos = domain.distinct_combinations()
        assert len(combos) == len(set(combos))

    @given(tag_tree())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.differing_executors])
    def test_antichain_property(self, tags: list[SecurityDomainTag]) -> None:
        """No combination contains both a tag and one of its ancestors."""
        domain = SecurityDomain(tags)
        for combo in domain.distinct_combinations():
            tag_list = list(combo)
            for i, t1 in enumerate(tag_list):
                for t2 in tag_list[i + 1 :]:
                    assert not t1.includes(t2) or t1 is t2
                    assert not t2.includes(t1) or t2 is t1

    @given(tag_tree())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.differing_executors])
    def test_every_tag_appears_in_some_combination(self, tags: list[SecurityDomainTag]) -> None:
        """Every tag in the domain appears in at least one combination."""
        domain = SecurityDomain(tags)
        combos = domain.distinct_combinations()
        all_tags_in_combos = set()
        for combo in combos:
            all_tags_in_combos.update(combo)

        for tag in tags:
            assert tag in all_tags_in_combos, f"Tag {tag.name} not in any combination"


# ---------------------------------------------------------------------------
# Trajectory emit/drain/snapshot consistency
# ---------------------------------------------------------------------------


class TestTrajectoryProperties:
    @given(st.lists(st.text(min_size=1, max_size=50), min_size=0, max_size=50))
    @settings(max_examples=200, suppress_health_check=[HealthCheck.differing_executors])
    def test_snapshot_preserves_all_emits(self, contents: list[str]) -> None:
        """snapshot() returns all emitted entries in order."""
        t = Trajectory()
        for c in contents:
            t.emit(_obs(c))
        snap = t.snapshot()
        assert [e.content for e in snap] == contents

    @given(st.lists(st.text(min_size=1, max_size=50), min_size=0, max_size=50))
    @settings(max_examples=200, suppress_health_check=[HealthCheck.differing_executors])
    def test_drain_returns_exactly_new_entries(self, contents: list[str]) -> None:
        """drain() returns entries emitted since the last drain, nothing more."""
        t = Trajectory()

        # Emit first half and drain
        mid = len(contents) // 2
        for c in contents[:mid]:
            t.emit(_obs(c))
        first_drain = t.drain()

        # Emit second half and drain
        for c in contents[mid:]:
            t.emit(_obs(c))
        second_drain = t.drain()

        assert [e.content for e in first_drain] == contents[:mid]
        assert [e.content for e in second_drain] == contents[mid:]

    @given(st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=50))
    @settings(max_examples=200, suppress_health_check=[HealthCheck.differing_executors])
    def test_snapshot_idempotent(self, contents: list[str]) -> None:
        """Calling snapshot() twice with no emits in between returns the same data."""
        t = Trajectory()
        for c in contents:
            t.emit(_obs(c))
        s1 = [e.content for e in t.snapshot()]
        s2 = [e.content for e in t.snapshot()]
        assert s1 == s2

    @given(
        st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=30),
        st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.differing_executors])
    def test_multiple_drains_cover_all_entries(self, contents: list[str], n_drains: int) -> None:
        """The union of all drain() calls equals the full snapshot."""
        t = Trajectory()
        for c in contents:
            t.emit(_obs(c))

        all_drained: list[str] = []
        for _ in range(n_drains):
            all_drained.extend(e.content for e in t.drain())

        # First drain gets everything; subsequent drains get nothing
        assert all_drained == contents
