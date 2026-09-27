"""Security domain tags for classifying attack surfaces.

Security domains are defined at runtime by the target system as a forest
(one or more trees) of :class:`SecurityDomainTag` nodes, grouped in a
:class:`SecurityDomain` instance.

Example::

    internal = SecurityDomainTag("internal")
    external = SecurityDomainTag("external", parent=internal)
    user = SecurityDomainTag("user", parent=external)
    api = SecurityDomainTag("api", parent=external)   # sibling of user

    domain = SecurityDomain([internal, external, user, api])

A tag *includes* all of its descendants: ``internal.includes(user)``
is ``True`` because internal scope encompasses user-controlled surfaces.
Unrelated roots form independent trees in the forest.

A :data:`Scope` is a ``frozenset[SecurityDomainTag]`` representing a
combination of security domain tags to test simultaneously.  Use
:func:`scope_includes` to check whether a scope covers a given tag.

The controller takes a read & write ``scope`` plus an optional
``read_only`` scope for visible-but-not-injectable surfaces (see
:class:`~anonframework.core.controller.Controller`); both are plain
:data:`Scope` values, so access level is expressed by which set a tag
lands in, not by any per-tag wrapper.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class SecurityDomainTag:
    """A single security domain — a node in the domain tree.

    Attributes:
        name: Unique human-readable identifier.
        parent: The parent tag, or ``None`` for a root.
    """

    name: str
    parent: SecurityDomainTag | None = None

    def includes(self, other: SecurityDomainTag) -> bool:
        """Whether this tag includes *other* (other is a descendant or equal)."""
        current: SecurityDomainTag | None = other
        while current is not None:
            if current is self:
                return True
            current = current.parent
        return False


class SecurityDomain:
    """A forest of security domain tags provided by a target system.

    Immutable after construction.

    Attributes:
        roots: Tags with no parent.
    """

    _tags: dict[str, SecurityDomainTag]

    def __init__(self, tags: Sequence[SecurityDomainTag]) -> None:
        tag_map: dict[str, SecurityDomainTag] = {}
        for tag in tags:
            if tag.name in tag_map:
                raise ValueError(f"Duplicate tag name: {tag.name!r}")
            tag_map[tag.name] = tag

        for tag in tag_map.values():
            if tag.parent is not None and tag.parent.name not in tag_map:
                raise ValueError(
                    f"Tag {tag.name!r} has parent {tag.parent.name!r} which is not in the domain"
                )

        object.__setattr__(self, "_tags", tag_map)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("SecurityDomain is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("SecurityDomain is immutable")

    @property
    def roots(self) -> list[SecurityDomainTag]:
        """Tags with no parent."""
        return [t for t in self._tags.values() if t.parent is None]

    def distinct_combinations(self) -> list[frozenset[SecurityDomainTag]]:
        """All distinct tag combinations to test.

        Within a single tree, a combination never includes both a node and
        any of its ancestors (the ancestor already covers the descendant).
        Across independent trees the result is the Cartesian product of
        each tree's distinct selections.

        Returns:
            A list of frozensets, each representing one combination to test.
            Includes the empty set (no tags selected).
        """
        # Build children map
        children: dict[str, list[SecurityDomainTag]] = {tag.name: [] for tag in self._tags.values()}
        for tag in self._tags.values():
            if tag.parent is not None and tag.parent.name in children:
                children[tag.parent.name].append(tag)

        def _antichains(node: SecurityDomainTag) -> list[frozenset[SecurityDomainTag]]:
            """Antichains of the subtree rooted at *node*."""
            child_nodes = children[node.name]
            if not child_nodes:
                return [frozenset(), frozenset({node})]

            # Exclude this node: Cartesian product of children's antichains
            without_node: list[frozenset[SecurityDomainTag]] = [frozenset()]
            for child in child_nodes:
                child_ac = _antichains(child)
                without_node = [existing | new for existing in without_node for new in child_ac]

            # Include this node: no descendants allowed
            without_node.append(frozenset({node}))
            return without_node

        # Antichains per root tree, then Cartesian product across trees
        result: list[frozenset[SecurityDomainTag]] = [frozenset()]
        for root in self.roots:
            tree_ac = _antichains(root)
            result = [existing | new for existing in result for new in tree_ac]

        return result


Scope = frozenset[SecurityDomainTag]
"""A set of security domain tags defining the attack surface scope.

An item is in scope if ANY tag in the set includes it.  Caller-specified
scopes are conventionally antichains (no tag an ancestor of another —
the ancestor already covers the descendant), but this is neither enforced
nor required.
"""


def scope_includes(scope: Scope, tag: SecurityDomainTag) -> bool:
    """Whether *scope* covers *tag*.

    Returns ``True`` if any tag in *scope* includes *tag* (i.e. *tag*
    is equal to or a descendant of at least one scope member).
    An empty scope covers nothing.
    """
    return any(s.includes(tag) for s in scope)
