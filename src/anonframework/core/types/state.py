"""Configuration and query specifications for targets.

Two distinct concepts:

- :class:`ConfigSpec` — pre-run configuration slots. The task sets these
  before a run via ``target.set_config()``. The description documents the
  accepted format.

- :class:`QuerySpec` — post-run interactions. The evaluator calls these
  after a run via ``target.query()``. May be a simple getter or an action
  with parameters. The description documents how to use it.

These are intentionally separate: what you configure before a run is not
the same as what you interact with after. Manual values (API keys,
credentials) are passed directly to the target's constructor.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from anonframework.core.types.security_domain import SecurityDomainTag


@dataclass(frozen=True)
class ConfigSpec:
    """A pre-run configuration slot on a target.

    Used by tasks to set up initial state. Values are always text;
    the description documents the expected format — that is the contract.

    Attributes:
        name: Unique identifier within the target.
        security_domain: Trust boundary this config belongs to.
            Must not be ``None``.
        description: Documents the accepted format.
    """

    name: str
    security_domain: SecurityDomainTag
    description: str


@dataclass(frozen=True)
class QueryParam:
    """A parameter for a post-run query.

    Attributes:
        name: Parameter name.
        description: Documents the expected format and purpose.
    """

    name: str
    description: str


@dataclass(frozen=True)
class QuerySpec:
    """A post-run interaction on a target.

    Used by the evaluator after a run. May be a simple getter (no params)
    or an action with parameters. All values (params and return) are text.

    Attributes:
        name: Unique identifier within the target.
        description: Documents how to use this query — what it does,
            what it returns.
        params: Parameters this query accepts. Empty for simple getters.
    """

    name: str
    description: str
    params: list[QueryParam] = field(default_factory=list)
