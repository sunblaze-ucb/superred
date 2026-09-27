"""Observables: global information the optimizer receives about the target system.

Observables are static context available before any run — system descriptions,
source code, configuration, tool catalogs, etc. Runtime execution data is
captured in trajectories instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from anonframework.core.types.security_domain import SecurityDomainTag


@dataclass(frozen=True)
class Observable:
    """Specification of an observable.

    Attributes:
        name: Unique identifier within the target module.
        security_domain: Trust boundary this observable belongs to.
            Must not be ``None`` — observables always belong to a
            specific domain.
        description: Human-readable description.
        observable_type: Type of content ("text", "code", "config", "json").
    """

    name: str
    security_domain: SecurityDomainTag
    description: str = ""
    observable_type: str = "text"


@dataclass(frozen=True)
class ObservableValue:
    """An observable with its content.

    Available before execution and stable across runs
    (e.g. system description, source code, configuration).
    """

    observable: Observable
    content: Any = None
