"""Controllable: an injection point exposed by a target module.

Each controllable is tagged with a security domain and represents an
attack surface that the optimizer can manipulate.
"""

from __future__ import annotations

from dataclasses import dataclass

from anonframework.core.types.security_domain import SecurityDomainTag


@dataclass(frozen=True)
class Controllable:
    """A controllable injection point (attack surface).

    Attributes:
        name: Unique identifier within the target module.
        security_domain: Trust boundary this controllable belongs to.
            Must not be ``None`` — controllables are physical injection
            points that always belong to a specific domain.
        description: Human-readable description of the injection point.
        value_type: Expected type of the controllable value ("text", "json",
            "modifier", "binary").
    """

    name: str
    security_domain: SecurityDomainTag
    description: str = ""
    value_type: str = "text"
