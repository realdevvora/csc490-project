"""Minimal interfaces shared by future mutation implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class MutationRecord:
    """Provenance required to reproduce one intentional mutation."""

    mutation_family: str
    mutation_type: str
    target: str
    before_value: Any
    after_value: Any
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("mutation_family", "mutation_type", "target"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly representation."""

        return asdict(self)


class Mutation(ABC):
    """Apply one controlled HTML mutation and return its provenance."""

    name: str
    family: str

    @abstractmethod
    def apply(self, html: str) -> tuple[str, MutationRecord]:
        """Return mutated HTML and the exact mutation record."""
