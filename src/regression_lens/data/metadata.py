"""Typed metadata for one generated reference/candidate pair."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any

from regression_lens.mutations import MutationRecord
from regression_lens.rendering import RenderConfig


INTERNAL_SPLITS = frozenset({"train", "validation", "test"})


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _portable_relative_path(value: object, field_name: str) -> str:
    raw_path = _required_text(value, field_name)
    normalized = PurePosixPath(raw_path.replace("\\", "/"))
    if (
        normalized.is_absolute()
        or PureWindowsPath(raw_path).is_absolute()
        or ".." in normalized.parts
        or normalized.as_posix() == "."
    ):
        raise ValueError(f"{field_name} must be a portable relative path")
    return normalized.as_posix()


@dataclass(frozen=True)
class GeneratedSample:
    """Core metadata for a generated internal-dataset sample.

    This intentionally models generated Design2Code pairs, not external benchmark
    records. Fields should be extended only as the pipeline needs them.
    """

    sample_id: str
    source_page_id: str
    split: str
    reference_image: str
    candidate_image: str
    is_regression: bool
    mutation: MutationRecord | None = None
    variation_type: str | None = None
    target_bbox: tuple[int, int, int, int] | None = None
    render: RenderConfig = field(default_factory=RenderConfig)
    schema_version: str = "1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_id", _required_text(self.sample_id, "sample_id"))
        object.__setattr__(
            self,
            "source_page_id",
            _required_text(self.source_page_id, "source_page_id"),
        )
        object.__setattr__(
            self,
            "reference_image",
            _portable_relative_path(self.reference_image, "reference_image"),
        )
        object.__setattr__(
            self,
            "candidate_image",
            _portable_relative_path(self.candidate_image, "candidate_image"),
        )

        if self.split not in INTERNAL_SPLITS:
            allowed = ", ".join(sorted(INTERNAL_SPLITS))
            raise ValueError(f"split must be one of: {allowed}")
        if type(self.is_regression) is not bool:
            raise TypeError("is_regression must be a boolean")
        if not isinstance(self.render, RenderConfig):
            raise TypeError("render must be a RenderConfig")
        _required_text(self.schema_version, "schema_version")

        if self.is_regression:
            if not isinstance(self.mutation, MutationRecord):
                raise ValueError("regression samples require a MutationRecord")
            if self.variation_type is not None:
                raise ValueError("regression samples cannot define variation_type")
        else:
            if self.mutation is not None:
                raise ValueError("benign samples cannot contain a mutation record")
            normalized_variation = _required_text(
                self.variation_type,
                "variation_type",
            )
            object.__setattr__(self, "variation_type", normalized_variation)

        if self.target_bbox is not None:
            if len(self.target_bbox) != 4 or any(
                type(value) is not int for value in self.target_bbox
            ):
                raise ValueError("target_bbox must contain four integers")
            x, y, width, height = self.target_bbox
            if x < 0 or y < 0 or width <= 0 or height <= 0:
                raise ValueError(
                    "target_bbox origin must be non-negative and dimensions positive"
                )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of the current schema."""

        result = asdict(self)
        if self.target_bbox is not None:
            result["target_bbox"] = list(self.target_bbox)
        return result
