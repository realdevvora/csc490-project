"""Configuration for deterministic webpage rendering."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RenderConfig:
    """Browser settings that materially affect generated screenshots."""

    browser: str = "chromium"
    viewport_width: int = 1280
    viewport_height: int = 800
    device_scale_factor: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.browser, str) or not self.browser.strip():
            raise ValueError("browser must be a non-empty string")
        if self.viewport_width <= 0 or self.viewport_height <= 0:
            raise ValueError("viewport dimensions must be positive")
        if self.device_scale_factor <= 0:
            raise ValueError("device_scale_factor must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly representation."""

        return asdict(self)
