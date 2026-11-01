import pytest

from regression_lens.rendering import RenderConfig


def test_render_config_has_documented_defaults() -> None:
    config = RenderConfig()

    assert config.to_dict() == {
        "browser": "chromium",
        "viewport_width": 1280,
        "viewport_height": 800,
        "device_scale_factor": 1.0,
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"browser": ""},
        {"viewport_width": 0},
        {"viewport_height": -1},
        {"device_scale_factor": 0},
    ],
)
def test_render_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        RenderConfig(**kwargs)
