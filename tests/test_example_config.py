from pathlib import Path

import yaml


def test_example_config_loads_with_safe_defaults() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "mutations.example.yaml"

    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    assert config["schema_version"] == 1
    assert config["render"]["browser"] == "chromium"
    assert config["generation"]["mutations_per_page"] is None
    assert all(
        mutation["status"] == "planned" and mutation["enabled"] is False
        for mutation in config["mutations"].values()
    )
