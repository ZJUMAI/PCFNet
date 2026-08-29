from pathlib import Path

import pytest

pytest.importorskip("yaml")

from petct.config import (
    ConfigError,
    apply_overrides,
    load_config,
    resolve_config_path,
    validate_config,
)


def test_yaml_overrides_preserve_types(tmp_path: Path) -> None:
    path = tmp_path / "experiment.yaml"
    path.write_text("training:\n  batch_size: 8\nmodel:\n  pretrained: true\n", encoding="utf-8")
    config = load_config(
        path,
        ["training.batch_size=4", "model.pretrained=false", "new.value=[1, 2]"],
    )
    assert config["training"]["batch_size"] == 4
    assert config["model"]["pretrained"] is False
    assert config["new"]["value"] == [1, 2]
    assert resolve_config_path(config, "data/file.csv") == tmp_path / "data/file.csv"


def test_invalid_override_is_rejected() -> None:
    with pytest.raises(ConfigError):
        apply_overrides({}, ["missing_equals"])


def test_training_schema_rejects_missing_split_fields() -> None:
    config = {
        "data": {"train": {}, "validation": {}},
        "model": {},
        "training": {},
        "output": {},
    }
    with pytest.raises(ConfigError, match="data.train"):
        validate_config(config, "train")
