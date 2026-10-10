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


@pytest.mark.parametrize(
    ("filename", "command"),
    [("train.yaml", "train"), ("two_stage.yaml", "two-stage"), ("evaluate.yaml", "evaluate")],
)
def test_france_external_configuration(filename: str, command: str) -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / filename)
    validate_config(config, command)
    assert set(config["data"]["external"]) == {"ruijin", "wuhan", "FUSCC", "SPH_test", "france"}
    france = config["data"]["external"]["france"]
    assert config["data"]["id_column"] == "影像组学序列号"
    assert france["id_column"] == config["data"]["id_column"]
    assert france["label_column"] == "pCR"
    assert france["metadata"].endswith("/PET_France/PETCT.xlsx")
    assert france["ct_root"].endswith("/clahed_64_0.2_20/france/ct")
    assert france["pet_root"].endswith("/clahed_64_0.2_20/france/pet")
    for name, cohort in config["data"]["external"].items():
        for modality in ("ct", "pet"):
            assert cohort[f"{modality}_root"] == (
                f"/data4/zhenglujie/petct/clahed_64_0.2_20/{name}/{modality}"
            )
    manifest = resolve_config_path(config, france["manifest"])
    assert manifest.name == "france.txt"
    assert manifest.is_file()


def test_hpo_configuration_has_no_external_cohorts() -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "hpo.yaml")
    validate_config(config, "hpo")
    assert not config["data"].get("external")
