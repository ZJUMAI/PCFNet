from pathlib import Path

import pytest

pytest.importorskip("yaml")

from petct.config import (
    ConfigError,
    apply_overrides,
    checkpoint_saving_enabled,
    external_selection_mode,
    load_config,
    resolve_config_path,
    validate_config,
    validate_split_config,
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
    external = config["data"]["external"]
    assert list(external) == ["multi_hospital", "france", "SPH_test"]
    members = external["multi_hospital"]["members"]
    assert list(members) == ["wuhan", "ruijin", "FUSCC"]
    france = config["data"]["external"]["france"]
    assert config["data"]["id_column"] == "影像组学序列号"
    assert france["id_column"] == config["data"]["id_column"]
    assert france["label_column"] == "pCR"
    assert france["metadata"].endswith("/PET_France/PETCT.xlsx")
    assert france["ct_root"].endswith("/clahed_64_0.2_20/france/ct")
    assert france["pet_root"].endswith("/clahed_64_0.2_20/france/pet")
    sources = dict(members, france=external["france"], SPH_test=external["SPH_test"])
    for name, cohort in sources.items():
        for modality in ("ct", "pet"):
            assert cohort[f"{modality}_root"] == (
                f"/data4/zhenglujie/petct/clahed_64_0.2_20/{name}/{modality}"
            )
    manifest = resolve_config_path(config, france["manifest"])
    assert manifest.name == "france.txt"
    assert manifest.is_file()

    combined_ids = []
    for member in members.values():
        member_manifest = resolve_config_path(config, member["manifest"])
        combined_ids.extend(member_manifest.read_text(encoding="utf-8").split())
    assert len(combined_ids) == len(set(combined_ids)) == 70


def _source() -> dict[str, str]:
    return dict(manifest="cases.txt", metadata="labels.csv", ct_root="ct", pet_root="pet")


def test_grouped_evaluation_schema() -> None:
    group = {"members": {"wuhan": _source(), "ruijin": _source(), "FUSCC": _source()}}
    validate_split_config(group, "external", allow_groups=True)
    with pytest.raises(ConfigError, match="only supported for evaluation"):
        validate_split_config(group, "data.train")


@pytest.mark.parametrize(
    ("group", "message"),
    [
        ({"members": {}}, "non-empty named mapping"),
        ({"members": []}, "non-empty named mapping"),
        ({"members": {"wuhan": {}}}, "members.wuhan is missing keys"),
        ({"members": {"wuhan": None}}, "members.wuhan must be a mapping"),
        ({"members": {"nested": {"members": {"wuhan": _source()}}}}, "only supported"),
        ({"members": {"wuhan": _source()}, "manifest": "cases.txt"}, "cannot mix"),
    ],
)
def test_invalid_group_schema(group, message) -> None:
    with pytest.raises(ConfigError, match=message):
        validate_split_config(group, "external", allow_groups=True)


def test_checkpoint_validation_checks_group_members() -> None:
    config = {
        "evaluate": {"task": "checkpoint"},
        "data": {"external": {"multi_hospital": {"members": {"wuhan": {}}}}},
    }
    with pytest.raises(ConfigError, match="data.external.multi_hospital.members.wuhan"):
        validate_config(config, "evaluate")


def test_hpo_configuration_has_no_external_cohorts() -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "hpo.yaml")
    validate_config(config, "hpo")
    assert not config["data"].get("external")


def test_external_epoch_selection_is_opt_in() -> None:
    assert external_selection_mode({}) == "validation_best"
    assert external_selection_mode({"evaluation": {}}) == "validation_best"
    path = Path(__file__).resolve().parents[1] / "configs/train.yaml"
    config = load_config(path)
    assert external_selection_mode(config) == "validation_best"
    config = load_config(path, ["evaluation.external_selection=per_cohort_best_auc"])
    validate_config(config, "train")
    assert external_selection_mode(config) == "per_cohort_best_auc"
    config = load_config(
        path,
        [
            "evaluation.external_selection=per_cohort_best_auc",
            "data.train.manifest=fold_{fold}_train.txt",
            "data.validation.manifest=fold_{fold}_validation.txt",
        ],
    )
    validate_config(config, "crossval")


@pytest.mark.parametrize("value", ["best", False, None, {}, []])
def test_invalid_external_epoch_selection_is_rejected(value) -> None:
    with pytest.raises(ConfigError, match="evaluation.external_selection"):
        external_selection_mode({"evaluation": {"external_selection": value}})


@pytest.mark.parametrize(
    ("filename", "command"), [("hpo.yaml", "hpo"), ("two_stage.yaml", "two-stage")]
)
def test_external_epoch_selection_is_not_supported_by_hpo_or_two_stage(filename, command):
    config = load_config(
        Path(__file__).resolve().parents[1] / "configs" / filename,
        ["evaluation.external_selection=per_cohort_best_auc"],
    )
    with pytest.raises(ConfigError, match="supported only by train and crossval"):
        validate_config(config, command)


@pytest.mark.parametrize(
    ("filename", "command"),
    [
        ("train.yaml", "train"),
        ("compare.yaml", "crossval"),
        ("two_stage.yaml", "two-stage"),
        ("hpo.yaml", "hpo"),
    ],
)
def test_checkpoint_export_defaults_to_disabled(filename, command):
    path = Path(__file__).resolve().parents[1] / "configs" / filename
    config = load_config(path)
    validate_config(config, command)
    assert config["output"]["save_checkpoint"] is False
    assert checkpoint_saving_enabled(config) is False
    enabled = load_config(path, ["output.save_checkpoint=true"])
    validate_config(enabled, command)
    assert checkpoint_saving_enabled(enabled) is True
    assert checkpoint_saving_enabled({}) is False


@pytest.mark.parametrize("value", [None, 0, 1, "false", "true", {}, []])
def test_checkpoint_export_requires_a_boolean(value):
    with pytest.raises(ConfigError, match="output.save_checkpoint must be a YAML boolean"):
        checkpoint_saving_enabled({"output": {"save_checkpoint": value}})
