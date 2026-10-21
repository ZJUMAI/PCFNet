from pathlib import Path

import pytest

pytest.importorskip("yaml")

from petct.config import (
    ConfigError,
    apply_overrides,
    checkpoint_saving_enabled,
    external_selection_mode,
    leakage_simulation_settings,
    load_config,
    resolve_config_path,
    training_prediction_export_enabled,
    validate_config,
    validate_split_config,
    validation_prediction_export_enabled,
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
    [
        ("train.yaml", "train"),
        ("train_spatial.yaml", "train"),
        ("train_spatial_70_30.yaml", "crossval"),
        ("two_stage.yaml", "two-stage"),
        ("evaluate.yaml", "evaluate"),
    ],
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
        ("train_spatial.yaml", "train"),
        ("train_spatial_70_30.yaml", "crossval"),
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


@pytest.mark.parametrize(
    "filename",
    [
        "train.yaml",
        "train_spatial.yaml",
        "train_spatial_70_30.yaml",
        "compare.yaml",
        "hpo.yaml",
        "two_stage.yaml",
        "evaluate.yaml",
        "split.yaml",
        "split_70_30.yaml",
    ],
)
def test_all_example_random_seeds_are_42(filename):
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / filename)
    if filename.startswith("split"):
        assert config["split"]["seed"] == 42
    else:
        assert config["seed"] == 42
    if filename == "hpo.yaml":
        assert config["hpo"]["seed"] == 42


def test_spatial_experiment_changes_only_model_and_run_name():
    root = Path(__file__).resolve().parents[1] / "configs"
    original = load_config(root / "train.yaml")
    spatial = load_config(root / "train_spatial.yaml")
    validate_config(spatial, "train")
    assert original["model"]["name"] == "fusion3d"
    assert spatial["model"] == {
        "name": "fusion3d_spatial",
        "params": {
            "num_classes": 2,
            "dropout": 0.5,
            "embedding_dim": 128,
            "num_heads": 4,
            "attention_dropout": 0.1,
            "pretrained": True,
        },
    }
    assert spatial["data"] == original["data"]
    assert spatial["training"] == original["training"]
    assert spatial["evaluation"] == original["evaluation"]
    assert spatial["output"] == dict(original["output"], run_name="fusion3d_spatial")
    assert spatial["seed"] == original["seed"] == 42
    assert spatial["device"] == original["device"]


def test_split_schema_rejects_invalid_repeat_count_before_metadata_loading():
    config = load_config(
        Path(__file__).resolve().parents[1] / "configs/split_70_30.yaml",
        ["split.repeats=0"],
    )
    with pytest.raises(ConfigError, match="split.repeats"):
        validate_config(config, "split")


@pytest.mark.parametrize(
    "filename", ["train.yaml", "train_spatial.yaml", "train_spatial_70_30.yaml", "compare.yaml"]
)
def test_leakage_simulation_is_disabled_by_default(filename):
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / filename)
    assert config["training"]["leakage_simulation"]["enabled"] is False
    assert leakage_simulation_settings(config) is None
    assert leakage_simulation_settings({}) is None


@pytest.mark.parametrize("command", ["train", "crossval"])
def test_leakage_simulation_overrides_and_cohort_selection(command):
    path = Path(__file__).resolve().parents[1] / "configs/train_spatial_70_30.yaml"
    overrides = [
        "training.leakage_simulation.enabled=true",
        "training.leakage_simulation.samples_per_cohort=2",
        "evaluation.external_selection=per_cohort_best_auc",
    ]
    config = load_config(path, overrides)
    validate_config(config, command)
    assert leakage_simulation_settings(config) == {
        "samples_per_cohort": 2,
        "cohorts": ["multi_hospital", "france", "SPH_test"],
        "cohort_samples": {"multi_hospital": 2, "france": 2, "SPH_test": 2},
    }
    config = load_config(path, [*overrides, "training.leakage_simulation.cohorts=[france]"])
    assert leakage_simulation_settings(config)["cohorts"] == ["france"]


@pytest.mark.parametrize(
    "settings",
    [
        None,
        [],
        {"enabled": "false"},
        {"enabled": 1},
        {"samples_per_cohort": 0},
        {"samples_per_cohort": -1},
        {"samples_per_cohort": True},
        {"samples_per_cohort": 1.5},
        {"samples_per_cohort": "3"},
        {"cohorts": "france"},
        {"cohorts": []},
        {"cohorts": ["france", "france"]},
        {"cohorts": [{}]},
        {"cohorts": [""]},
        {"enabled": True, "cohorts": ["missing"]},
        {"resample_each_epoch": False},
    ],
)
def test_leakage_simulation_rejects_invalid_settings(settings):
    config = {"training": {"leakage_simulation": settings}, "data": {"external": {"france": {}}}}
    with pytest.raises(ConfigError, match="leakage_simulation"):
        leakage_simulation_settings(config)


@pytest.mark.parametrize("external", [None, {}, []])
def test_leakage_simulation_requires_external_cohorts(external):
    config = {
        "training": {"leakage_simulation": {"enabled": True}},
        "data": {"external": external},
    }
    with pytest.raises(ConfigError, match="data.external"):
        leakage_simulation_settings(config)


def test_leakage_simulation_requires_external_access():
    config = {
        "training": {"leakage_simulation": {"enabled": True}},
        "data": {"external": {"france": {}}},
        "evaluation": {"external_after_training": False},
    }
    with pytest.raises(ConfigError, match="external_after_training=true"):
        leakage_simulation_settings(config)


@pytest.mark.parametrize("command", ["hpo", "two-stage"])
def test_leakage_simulation_is_rejected_outside_training_commands(command):
    config = load_config(
        Path(__file__).resolve().parents[1] / "configs/train.yaml",
        ["training.leakage_simulation.enabled=true"],
    )
    with pytest.raises(ConfigError, match="supported only by train and crossval"):
        validate_config(config, command)


def test_leakage_counts_can_be_set_independently_with_cli_overrides():
    path = Path(__file__).resolve().parents[1] / "configs/train.yaml"
    config = load_config(
        path,
        [
            "training.leakage_simulation.enabled=true",
            "training.leakage_simulation.samples_per_cohort=10",
            "training.leakage_simulation.cohort_samples.multi_hospital=5",
            "training.leakage_simulation.cohort_samples.france=2",
            "training.leakage_simulation.cohort_samples.SPH_test=0",
        ],
    )
    validate_config(config, "train")
    assert leakage_simulation_settings(config)["cohort_samples"] == {
        "multi_hospital": 5,
        "france": 2,
        "SPH_test": 0,
    }


def test_leakage_individual_counts_fall_back_to_uniform_count():
    path = Path(__file__).resolve().parents[1] / "configs/train.yaml"
    config = load_config(
        path,
        [
            "training.leakage_simulation.enabled=true",
            "training.leakage_simulation.samples_per_cohort=10",
            "training.leakage_simulation.cohort_samples.france=0",
        ],
    )
    assert leakage_simulation_settings(config)["cohort_samples"] == {
        "multi_hospital": 10,
        "france": 0,
        "SPH_test": 10,
    }


@pytest.mark.parametrize(
    "overrides",
    [
        None,
        [],
        True,
        {"france": -1},
        {"france": True},
        {"france": 1.5},
        {"france": "2"},
        {"": 1},
        {1: 2},
    ],
)
def test_leakage_individual_counts_reject_invalid_types(overrides):
    config = {"training": {"leakage_simulation": {"cohort_samples": overrides}}}
    with pytest.raises(ConfigError, match="cohort_samples"):
        leakage_simulation_settings(config)


@pytest.mark.parametrize(
    "settings, message",
    [
        ({"cohort_samples": {"missing": 1}}, "name groups in data.external"),
        ({"cohorts": ["france"], "cohort_samples": {"SPH_test": 1}}, "unselected cohorts"),
        ({"cohort_samples": {"multi_hospital": 0, "france": 0, "SPH_test": 0}}, "positive count"),
    ],
)
def test_leakage_individual_counts_reject_invalid_selection(settings, message):
    config = load_config(Path(__file__).resolve().parents[1] / "configs/train.yaml")
    config["training"]["leakage_simulation"].update(settings, enabled=True)
    with pytest.raises(ConfigError, match=message):
        leakage_simulation_settings(config)


@pytest.mark.parametrize(
    "filename", ["train.yaml", "train_spatial.yaml", "train_spatial_70_30.yaml", "compare.yaml"]
)
def test_main_training_configs_export_validation_predictions(filename):
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / filename)
    assert validation_prediction_export_enabled(config) is True


def test_hpo_config_keeps_validation_export_off():
    config = load_config(Path(__file__).resolve().parents[1] / "configs/hpo.yaml")
    assert validation_prediction_export_enabled(config) is False


def test_validation_export_defaults_on_and_can_be_disabled():
    assert validation_prediction_export_enabled({}) is True
    assert validation_prediction_export_enabled({"output": {}}) is True
    assert (
        validation_prediction_export_enabled({"output": {"save_validation_predictions": False}})
        is False
    )


@pytest.mark.parametrize("value", [None, 1, "true", [], {}])
def test_validation_export_requires_boolean(value):
    with pytest.raises(ConfigError, match="save_validation_predictions"):
        validation_prediction_export_enabled({"output": {"save_validation_predictions": value}})


@pytest.mark.parametrize(
    "filename",
    [
        "train.yaml",
        "train_spatial.yaml",
        "train_spatial_70_30.yaml",
        "compare.yaml",
        "two_stage.yaml",
    ],
)
def test_training_configs_export_original_training_predictions(filename):
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / filename)
    assert training_prediction_export_enabled(config) is True


def test_training_export_defaults_on_and_can_be_disabled():
    assert training_prediction_export_enabled({}) is True
    assert training_prediction_export_enabled({"output": {}}) is True
    assert (
        training_prediction_export_enabled({"output": {"save_training_predictions": False}})
        is False
    )
    config = load_config(Path(__file__).resolve().parents[1] / "configs/hpo.yaml")
    assert training_prediction_export_enabled(config) is False


@pytest.mark.parametrize("value", [None, 1, "true", [], {}])
def test_training_export_requires_boolean(value):
    with pytest.raises(ConfigError, match="save_training_predictions"):
        training_prediction_export_enabled({"output": {"save_training_predictions": value}})
