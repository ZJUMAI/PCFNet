import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from petct.config import ConfigError, load_config, resolve_config_path
from petct.splits import run_split


def _config(tmp_path: Path) -> dict:
    pd.DataFrame(
        {"patient_id": [f"case_{i:03d}" for i in range(100)], "label": [0] * 70 + [1] * 30}
    ).to_csv(tmp_path / "metadata.csv", index=False)
    return {
        "_meta": {"config_dir": str(tmp_path)},
        "split": {
            "metadata": "metadata.csv",
            "id_column": "patient_id",
            "label_column": "label",
            "mode": "holdout",
            "validation_fraction": 0.2,
            "seed": 513,
            "output_directory": "splits",
        },
    }


def _ids(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def test_holdout_is_stratified_disjoint_complete_and_reproducible(tmp_path: Path) -> None:
    config = _config(tmp_path)
    summary = run_split(config)
    train_path = tmp_path / "splits/train.txt"
    validation_path = tmp_path / "splits/validation.txt"
    train, validation = _ids(train_path), _ids(validation_path)
    assert len(train) == len(set(train)) == 80
    assert len(validation) == len(set(validation)) == 20
    assert set(train).isdisjoint(validation)
    assert set(train) | set(validation) == {f"case_{i:03d}" for i in range(100)}
    frame = pd.read_csv(tmp_path / "metadata.csv").set_index("patient_id")
    assert frame.loc[train, "label"].sum() == 24
    assert frame.loc[validation, "label"].sum() == 6
    assert summary["train"] == {"cases": 80, "positive": 24, "negative": 56}
    assert summary["validation"] == {"cases": 20, "positive": 6, "negative": 14}
    assert json.loads((tmp_path / "splits/summary.json").read_text()) == summary
    run_split(config)
    assert _ids(train_path) == train
    assert _ids(validation_path) == validation


def test_configs_without_mode_or_fold_count_default_to_five_folds(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["split"].pop("mode")
    summary = run_split(config)
    assert summary["folds"] == 5
    validation_ids = []
    for fold in range(1, 6):
        train = _ids(tmp_path / f"splits/fold_{fold}_train.txt")
        validation = _ids(tmp_path / f"splits/fold_{fold}_validation.txt")
        assert len(train) == 80
        assert len(validation) == 20
        assert set(train).isdisjoint(validation)
        assert set(train) | set(validation) == {f"case_{i:03d}" for i in range(100)}
        frame = pd.read_csv(tmp_path / "metadata.csv").set_index("patient_id")
        assert frame.loc[train, "label"].sum() == 24
        assert frame.loc[validation, "label"].sum() == 6
        validation_ids.extend(validation)
    assert len(set(validation_ids)) == len(validation_ids) == 100


def test_explicit_fold_count_remains_configurable(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["split"].update(mode="kfold", folds=4)
    assert run_split(config)["folds"] == 4
    assert len(list((tmp_path / "splits").glob("fold_*_validation.txt"))) == 4


def test_default_split_yaml_generates_five_folds(tmp_path: Path) -> None:
    _config(tmp_path)
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/split.yaml")
    config["split"].update(
        metadata=str(tmp_path / "metadata.csv"),
        id_column="patient_id",
        label_column="label",
        source_manifests=None,
        output_directory=str(tmp_path / "generated"),
    )
    assert run_split(config)["folds"] == 5
    assert len(list((tmp_path / "generated").glob("*.txt"))) == 10
    for fold in range(1, 6):
        assert len(_ids(tmp_path / f"generated/fold_{fold}_train.txt")) == 80
        assert len(_ids(tmp_path / f"generated/fold_{fold}_validation.txt")) == 20


def test_source_manifests_restrict_split_to_existing_cohort(tmp_path: Path) -> None:
    config = _config(tmp_path)
    train_ids = [f"case_{i:03d}" for i in range(0, 100, 2)]
    first, second = train_ids[:25], train_ids[25:]
    (tmp_path / "existing_train.txt").write_text("\n".join(first) + "\n", encoding="utf-8")
    (tmp_path / "existing_valid.txt").write_text("\n".join(second) + "\n", encoding="utf-8")
    config["split"]["source_manifests"] = ["existing_train.txt", "existing_valid.txt"]
    summary = run_split(config)
    selected = _ids(tmp_path / "splits/train.txt") + _ids(tmp_path / "splits/validation.txt")
    assert set(selected) == set(train_ids)
    assert len(selected) == summary["cases"] == 50
    assert summary["train"]["cases"] == 40
    assert summary["validation"]["cases"] == 10


def test_missing_source_manifest_labels_fail_before_writing(tmp_path: Path) -> None:
    config = _config(tmp_path)
    (tmp_path / "existing.txt").write_text("missing_case\n", encoding="utf-8")
    config["split"]["source_manifests"] = ["existing.txt"]
    with pytest.raises(ConfigError, match="absent"):
        run_split(config)
    assert not (tmp_path / "splits").exists()


@pytest.mark.parametrize("fraction", [0, 1, -0.2, 1.2])
def test_invalid_holdout_ratio_creates_no_outputs(tmp_path: Path, fraction: float) -> None:
    config = _config(tmp_path)
    config["split"]["validation_fraction"] = fraction
    with pytest.raises(ValueError, match="validation_fraction"):
        run_split(config)
    assert not (tmp_path / "splits").exists()


@pytest.mark.parametrize("invalid", ["duplicate", "empty", "single_class", "unknown_label"])
def test_invalid_metadata_creates_no_outputs(tmp_path: Path, invalid: str) -> None:
    config = _config(tmp_path)
    frame = pd.read_csv(tmp_path / "metadata.csv")
    if invalid == "duplicate":
        frame.loc[1, "patient_id"] = frame.loc[0, "patient_id"]
    elif invalid == "empty":
        frame.loc[0, "patient_id"] = " "
    elif invalid == "single_class":
        frame["label"] = 0
    else:
        frame["label"] = "unknown"
    frame.to_csv(tmp_path / "metadata.csv", index=False)
    with pytest.raises((ConfigError, ValueError)):
        run_split(config)
    assert not (tmp_path / "splits").exists()


@pytest.mark.parametrize("filename", ["metadata.csv", "metadata.xlsx"])
def test_numeric_ids_match_training_normalization(tmp_path: Path, filename: str) -> None:
    config = _config(tmp_path)
    config["split"]["metadata"] = filename
    frame = pd.DataFrame({"patient_id": [float(i) for i in range(100)], "label": [0, 1] * 50})
    if filename.endswith(".xlsx"):
        pytest.importorskip("openpyxl")
        frame.to_excel(tmp_path / filename, index=False)
    else:
        frame.to_csv(tmp_path / filename, index=False)
    run_split(config)
    identifiers = _ids(tmp_path / "splits/train.txt") + _ids(tmp_path / "splits/validation.txt")
    assert set(identifiers) == {str(i) for i in range(100)}


def test_split_cli_runs_without_importing_training(tmp_path: Path) -> None:
    config = _config(tmp_path)
    path = tmp_path / "split.yaml"
    path.write_text(yaml.safe_dump({"split": config["split"]}), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "petct", "split", "--config", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert len(_ids(tmp_path / "splits/train.txt")) == 80
    assert len(_ids(tmp_path / "splits/validation.txt")) == 20


def test_example_configurations_use_generated_five_fold_paths() -> None:
    root = Path(__file__).resolve().parents[1]
    split = load_config(root / "configs/split.yaml")
    output = resolve_config_path(split, split["split"]["output_directory"])
    assert split["split"]["mode"] == "kfold"
    assert split["split"]["folds"] == 5
    pattern = split["split"]["filename_pattern"]
    compare = load_config(root / "configs/compare.yaml")
    assert compare["cross_validation"]["folds"] == 5
    for fold in range(1, 6):
        for name in ("train", "validation"):
            manifest = compare["data"][name]["manifest"].format(fold=fold)
            assert resolve_config_path(compare, manifest) == output / pattern.format(
                fold=fold, split=name
            )
    for filename in ("train.yaml", "hpo.yaml", "two_stage.yaml"):
        config = load_config(root / "configs" / filename)
        for name in ("train", "validation"):
            assert resolve_config_path(config, config["data"][name]["manifest"]) == (
                output / pattern.format(fold=1, split=name)
            )
