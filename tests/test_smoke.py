import copy
import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
Image = pytest.importorskip("PIL.Image")
torch = pytest.importorskip("torch")
pytest.importorskip("torchio")
pytest.importorskip("sklearn")

from sklearn.linear_model import LogisticRegression

import petct.engine as engine
import petct.two_stage as two_stage
import petct.workflows as workflows
from petct.evaluation import run_evaluation
from petct.metrics import binary_metrics


class TinyPETCT(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = torch.nn.Linear(2, 2)

    def extract_features(self, ct, pet):
        return torch.stack([ct.mean(dim=(1, 2, 3)), pet.mean(dim=(1, 2, 3))], dim=1)

    def forward(self, ct, pet):
        return self.classifier(self.extract_features(ct, pet))


@pytest.mark.parametrize("settings, expected", [(None, 5), ({}, 5), ({"folds": 3}, 3)])
def test_cross_validation_default_fold_count(tmp_path: Path, monkeypatch, settings, expected):
    config = {
        "data": {
            "train": {"manifest": "fold_{fold}_train.txt"},
            "validation": {"manifest": "fold_{fold}_validation.txt"},
        },
    }
    if settings is not None:
        config["cross_validation"] = settings
    calls = []

    def mock_training(fold_config, *, run_directory):
        calls.append((fold_config, run_directory))
        return {"results": {"validation": {"auc": 0.75}}}

    monkeypatch.setattr(workflows, "prepare_run_directory", lambda config: tmp_path)
    monkeypatch.setattr(workflows, "run_training", mock_training)
    result = workflows.run_cross_validation(config)
    assert len(result["folds"]) == len(calls) == expected
    for fold, (fold_config, directory) in enumerate(calls, start=1):
        assert fold_config["data"]["train"]["manifest"] == f"fold_{fold}_train.txt"
        assert fold_config["data"]["validation"]["manifest"] == f"fold_{fold}_validation.txt"
        assert directory == tmp_path / f"fold_{fold}"
    assert config["data"]["train"]["manifest"] == "fold_{fold}_train.txt"


def _images(root: Path, identifiers: list[str], offset: int) -> None:
    for position, identifier in enumerate(identifiers):
        directory = root / identifier
        directory.mkdir(parents=True)
        for index in range(4):
            value = offset + position * 20 + index
            Image.fromarray(np.full((8, 8), value, dtype=np.uint8)).save(
                directory / f"{index:03d}.png"
            )


def test_cpu_micro_end_to_end_workflows(tmp_path: Path, monkeypatch) -> None:
    identifiers = ["1", "2", "3", "4", "5", "6"]
    labels = [0, 1, 0, 1, 0, 1]
    ct, pet = tmp_path / "ct", tmp_path / "pet"
    _images(ct, identifiers, 10)
    _images(pet, identifiers, 20)
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame({"id": identifiers, "label": labels}).to_csv(metadata, index=False)
    for fold in (1, 2):
        (tmp_path / f"train_{fold}.txt").write_text("1\n2\n3\n4\n", encoding="utf-8")
        (tmp_path / f"valid_{fold}.txt").write_text("5\n6\n", encoding="utf-8")
    config = {
        "_meta": {"config_dir": str(tmp_path)},
        "seed": 42,
        "device": "cpu",
        "data": {
            "id_column": "id",
            "label_column": "label",
            "depth": 4,
            "image_size": 8,
            "augment_probability": 0.0,
            "train": {
                "manifest": "train_1.txt",
                "metadata": "metadata.csv",
                "ct_root": "ct",
                "pet_root": "pet",
            },
            "validation": {
                "manifest": "valid_1.txt",
                "metadata": "metadata.csv",
                "ct_root": "ct",
                "pet_root": "pet",
            },
            "external": {},
        },
        "model": {"name": "tiny", "params": {}},
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
            "learning_rate": 0.01,
            "loss": {"name": "cross_entropy"},
        },
        "output": {"root": ".", "run_name": "train", "save_predictions": False},
    }
    monkeypatch.setattr(workflows, "build_model", lambda config: TinyPETCT())
    train_result = workflows.run_training(config)
    assert train_result["results"]["validation"]["auc"] is not None
    assert not (train_result["run_directory"] / "best.pt").exists()
    validation_table = pd.read_csv(train_result["run_directory"] / "predictions_validation.csv")
    assert validation_table.columns.tolist() == ["label", "probability"]
    assert validation_table["label"].tolist() == [0, 1]
    training_table = pd.read_csv(train_result["run_directory"] / "predictions_train.csv")
    assert training_table.columns.tolist() == ["label", "probability"]
    assert training_table["label"].tolist() == [0, 1, 0, 1]

    crossval_config = dict(config)
    crossval_config["data"] = dict(config["data"])
    crossval_config["data"]["train"] = dict(config["data"]["train"], manifest="train_{fold}.txt")
    crossval_config["data"]["validation"] = dict(
        config["data"]["validation"], manifest="valid_{fold}.txt"
    )
    crossval_config["cross_validation"] = {"folds": 2}
    crossval_config["output"] = {"root": ".", "run_name": "crossval", "save_predictions": False}
    assert len(workflows.run_cross_validation(crossval_config)["folds"]) == 2
    assert not list((tmp_path / "crossval").rglob("*.pt"))
    for fold in (1, 2):
        table = pd.read_csv(tmp_path / f"crossval/fold_{fold}/predictions_validation.csv")
        assert table["label"].tolist() == [0, 1]
        training_table = pd.read_csv(tmp_path / f"crossval/fold_{fold}/predictions_train.csv")
        assert training_table["label"].tolist() == [0, 1, 0, 1]

    monkeypatch.setattr(two_stage, "build_model", lambda config: TinyPETCT())
    monkeypatch.setattr(two_stage, "_tree_classifier", lambda config: LogisticRegression())
    two_config = dict(config)
    two_config["output"] = {"root": ".", "run_name": "two", "save_predictions": False}
    assert two_stage.run_two_stage(two_config)["validation"]["auc"] is not None
    assert not (tmp_path / "two_two_stage/best.pt").exists()
    tree_training_table = pd.read_csv(tmp_path / "two_two_stage/predictions_train.csv")
    assert tree_training_table["label"].tolist() == [0, 1, 0, 1]

    predictions = tmp_path / "predictions.csv"
    pd.DataFrame({"label": labels, "probability": np.linspace(0.1, 0.9, 6)}).to_csv(
        predictions, index=False
    )
    evaluation = run_evaluation(
        {
            "_meta": {"config_dir": str(tmp_path)},
            "evaluate": {
                "task": "metrics",
                "predictions": "predictions.csv",
                "output": "evaluation.json",
            },
        }
    )
    assert "auc" in evaluation


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("selection", ["validation_best", "per_cohort_best_auc"])
def test_final_external_predictions_and_checkpoint_export(
    tmp_path: Path, monkeypatch, grouped, selection
) -> None:
    identifiers = ["003", "001", "002", "004"]
    _images(tmp_path / "ct", identifiers, 10)
    _images(tmp_path / "pet", identifiers, 20)
    pd.DataFrame({"id": identifiers, "label": [0, 1, 0, 1]}).to_csv(
        tmp_path / "metadata.csv", index=False
    )
    (tmp_path / "cases.txt").write_text("\n".join(identifiers) + "\n", encoding="utf-8")
    split = {
        "manifest": "cases.txt",
        "metadata": "metadata.csv",
        "ct_root": "ct",
        "pet_root": "pet",
    }
    cohort_name = "multi_hospital" if grouped else "synthetic"
    external_source = split
    if grouped:
        members = {}
        for hospital, cases in zip(
            ("wuhan", "ruijin", "FUSCC"),
            (identifiers[:2], identifiers[2:3], identifiers[3:]),
            strict=True,
        ):
            manifest = f"{hospital}.txt"
            (tmp_path / manifest).write_text("\n".join(cases), encoding="utf-8")
            members[hospital] = dict(split, manifest=manifest)
        external_source = {"members": members}
    config = {
        "_meta": {"config_dir": str(tmp_path)},
        "device": "cpu",
        "data": {
            "id_column": "id",
            "label_column": "label",
            "depth": 4,
            "image_size": 8,
            "train": split,
            "validation": split,
            "external": {cohort_name: external_source},
        },
        "model": {"name": "tiny", "params": {}},
        "evaluation": {"external_selection": selection},
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
            "loss": {"name": "cross_entropy"},
        },
        "output": {
            "root": ".",
            "run_name": "train",
            "save_checkpoint": True,
            "save_predictions": True,
            "save_prediction_ids": True,
        },
    }
    monkeypatch.setattr(workflows, "build_model", lambda config: TinyPETCT())
    result = workflows.run_training(config)
    assert set(result["results"]["external"]) == {cohort_name}
    path = result["run_directory"] / f"predictions_{cohort_name}.csv"
    exported = pd.read_csv(path, dtype={"影像组学序列号": str})
    assert exported["影像组学序列号"].tolist() == identifiers
    assert exported["ground truth"].tolist() == [0, 1, 0, 1]
    expected_auc = binary_metrics(exported["ground truth"], exported["预测概率"])["auc"]
    assert result["results"]["external"][cohort_name]["auc"] == pytest.approx(expected_auc)
    assert (
        exported["是否预测成功"].tolist()
        == (exported["预测结果"] == exported["ground truth"]).astype(int).tolist()
    )

    import petct.models as models

    monkeypatch.setattr(models, "build_model", lambda config: TinyPETCT())
    checkpoint_config = dict(
        config,
        evaluation={"external_selection": "validation_best"},
        evaluate={
            "task": "checkpoint",
            "checkpoint": str(result["run_directory"] / "best.pt"),
            "output": "final_metrics.json",
            "prediction_directory": "final_predictions",
            "save_predictions": True,
            "save_prediction_ids": True,
        },
    )
    run_evaluation(checkpoint_config)
    final_table = pd.read_csv(
        tmp_path / f"final_predictions/{cohort_name}.csv", dtype={"影像组学序列号": str}
    )
    assert final_table["影像组学序列号"].tolist() == identifiers
    np.testing.assert_allclose(final_table["预测概率"], exported["预测概率"])

    monkeypatch.setattr(two_stage, "build_model", lambda config: TinyPETCT())
    monkeypatch.setattr(two_stage, "_tree_classifier", lambda config: LogisticRegression())
    two_config = dict(
        config,
        output=dict(config["output"], run_name="two"),
        evaluation={"external_selection": "validation_best"},
    )
    tree_result = two_stage.run_two_stage(two_config)
    tree_table = pd.read_csv(
        tmp_path / f"two_two_stage/predictions_{cohort_name}.csv", dtype={"影像组学序列号": str}
    )
    assert tree_table["影像组学序列号"].tolist() == identifiers
    assert tree_table["ground truth"].tolist() == [0, 1, 0, 1]
    expected_tree_auc = binary_metrics(tree_table["ground truth"], tree_table["预测概率"])["auc"]
    assert tree_result["external"][cohort_name]["auc"] == pytest.approx(expected_tree_auc)


@pytest.mark.parametrize("selection", ["validation_best", "per_cohort_best_auc"])
@pytest.mark.parametrize("individual_counts", [False, True])
def test_leakage_end_to_end_preserves_all_output_schemas(
    tmp_path, monkeypatch, selection, individual_counts
):
    identifiers = [str(index) for index in range(1, 13)]
    _images(tmp_path / "ct", identifiers, 0)
    _images(tmp_path / "pet", identifiers, 0)
    pd.DataFrame({"id": identifiers, "label": [0, 1] * 6}).to_csv(
        tmp_path / "metadata.csv", index=False
    )

    def source(name, cases):
        manifest = tmp_path / f"{name}.txt"
        manifest.write_text("\n".join(cases) + "\n", encoding="utf-8")
        return dict(manifest=manifest.name, metadata="metadata.csv", ct_root="ct", pet_root="pet")

    training = source("train", identifiers[:4])
    validation = source("valid", identifiers[4:6])
    external = {
        "multi_hospital": {
            "members": {
                "a": source("a", identifiers[6:7]),
                "b": source("b", identifiers[7:8]),
            }
        },
        "france": source("france", identifiers[8:10]),
        "SPH_test": source("SPH_test", identifiers[10:]),
    }
    original_manifests = {path.name: path.read_bytes() for path in tmp_path.glob("*.txt")}
    config = {
        "_meta": {"config_dir": str(tmp_path)},
        "seed": 42,
        "device": "cpu",
        "model": {"name": "tiny"},
        "data": {
            "id_column": "id",
            "label_column": "label",
            "depth": 4,
            "image_size": 8,
            "augment_probability": 0.0,
            "train": training,
            "validation": validation,
            "external": external,
        },
        "training": {
            "epochs": 3,
            "early_stopping_patience": 0,
            "batch_size": 2,
            "num_workers": 0,
            "loss": {"name": "cross_entropy"},
            "leakage_simulation": {"enabled": True, "samples_per_cohort": 1},
        },
        "evaluation": {"external_selection": selection},
        "output": {
            "root": ".",
            "run_name": "simulation",
            "save_predictions": True,
            "save_prediction_ids": True,
            "save_validation_predictions": False,
        },
    }
    if individual_counts:
        config["training"]["leakage_simulation"]["cohort_samples"] = {
            "multi_hospital": 2,
            "france": 1,
            "SPH_test": 0,
        }
    train_epoch = engine.train_epoch
    draws = []

    def record_training(model, loader, *args):
        assert len(loader.dataset) == 7  # Original 4 plus 3 total drawn test cases.
        assert loader.dataset.datasets[0].training is True
        subsets = loader.dataset.datasets[1:]
        assert [len(subset) for subset in subsets] == ([2, 1] if individual_counts else [1, 1, 1])
        for subset in subsets:
            dataset = subset.dataset
            members = dataset.datasets if hasattr(dataset, "datasets") else [dataset]
            assert all(member.training is True for member in members)
        draws.append([tuple(subset.indices) for subset in subsets])
        return train_epoch(model, loader, *args)

    monkeypatch.setattr(engine, "train_epoch", record_training)
    monkeypatch.setattr(workflows, "build_model", lambda config: TinyPETCT())
    evaluate_loader = workflows.evaluate_loader
    training_exports = []

    def record_evaluation(model, loader, *args):
        dataset = loader.dataset
        manifest = getattr(dataset, "manifest_path", None)
        if manifest is not None and manifest.stem.startswith("train"):
            assert dataset.training is False
            assert dataset.augment_probability == 0.0
            assert isinstance(loader.sampler, torch.utils.data.SequentialSampler)
            assert not loader.drop_last
            assert dataset.identifiers == identifiers[:4]
            training_exports.append(list(dataset.identifiers))
        return evaluate_loader(model, loader, *args)

    monkeypatch.setattr(workflows, "evaluate_loader", record_evaluation)
    result = workflows.run_training(config)
    assert len(draws) == 3
    assert len(set(map(tuple, draws))) > 1
    directory = result["run_directory"]
    assert directory == tmp_path / "simulation"  # No automatic directory suffix.
    assert {path.name for path in directory.iterdir()} == {
        "resolved_config.yaml",
        "run.log",
        "history.csv",
        "metrics.json",
        "predictions_train.csv",
        *{f"predictions_{name}.csv" for name in external},
    }
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    assert set(metrics) == {"best_epoch", "validation", "external"}
    assert set(metrics["external"]) == set(external)
    training_table = pd.read_csv(directory / "predictions_train.csv", dtype={"影像组学序列号": str})
    assert training_table["影像组学序列号"].tolist() == identifiers[:4]
    assert not set(training_table["影像组学序列号"]).intersection(identifiers[4:])
    assert training_exports == [identifiers[:4]]
    for name, cases in zip(
        external, [identifiers[6:8], identifiers[8:10], identifiers[10:]], strict=True
    ):
        table = pd.read_csv(directory / f"predictions_{name}.csv", dtype={"影像组学序列号": str})
        assert table.columns.tolist() == [
            "影像组学序列号",
            "是否预测成功",
            "预测概率",
            "预测结果",
            "ground truth",
        ]
        assert table["影像组学序列号"].tolist() == cases  # No drawn cases removed from testing.
        assert set(metrics["external"][name]) == set(binary_metrics([0, 1], [0.2, 0.8])) | {"loss"}
    history = pd.read_csv(directory / "history.csv")
    assert len(history) == 3
    assert not any("leakage" in column for column in history.columns)
    assert "LEAKAGE SIMULATION ENABLED" in (directory / "run.log").read_text(encoding="utf-8")
    assert "leakage_simulation:" in (directory / "resolved_config.yaml").read_text(encoding="utf-8")
    assert {path.name: path.read_bytes() for path in tmp_path.glob("*.txt")} == original_manifests

    crossval = copy.deepcopy(config)
    repeats = 5
    crossval["cross_validation"] = {"folds": repeats}
    crossval["output"]["run_name"] = "repeats"
    for split, cases in (("train", identifiers[:4]), ("validation", identifiers[4:6])):
        for fold in range(1, repeats + 1):
            source(f"{split}_{fold}", cases)
        crossval["data"][split]["manifest"] = f"{split}_{{fold}}.txt"
    draws.clear()
    payload = workflows.run_cross_validation(crossval)
    assert len(payload["folds"]) == repeats
    assert set(payload) == {"folds", "summary"}
    assert len(draws) == repeats * 3
    assert training_exports == [identifiers[:4]] * (repeats + 1)
    assert all(draws[fold * 3 : (fold + 1) * 3] == draws[:3] for fold in range(repeats))
    assert {path.name for path in (tmp_path / "repeats").iterdir()} == {
        "resolved_config.yaml",
        "cross_validation.json",
        *{f"fold_{fold}" for fold in range(1, repeats + 1)},
    }
    for fold in range(1, repeats + 1):
        training_table = pd.read_csv(
            tmp_path / f"repeats/fold_{fold}/predictions_train.csv", dtype={"影像组学序列号": str}
        )
        assert training_table["影像组学序列号"].tolist() == identifiers[:4]
        assert {path.name for path in (tmp_path / f"repeats/fold_{fold}").iterdir()} == {
            path.name for path in directory.iterdir()
        }
