from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
Image = pytest.importorskip("PIL.Image")
torch = pytest.importorskip("torch")
pytest.importorskip("torchio")
pytest.importorskip("sklearn")

from sklearn.linear_model import LogisticRegression

import petct.two_stage as two_stage
import petct.workflows as workflows
from petct.evaluation import run_evaluation


class TinyPETCT(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = torch.nn.Linear(2, 2)

    def extract_features(self, ct, pet):
        return torch.stack([ct.mean(dim=(1, 2, 3)), pet.mean(dim=(1, 2, 3))], dim=1)

    def forward(self, ct, pet):
        return self.classifier(self.extract_features(ct, pet))


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
        "seed": 4,
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

    crossval_config = dict(config)
    crossval_config["data"] = dict(config["data"])
    crossval_config["data"]["train"] = dict(config["data"]["train"], manifest="train_{fold}.txt")
    crossval_config["data"]["validation"] = dict(
        config["data"]["validation"], manifest="valid_{fold}.txt"
    )
    crossval_config["cross_validation"] = {"folds": 2}
    crossval_config["output"] = {"root": ".", "run_name": "crossval", "save_predictions": False}
    assert len(workflows.run_cross_validation(crossval_config)["folds"]) == 2

    monkeypatch.setattr(two_stage, "build_model", lambda config: TinyPETCT())
    monkeypatch.setattr(two_stage, "_tree_classifier", lambda config: LogisticRegression())
    two_config = dict(config)
    two_config["output"] = {"root": ".", "run_name": "two", "save_predictions": False}
    assert two_stage.run_two_stage(two_config)["validation"]["auc"] is not None

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
