"""Validation export is an output-only operation on the restored best model."""

import json
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("torch")
pytest.importorskip("torchio")

import petct.workflows as workflows
from petct.metrics import binary_metrics


def _config():
    return {
        "data": {
            "train": {"name": "train"},
            "validation": {"name": "validation"},
            "external": {},
        },
        "model": {},
        "training": {},
        "evaluation": {"external_after_training": False},
        "output": {"save_validation_predictions": True, "save_training_predictions": False},
    }


def _mock_training(monkeypatch):
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.3, 0.4])
    best_metrics = dict(binary_metrics(labels, probabilities), loss=0.2)
    events = []

    def fit(model, *args, **kwargs):
        # Simulate restoring epoch 2 even though the last epoch was 3.
        model.epoch = 2
        events.append("restored_best")
        return dict(
            model=model, criterion=None, device="cpu", best_epoch=2, validation=best_metrics
        )

    def evaluate(model, loader, *args):
        assert model.epoch == 2 and loader == "validation"
        events.append("export_best_validation")
        return best_metrics, labels, probabilities

    monkeypatch.setattr(workflows, "set_seed", lambda *args: None)
    monkeypatch.setattr(
        workflows, "loader_from_config", lambda root, split, training: split["name"]
    )
    monkeypatch.setattr(workflows, "build_model", lambda config: SimpleNamespace(epoch=3))
    monkeypatch.setattr(workflows, "fit", fit)
    monkeypatch.setattr(workflows, "evaluate_loader", evaluate)
    monkeypatch.setattr(
        workflows,
        "prediction_identifiers",
        lambda loader, enabled: ["v1", "v2", "v3", "v4"] if enabled else None,
    )
    return events, labels, probabilities, best_metrics


@pytest.mark.parametrize("identified", [False, True])
@pytest.mark.parametrize("save_external_predictions", [False, True])
def test_validation_export_preserves_metrics_and_uses_restored_model(
    tmp_path, monkeypatch, identified, save_external_predictions
):
    events, labels, probabilities, best_metrics = _mock_training(monkeypatch)
    config = _config()
    config["output"].update(
        save_prediction_ids=identified, save_predictions=save_external_predictions
    )
    result = workflows.run_training(config, run_directory=tmp_path)
    assert events == ["restored_best", "export_best_validation"]
    assert result["validation"] == best_metrics
    payload = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert payload == {"best_epoch": 2, "validation": best_metrics}
    assert payload["validation"]["threshold"] == 0.5
    table = pd.read_csv(tmp_path / "predictions_validation.csv", float_precision="round_trip")
    if identified:
        assert table.columns.tolist() == [
            "影像组学序列号",
            "是否预测成功",
            "预测概率",
            "预测结果",
            "ground truth",
        ]
        assert table["预测结果"].tolist() == [0, 0, 0, 0]
        assert table["ground truth"].tolist() == labels.tolist()
        assert table["预测概率"].tolist() == probabilities.tolist()
    else:
        assert table.columns.tolist() == ["label", "probability"]
        assert table["label"].tolist() == labels.tolist()
        assert table["probability"].tolist() == probabilities.tolist()
    assert not (tmp_path / "threshold_analysis.json").exists()
    assert not (tmp_path / "best.pt").exists()


def test_validation_export_can_be_disabled(tmp_path, monkeypatch):
    events, _, _, _ = _mock_training(monkeypatch)
    config = _config()
    config["output"] = {
        "save_predictions": True,
        "save_validation_predictions": False,
        "save_training_predictions": False,
    }
    workflows.run_training(config, run_directory=tmp_path)
    assert events == ["restored_best"]
    assert not (tmp_path / "predictions_validation.csv").exists()


@pytest.mark.parametrize("output", [{}, {"save_predictions": False}, {"save_predictions": True}])
def test_validation_export_defaults_on_for_existing_configs(tmp_path, monkeypatch, output):
    events, _, _, _ = _mock_training(monkeypatch)
    config = _config()
    config["output"] = {**output, "save_training_predictions": False}
    workflows.run_training(config, run_directory=tmp_path)
    assert events == ["restored_best", "export_best_validation"]
    assert (tmp_path / "predictions_validation.csv").is_file()


def test_validation_export_runs_for_each_fold_in_its_own_directory(tmp_path, monkeypatch):
    events, _, probabilities, _ = _mock_training(monkeypatch)
    config = _config()
    config["cross_validation"] = {"folds": 5}
    config["data"]["train"]["manifest"] = "train_{fold}.txt"
    config["data"]["validation"]["manifest"] = "valid_{fold}.txt"
    monkeypatch.setattr(workflows, "prepare_run_directory", lambda config: tmp_path)
    payload = workflows.run_cross_validation(config)
    assert len(payload["folds"]) == 5
    assert events == ["restored_best", "export_best_validation"] * 5
    for fold in range(1, 6):
        table = pd.read_csv(tmp_path / f"fold_{fold}/predictions_validation.csv")
        np.testing.assert_allclose(table["probability"], probabilities)
        assert payload["folds"][fold - 1]["validation_threshold"] == 0.5
