from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

import petct.engine as engine
import petct.workflows as workflows
from petct.artifacts import save_predictions
from petct.metrics import binary_metrics


def test_fit_selects_highest_validation_auc_not_lowest_loss(tmp_path: Path, monkeypatch) -> None:
    model = torch.nn.Linear(1, 2)
    validation = iter(
        [
            ({"auc": 0.50, "loss": 0.30}, None, None),
            ({"auc": 0.80, "loss": 0.20}, None, None),
            ({"auc": 0.70, "loss": 0.10}, None, None),
        ]
    )
    monkeypatch.setattr(
        engine,
        "train_epoch",
        lambda *args: ({"auc": 0.5, "loss": 1.0}, None, None),
    )
    monkeypatch.setattr(engine, "evaluate_loader", lambda *args: next(validation))
    monkeypatch.setattr(
        engine,
        "save_checkpoint",
        lambda path, current, *args: torch.save(current.state_dict(), path),
    )
    logger = type("Logger", (), {"info": lambda *args, **kwargs: None})()
    result = engine.fit(
        model,
        [],
        [],
        {"device": "cpu", "training": {"epochs": 3, "early_stopping_patience": 0}},
        tmp_path,
        logger,
    )
    assert result["best_epoch"] == 2
    assert result["validation"]["auc"] == pytest.approx(0.8)


def test_external_loader_is_not_built_when_final_evaluation_is_disabled(
    tmp_path: Path, monkeypatch
) -> None:
    accesses: list[str] = []
    config = {
        "data": {
            "train": {"name": "train"},
            "validation": {"name": "validation"},
            "external": {"site": {"name": "external"}},
        },
        "model": {},
        "training": {},
        "output": {},
        "evaluation": {"external_after_training": False},
    }
    monkeypatch.setattr(workflows, "set_seed", lambda *args: None)
    monkeypatch.setattr(workflows, "configure_logging", lambda *args: object())
    monkeypatch.setattr(workflows, "save_resolved_config", lambda *args: None)
    monkeypatch.setattr(workflows, "save_json", lambda *args: None)
    monkeypatch.setattr(
        workflows,
        "loader_from_config",
        lambda root, split, training: accesses.append(split["name"]) or split["name"],
    )
    monkeypatch.setattr(workflows, "build_model", lambda config: object())
    monkeypatch.setattr(
        workflows,
        "fit",
        lambda *args: {
            "model": object(),
            "criterion": object(),
            "device": "cpu",
            "best_epoch": 1,
            "validation": {"auc": 0.8},
        },
    )
    workflows.run_training(config, run_directory=tmp_path)
    assert accesses == ["train", "validation"]


def test_predictions_are_opt_in(tmp_path: Path) -> None:
    import numpy as np

    path = tmp_path / "predictions.csv"
    save_predictions(path, np.array([0, 1]), np.array([0.2, 0.8]), False)
    assert not path.exists()
    save_predictions(path, np.array([0, 1]), np.array([0.2, 0.8]), True)
    assert path.read_text(encoding="utf-8").splitlines()[0] == "label,probability"
    assert "patient" not in path.read_text(encoding="utf-8").lower()


def test_epoch_observer_does_not_change_validation_checkpoint_or_history(tmp_path, monkeypatch):
    model = torch.nn.Linear(1, 2)
    epoch_count = 0
    validation = iter([0.5, 0.8, 0.7])
    observed = []

    def train_epoch(*args):
        nonlocal epoch_count
        epoch_count += 1
        with torch.no_grad():
            model.weight.fill_(epoch_count)
        return {"loss": 1.0}, None, None

    def observe(epoch, current_model, criterion, device):
        observed.append(epoch)
        assert current_model.weight[0, 0].item() == epoch
        assert str(device) == "cpu"

    monkeypatch.setattr(engine, "train_epoch", train_epoch)
    monkeypatch.setattr(
        engine,
        "evaluate_loader",
        lambda *args: ({"auc": next(validation), "loss": 0.2}, None, None),
    )
    logger = type("Logger", (), {"info": lambda *args, **kwargs: None})()
    result = engine.fit(
        model,
        [],
        [],
        {"device": "cpu", "training": {"epochs": 3, "early_stopping_patience": 0}},
        tmp_path,
        logger,
        epoch_observer=observe,
    )
    assert observed == [1, 2, 3]
    assert result["best_epoch"] == 2
    assert result["model"].weight[0, 0].item() == 2
    assert set(result["history"][0]) == {
        "epoch",
        "learning_rate",
        "train_loss",
        "validation_auc",
        "validation_loss",
    }


def _external_config(mode="validation_best"):
    return {
        "data": {
            "train": {"name": "train"},
            "validation": {"name": "validation"},
            "external": {name: {"name": name} for name in ("multi_hospital", "france", "SPH_test")},
        },
        "model": {},
        "training": {},
        "output": {"save_predictions": True, "save_prediction_ids": True},
        "evaluation": {"external_selection": mode},
    }


def test_workflow_exports_each_cohorts_best_epoch_without_changing_metrics_schema(
    tmp_path, monkeypatch
):
    import json

    import numpy as np
    import pandas as pd

    labels = np.array([0, 0, 1, 1])
    scores = {
        1.0: np.array([0.1, 0.2, 0.8, 0.9]),
        0.75: np.array([0.1, 0.7, 0.5, 0.9]),
        0.5: np.array([0.1, 0.9, 0.5, 0.7]),
    }
    curves = {
        "multi_hospital": [1.0, 0.75, 0.5],
        "france": [0.5, 1.0, 0.75],
        "SPH_test": [0.5, 0.75, 1.0],
    }
    validation_curve = [0.5, 0.8, 0.7]
    validation_reports, evaluations = [], []
    model = SimpleNamespace(epoch=0)

    def fake_fit(model, train, validation, config, directory, logger, callback, *, epoch_observer):
        for epoch, auc in enumerate(validation_curve, start=1):
            model.epoch = epoch
            callback(epoch, {"auc": auc, "loss": 0.2})
            epoch_observer(epoch, model, None, "cpu")
        model.epoch = 2  # Simulate fit restoring the validation-selected checkpoint.
        return dict(
            model=model, criterion=None, device="cpu", best_epoch=2, validation={"auc": 0.8}
        )

    def evaluate(model, cohort, criterion, device):
        evaluations.append((model.epoch, cohort))
        probabilities = scores[curves[cohort][model.epoch - 1]]
        return (
            dict(binary_metrics(labels, probabilities), loss=model.epoch / 10),
            labels,
            probabilities,
        )

    monkeypatch.setattr(workflows, "set_seed", lambda *args: None)
    monkeypatch.setattr(
        workflows, "loader_from_config", lambda root, split, training: split["name"]
    )
    monkeypatch.setattr(workflows, "build_model", lambda config: model)
    monkeypatch.setattr(workflows, "fit", fake_fit)
    monkeypatch.setattr(workflows, "evaluate_loader", evaluate)
    identifiers = ["003", "001", "004", "002"]
    monkeypatch.setattr(workflows, "prediction_identifiers", lambda loader, enabled: identifiers)
    result = workflows.run_training(
        _external_config("per_cohort_best_auc"),
        run_directory=tmp_path,
        epoch_callback=lambda epoch, metrics: validation_reports.append((epoch, metrics["auc"])),
    )
    assert validation_reports == list(enumerate(validation_curve, start=1))
    assert evaluations == [(epoch, name) for epoch in range(1, 4) for name in curves]
    payload = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert set(payload) == {"best_epoch", "validation", "external"}
    assert payload["best_epoch"] == result["best_epoch"] == 2
    for epoch, cohort in enumerate(curves, start=1):
        metrics = payload["external"][cohort]
        assert metrics["auc"] == 1.0
        assert metrics["loss"] == epoch / 10
        assert set(metrics) == set(binary_metrics(labels, scores[1.0])) | {"loss"}
        table = pd.read_csv(tmp_path / f"predictions_{cohort}.csv", dtype={"影像组学序列号": str})
        assert table["影像组学序列号"].tolist() == identifiers
        np.testing.assert_array_equal(table["预测概率"], scores[1.0])


def test_default_workflow_evaluates_external_only_after_fit(tmp_path, monkeypatch):
    import numpy as np

    events = []
    config = _external_config()
    config.pop("evaluation")  # Missing setting must retain the existing policy.
    config["output"]["save_predictions"] = False

    def fake_fit(*args, **kwargs):
        assert "epoch_observer" not in kwargs
        events.append("fit")
        return dict(model=None, criterion=None, device="cpu", best_epoch=2, validation={"auc": 0.8})

    def evaluate(model, cohort, criterion, device):
        events.append(cohort)
        return {"auc": 0.5}, np.array([0, 1]), np.array([0.2, 0.8])

    monkeypatch.setattr(workflows, "set_seed", lambda *args: None)
    monkeypatch.setattr(
        workflows, "loader_from_config", lambda root, split, training: split["name"]
    )
    monkeypatch.setattr(workflows, "build_model", lambda config: None)
    monkeypatch.setattr(workflows, "fit", fake_fit)
    monkeypatch.setattr(workflows, "evaluate_loader", evaluate)
    workflows.run_training(config, run_directory=tmp_path)
    assert events == ["fit", "multi_hospital", "france", "SPH_test"]


@pytest.mark.parametrize("disable", ["api", "config"])
def test_test_epoch_selection_does_not_bypass_external_evaluation_disable(
    tmp_path, monkeypatch, disable
):
    accesses = []
    config = _external_config("per_cohort_best_auc")
    if disable == "config":
        config["evaluation"]["external_after_training"] = False

    def fake_fit(*args, **kwargs):
        assert "epoch_observer" not in kwargs
        return dict(model=None, criterion=None, device="cpu", best_epoch=1, validation={"auc": 0.8})

    monkeypatch.setattr(workflows, "set_seed", lambda *args: None)
    monkeypatch.setattr(
        workflows,
        "loader_from_config",
        lambda root, split, training: accesses.append(split["name"]) or split["name"],
    )
    monkeypatch.setattr(workflows, "build_model", lambda config: None)
    monkeypatch.setattr(workflows, "fit", fake_fit)
    result = workflows.run_training(
        config, run_directory=tmp_path, evaluate_external=disable != "api"
    )
    assert accesses == ["train", "validation"]
    assert "external" not in result["results"]
