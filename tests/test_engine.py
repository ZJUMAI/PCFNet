from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

import petct.engine as engine
import petct.workflows as workflows
from petct.artifacts import save_predictions, snapshot_state_dict
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
        lambda *args: pytest.fail("Checkpoint export must be disabled by default"),
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
    assert not (tmp_path / "best.pt").exists()


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


@pytest.mark.parametrize("save_checkpoint", [None, False, True])
@pytest.mark.parametrize("stale_checkpoint", [False, True])
def test_epoch_observer_does_not_change_best_weights_or_history(
    tmp_path, monkeypatch, save_checkpoint, stale_checkpoint
):
    model = torch.nn.Linear(1, 2)
    model.register_buffer("steps", torch.tensor(0))
    epoch_count = 0
    validation = iter([0.5, 0.8, 0.7])
    observed = []

    def train_epoch(*args):
        nonlocal epoch_count
        epoch_count += 1
        with torch.no_grad():
            model.weight.fill_(epoch_count)
            model.steps.fill_(epoch_count)
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
    checkpoint_path = tmp_path / "best.pt"
    stale_bytes = b"pre-existing checkpoint must not be loaded or deleted"
    if stale_checkpoint:
        checkpoint_path.write_bytes(stale_bytes)
    config = {"device": "cpu", "training": {"epochs": 3, "early_stopping_patience": 0}}
    if save_checkpoint is not None:
        config["output"] = {"save_checkpoint": save_checkpoint}
    original_load = torch.load
    monkeypatch.setattr(
        torch, "load", lambda *args, **kwargs: pytest.fail("fit must restore from memory")
    )
    result = engine.fit(
        model,
        [],
        [],
        config,
        tmp_path,
        logger,
        epoch_observer=observe,
    )
    assert observed == [1, 2, 3]
    assert result["best_epoch"] == 2
    assert result["model"].weight[0, 0].item() == 2
    assert result["model"].steps.item() == 2
    assert checkpoint_path.exists() == (stale_checkpoint or bool(save_checkpoint))
    if save_checkpoint:
        state = original_load(checkpoint_path, map_location="cpu", weights_only=True)
        assert state["weight"][0, 0].item() == state["steps"].item() == 2
    elif stale_checkpoint:
        assert checkpoint_path.read_bytes() == stale_bytes
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


def test_state_snapshot_copies_cpu_tensors_buffers_extra_state_and_metadata():
    class StatefulModel(torch.nn.Linear):
        def __init__(self):
            super().__init__(1, 2)
            self.payload = {"values": [1]}
            self.register_buffer("counter", torch.tensor(1))

        def get_extra_state(self):
            return self.payload

        def set_extra_state(self, state):
            self.payload = state

    model = StatefulModel()
    snapshot = snapshot_state_dict(model)
    original_weight = snapshot["weight"].clone()
    assert snapshot["weight"].device.type == "cpu"
    assert snapshot["weight"].data_ptr() != model.weight.data_ptr()
    assert snapshot._metadata == model.state_dict()._metadata
    with torch.no_grad():
        model.weight.fill_(9)
        model.counter.fill_(9)
    model.payload["values"][0] = 9
    assert torch.equal(snapshot["weight"], original_weight)
    assert snapshot["counter"].item() == 1
    assert snapshot["_extra_state"] == {"values": [1]}
    model.load_state_dict(snapshot)
    assert torch.equal(model.weight, original_weight)
    assert model.counter.item() == 1
    assert model.payload == {"values": [1]}


def test_zero_epochs_cannot_restore_a_stale_checkpoint(tmp_path):
    (tmp_path / "best.pt").write_bytes(b"old checkpoint")
    logger = type("Logger", (), {"info": lambda *args, **kwargs: None})()
    with pytest.raises(RuntimeError, match="did not produce a best model"):
        engine.fit(
            torch.nn.Linear(1, 2),
            [],
            [],
            {"device": "cpu", "training": {"epochs": 0}},
            tmp_path,
            logger,
        )


def test_fit_rebuilds_only_training_loader_before_each_epoch(tmp_path, monkeypatch):
    events = []
    original_loader, validation_loader = object(), object()

    def epoch_loader(epoch):
        events.append(("draw", epoch))
        return epoch

    def train(model, loader, *args):
        events.append(("train", loader))
        return {"loss": 0.4}, None, None

    def evaluate(model, loader, *args):
        assert loader is validation_loader
        events.append(("validate", None))
        return {"auc": 0.7, "loss": 0.3}, None, None

    monkeypatch.setattr(engine, "train_epoch", train)
    monkeypatch.setattr(engine, "evaluate_loader", evaluate)
    engine.fit(
        torch.nn.Linear(1, 2),
        original_loader,
        validation_loader,
        {"device": "cpu", "training": {"epochs": 3, "early_stopping_patience": 0}},
        tmp_path,
        type("Logger", (), {"info": lambda *args, **kwargs: None})(),
        epoch_train_loader=epoch_loader,
    )
    assert events == [
        event
        for epoch in (1, 2, 3)
        for event in (("draw", epoch), ("train", epoch), ("validate", None))
    ]


def test_disabled_leakage_never_builds_external_training_views(tmp_path, monkeypatch):
    config = _external_config()
    config["training"]["leakage_simulation"] = {"enabled": False}
    config["evaluation"]["external_after_training"] = False
    monkeypatch.setattr(
        workflows, "LeakageEpochLoader", lambda *args: pytest.fail("Leakage is disabled")
    )
    monkeypatch.setattr(workflows, "set_seed", lambda *args: None)
    monkeypatch.setattr(
        workflows, "loader_from_config", lambda root, split, training: split["name"]
    )
    monkeypatch.setattr(workflows, "build_model", lambda config: None)

    def fit(*args, **kwargs):
        assert "epoch_train_loader" not in kwargs
        return dict(model=None, criterion=None, device="cpu", best_epoch=1, validation={"auc": 0.8})

    monkeypatch.setattr(workflows, "fit", fit)
    workflows.run_training(config, run_directory=tmp_path)


def test_leakage_api_cannot_bypass_disabled_external_access(tmp_path, monkeypatch):
    config = _external_config()
    config["training"]["leakage_simulation"] = {"enabled": True}
    monkeypatch.setattr(
        workflows, "loader_from_config", lambda *args: pytest.fail("Must reject before data access")
    )
    with pytest.raises(workflows.ConfigError, match="external access is disabled"):
        workflows.run_training(config, run_directory=tmp_path, evaluate_external=False)


@pytest.mark.parametrize(
    "module_name, function_name", [("hpo", "run_hpo"), ("two_stage", "run_two_stage")]
)
def test_leakage_is_rejected_by_hpo_and_two_stage_public_apis(module_name, function_name):
    from importlib import import_module

    module = import_module(f"petct.{module_name}")
    config = _external_config()
    config["training"]["leakage_simulation"] = {"enabled": True}
    with pytest.raises(workflows.ConfigError, match="not supported"):
        getattr(module, function_name)(config)
